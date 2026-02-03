#!/usr/bin/env python3
"""
Extract audio from a video, pick a non-solid frame at 10% of duration,
and embed that image as cover art into the extracted audio (M4A/Opus).
Requires: av, mutagen, pillow, numpy
Install: pip install av mutagen pillow numpy
"""

import sys
import os
import av
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggopus import OggOpus
from mutagen.flac import Picture

# -------------------------
# Audio extraction (remux)
# -------------------------
def extract_audio_pure_python(input_video, output_folder="."):
    """Remux first audio stream from input_video into an output file and return its path."""
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input not found: {input_video}")

    container = av.open(input_video)
    try:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in input file.")

        codec_name = audio_stream.codec_context.name
        extension_map = {
            "aac": "m4a", "mp3": "mp3", "flac": "flac",
            "opus": "opus", "vorbis": "ogg", "alac": "m4a",
            "pcm_s16le": "wav"
        }
        ext = extension_map.get(codec_name, codec_name)
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        output_filename = os.path.join(output_folder, f"{base_name}.{ext}")

        out_container = av.open(output_filename, mode="w")
        try:
            # create output stream with codec name (positional arg)
            out_stream = out_container.add_stream(codec_name)

            # try to set only safe attributes; skip read-only ones
            src_cc = audio_stream.codec_context
            dst_cc = out_stream.codec_context
            for attr in ("sample_rate", "channel_layout",):
                val = getattr(src_cc, attr, None)
                if val is not None:
                    try:
                        setattr(dst_cc, attr, val)
                    except Exception:
                        pass
            try:
                out_stream.time_base = audio_stream.time_base
            except Exception:
                pass

            # remux packets
            for packet in container.demux(audio_stream):
                if packet.dts is None:
                    continue
                packet.stream = out_stream
                out_container.mux(packet)
        finally:
            out_container.close()
    finally:
        container.close()

    return output_filename

# -------------------------
# Frame extraction + solid check
# -------------------------
def is_solid_color_image(pil_img, tolerance=5, unique_color_threshold=10):
    arr = np.asarray(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    sample = arr
    if h * w > 500_000:
        sample = arr[::4, ::4]
    unique_colors = np.unique(sample.reshape(-1, 3), axis=0)
    if unique_colors.shape[0] <= unique_color_threshold:
        return True
    stds = sample.reshape(-1, 3).std(axis=0)
    if np.all(stds <= tolerance):
        return True
    return False

def extract_non_solid_frame(input_path, percent=0.1, output_path=None,
                            image_format="png", max_attempts=7, step_seconds=0.5,
                            tolerance=5, unique_color_threshold=10):
    """Extract a non-solid frame near percent of duration. Returns saved image path."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not (0.0 <= percent <= 1.0):
        raise ValueError("percent must be between 0.0 and 1.0")

    container = av.open(input_path)
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("No video stream found in input file.")

        # Determine duration
        duration_seconds = None
        try:
            if container.duration is not None:
                duration_seconds = float(container.duration) / float(av.time_base)
        except Exception:
            duration_seconds = None
        if duration_seconds is None and getattr(stream, "duration", None) is not None:
            duration_seconds = float(stream.duration) * float(stream.time_base)
        if duration_seconds is None:
            raise ValueError("Could not determine video duration; try ffprobe or provide timestamp.")

        target_time = duration_seconds * percent

        # offsets: 0, -step, +step, -2*step, +2*step, ...
        offsets = [0.0]
        for i in range(1, max_attempts):
            sign = -1 if i % 2 == 1 else 1
            mult = (i + 1) // 2
            offsets.append(sign * mult * step_seconds)

        base = os.path.splitext(os.path.basename(input_path))[0]
        first_attempt_path = None
        saved_path = None

        for i, off in enumerate(offsets[:max_attempts]):
            ts = target_time + off
            ts = max(0.0, min(ts, duration_seconds - 1e-3))
            time_base_seconds = float(stream.time_base)
            seek_ts = int(ts / time_base_seconds)
            try:
                container.seek(seek_ts, any_frame=False, stream=stream)
            except Exception:
                pass

            for frame in container.decode(stream):
                img = frame.to_image()
                if output_path:
                    out = output_path
                else:
                    pct = int(percent * 100)
                    out = f"{base}_{pct}pct_try{i+1}.{image_format}"
                img.save(out)
                if first_attempt_path is None:
                    first_attempt_path = out
                
                if not is_solid_color_image(img, tolerance=tolerance, unique_color_threshold=unique_color_threshold):
                    saved_path = out
                else:
                    if out != first_attempt_path:
                        try:
                            os.remove(out)
                        except Exception:
                            pass
                break

            if saved_path:
                break

        if saved_path is None:
            if first_attempt_path is None:
                raise RuntimeError("No non-solid frame found within attempted timestamps.")
            saved_path = first_attempt_path

        return saved_path
    finally:
        container.close()

# -------------------------
# Embed cover (M4A / Opus)
# -------------------------
def _read_and_optionally_resize(image_path, max_size=None, quality=90):
    with Image.open(image_path) as im:
        fmt = im.format or "JPEG"
        mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else f"image/{fmt.lower()}"
        if max_size:
            w, h = im.size
            longest = max(w, h)
            if longest > max_size:
                scale = max_size / float(longest)
                new_size = (int(w * scale), int(h * scale))
                im = im.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        save_kwargs = {"format": fmt}
        if fmt.upper() in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        im.save(buf, **save_kwargs)
        data = buf.getvalue()
        width, height = im.size
    return data, mime, width, height

def embed_cover(audio_path, image_path, max_image_side=None, picture_type=3, description="Cover (front)"):
    """Embed image_path into audio_path (supports M4A/MP4 and .opus). Returns audio_path."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    data, mime, width, height = _read_and_optionally_resize(image_path, max_size=max_image_side)
    ext = os.path.splitext(audio_path)[1].lower()
    is_mp4 = ext in (".m4a", ".mp4", ".m4b", ".m4r")
    is_opus = ext == ".opus"

    if is_mp4:
        mp4 = MP4(audio_path)
        fmt = MP4Cover.FORMAT_JPEG if mime == "image/jpeg" else MP4Cover.FORMAT_PNG
        cover = MP4Cover(data, imageformat=fmt)
        mp4.tags["covr"] = [cover]
        mp4.save()
        return audio_path

    if is_opus:
        opus = OggOpus(audio_path)
        pic = Picture()
        pic.data = data
        pic.type = picture_type
        pic.mime = mime
        pic.desc = description
        pic.width = int(width)
        pic.height = int(height)
        pic.depth = 24
        pic_data = pic.write()
        b64 = base64.b64encode(pic_data).decode("ascii")
        tags = opus.tags or {}
        tags["metadata_block_picture"] = [b64]
        opus.tags = tags
        opus.save()
        return audio_path

    # fallback: try mutagen detection
    from mutagen import File as MutagenFile
    mf = MutagenFile(audio_path)
    if isinstance(mf, MP4):
        fmt = MP4Cover.FORMAT_JPEG if mime == "image/jpeg" else MP4Cover.FORMAT_PNG
        cover = MP4Cover(data, imageformat=fmt)
        mf.tags["covr"] = [cover]
        mf.save()
        return audio_path
    if isinstance(mf, OggOpus):
        pic = Picture()
        pic.data = data
        pic.type = picture_type
        pic.mime = mime
        pic.desc = description
        pic.width = int(width)
        pic.height = int(height)
        pic.depth = 24
        b64 = base64.b64encode(pic.write()).decode("ascii")
        tags = mf.tags or {}
        tags["metadata_block_picture"] = [b64]
        mf.tags = tags
        mf.save()
        return audio_path

    raise ValueError("Unsupported audio format. Only M4A/MP4 and Opus are supported.")

# -------------------------
# Main CLI
# -------------------------
def is_video_with_audio(path):
    """Return (has_video, has_audio)."""
    if not os.path.isfile(path):
        return False, False
    try:
        c = av.open(path)
    except Exception:
        return False, False
    try:
        has_video = any(s.type == "video" for s in c.streams)
        has_audio = any(s.type == "audio" for s in c.streams)
        return has_video, has_audio
    finally:
        c.close()
        
def is_supported_for_embedding(audio_path):
    ext = os.path.splitext(audio_path)[1].lower()
    return ext in (".m4a", ".mp4", ".m4b", ".m4r", ".opus")

if __name__ == "__main__":
    import traceback

    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <video_file> [--max-image-side N]")
        sys.exit(1)

    video_in = sys.argv[1]
    max_image_side = None
    if "--max-image-side" in sys.argv:
        try:
            idx = sys.argv.index("--max-image-side")
            max_image_side = int(sys.argv[idx + 1])
        except Exception:
            max_image_side = None

    try:
        if not os.path.isfile(video_in):
            print(f"Error: file not found: {video_in}")
            sys.exit(2)

        has_video, has_audio = is_video_with_audio(video_in)
        if not has_video:
            print("Error: input does not contain a video stream.")
            sys.exit(3)
        if not has_audio:
            print("Error: input does not contain an audio stream.")
            sys.exit(4)

        print("Extracting audio...")
        audio_out = extract_audio_pure_python(video_in)
        print("Audio saved to:", audio_out)
        
        if is_supported_for_embedding(audio_out):
            print("Extracting non-solid frame at 10%...")
            image_out = extract_non_solid_frame(video_in, percent=0.1)
            print("Image saved to:", image_out)
            
            print("Embedding cover into audio...")
            embed_cover(audio_out, image_out, max_image_side=max_image_side)
            print("Done. Cover embedded into:", audio_out)

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        sys.exit(10)

