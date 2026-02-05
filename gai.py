import os
import subprocess
from pymediainfo import MediaInfo

def get_audio_info(input_file, o=False):
    media_info = MediaInfo.parse(input_file)
    audio_track = next((t for t in media_info.tracks if t.track_type == "Audio"), None)
    general_track = next((t for t in media_info.tracks if t.track_type == "General"), None)

    if not audio_track:
        raise ValueError("No audio track found")
      
    path_wo_ext, ext = get_path_without_ext(input_file)
    file_size_bytes = int(general_track.file_size) if general_track and general_track.file_size else 0

    duration_ms = float(audio_track.duration) if audio_track.duration else 0
    duration_secs = duration_ms / 1000

    stream_size = audio_track.stream_size or (general_track.file_size if general_track else None)
    if stream_size and duration_secs > 0:
        calc_bitrate = int((int(stream_size) * 8) / duration_secs)
    else:
        calc_bitrate = int(audio_track.bit_rate) if audio_track.bit_rate else None
      
    sample_rate_value = int(audio_track.sampling_rate) if audio_track.sampling_rate else 48000
    base_sample_rate = get_base_sample_rate(sample_rate_value)
    bit_depth_value = "N/A (Lossy)" if audio_track.compression_mode == "Lossy" else int(audio_track.bit_depth or audio_track.bit_resolution or 16)

    info = {
        "file_size": file_size_bytes,
        "size_mb": f"{round(file_size_bytes / (1024 * 1024), 2)} MB",
        "bitrate": calc_bitrate,
        "kbps": f"{round(calc_bitrate / 1000)}kbps" if calc_bitrate else "N/A",
        "length": duration_secs,
        "codec": audio_track.format,
        "sample_rate": sample_rate_value,
        "base_sample_rate": base_sample_rate,
        "bit_depth": bit_depth_value,
        "channels": int(audio_track.channel_s) if audio_track.channel_s else 2,
        "bitrate_mode": audio_track.bit_rate_mode,
        "ext": ext,
        "path_wo_ext": path_wo_ext
    }

    if o:
        print(f"--- Metadata for: {input_file} ---")
        for key, value in info.items():
            print(f"{key}: {value}")
          
    return info

def get_filename(path):
    filename_with_ext = os.path.basename(path)
    filename, ext = os.path.splitext(filename_with_ext)
    return filename, ext

def get_path_without_ext(path):
    filename, ext = get_filename(path)
    dir_path = os.path.dirname(path)
    return os.path.join(dir_path, filename), ext

def verify(input_file):
    result = subprocess.run(['flac', '-t', input_file], capture_output=True)
    if result.returncode == 0:
        print(f'Successfully encoded and verified: {input_file}')
    else:
        print(f'Failed to verify: {input_file}')

def get_base_sample_rate(rate: int) -> int:
    if rate < 44100:
        return rate
    if rate % 44100 == 0:
        return 44100
    elif rate % 48000 == 0:
        return 48000
    else:
        return 48000
