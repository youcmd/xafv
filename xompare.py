#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from zimtohrli import mos_from_signals

def load_audio(filepath, target_channels=None):
    ext = os.path.splitext(filepath)[1].lower()
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    success = False

    # 1. Use opusdec ONLY if target_channels is 2 (opusdec forces stereo output)
    if ext == ".opus":
        try:
            cmd = ["opusdec", "--rate", "48000", "--no-dither", "--float", filepath, temp_wav_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # 2. Try soundfile for standard formats if channel count matches
    if not success:
        try:
            y, sr = sf.read(filepath, always_2d=True)
            if sr == 48000 and (target_channels is None or y.shape[1] == target_channels):
                y = y.T.astype(np.float32)
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                return y
        except Exception:
            pass

    # 3. Universal multi-channel fallback via ffmpeg (handles mono, 5.1, 7.1, etc.)
    cmd = [
        "ffmpeg", "-y", "-i", filepath, "-vn", "-sn", "-dn",
        "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
        "-f", "wav", "-c:a", "pcm_f32le", "-map_metadata", "-1"
    ]
    if target_channels is not None:
        cmd.extend(["-ac", str(target_channels)])
    cmd.append(temp_wav_path)

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    y, sr = sf.read(temp_wav_path, always_2d=True)
    y = y.T.astype(np.float32)

    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    return y

def calculate_non_linear_score(mos, bitrate, threshold=4.75):
    threshold_penalty = np.where(mos < threshold, np.exp(threshold - mos) - 1.0, 0.0)
    bitrate_cost = 0.04 * np.log(bitrate + 1.0)
    final_score = mos - bitrate_cost - threshold_penalty
    return final_score

def get_bitrate(filepath):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        data = json.loads(result.stdout)
        return (
            float(data["format"]["bit_rate"]) / 1000.0
            if "bit_rate" in data["format"]
            else 0.0
        )
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a lossy audio file against a reference using Zimtohrli (supports multi-channel audio and long chunks)."
        )
    )
    parser.add_argument("lossy", help="Path to the lossy/test audio file")
    parser.add_argument("ref", help="Path to the reference audio file")
    parser.add_argument("--chunk_duration", type=float, default=20.0, help="Chunk duration in seconds (default: 20s)")
    parser.add_argument("--skip_duration", type=float, default=20.0, help="Skip duration after each chunk in seconds (default: 20s)")
    args = parser.parse_args()

    # Load reference first to dynamically determine channel count
    ref_audio = load_audio(args.ref)
    num_channels = ref_audio.shape[0]

    # Load lossy audio forcing it to match reference channel count via ffmpeg
    test_audio = load_audio(args.lossy, target_channels=num_channels)

    min_len = min(ref_audio.shape[1], test_audio.shape[1])
    ref_audio = ref_audio[:, :min_len]
    test_audio = test_audio[:, :min_len]

    sr = 48000
    chunk_samples = int(args.chunk_duration * sr)
    skip_samples = int(args.skip_duration * sr)
    step_samples = chunk_samples + skip_samples
    
    channel_scores = [[] for _ in range(num_channels)]

    for start in range(0, min_len, step_samples):
        end = min(start + chunk_samples, min_len)
        
        if end - start < sr * 0.5:
            continue

        # Evaluate all channels concurrently for the current chunk
        with ThreadPoolExecutor(max_workers=num_channels) as executor:
            futures = [
                executor.submit(mos_from_signals, ref_audio[c, start:end], test_audio[c, start:end])
                for c in range(num_channels)
            ]
            for c, future in enumerate(futures):
                channel_scores[c].append(future.result())

    mean_channel_scores = [float(np.mean(scores)) if scores else 0.0 for scores in channel_scores]
    score = float(np.mean(mean_channel_scores)) if mean_channel_scores else 0.0
    
    bitrate = get_bitrate(args.lossy)
    final = calculate_non_linear_score(score, bitrate)

    print(f"Score:{score:.6f}", end="\t")
    print(f"kbps:{bitrate:.3f}", end="\t")
    print(f"{final:.6f}")

if __name__ == "__main__":
    main()
