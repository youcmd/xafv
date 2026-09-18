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

def load_audio_stereo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    success = False

    # 1. If .opus, try opusdec first
    if ext == ".opus":
        try:
            cmd = ["opusdec", "--rate", "48000", "--force-stereo", "--no-dither", "--float", filepath, temp_wav_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                success = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # 2. Default: try soundfile first
    if not success:
        try:
            y, sr = sf.read(filepath, always_2d=True)
            # Ensure 48kHz stereo format matching requirements
            if sr == 48000 and y.shape[1] == 2:
                y = y.T.astype(np.float32)
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                return y
            else:
                raise ValueError("Resampling/rechanneling needed via fallback")
        except Exception:
            pass

    # 3. Fallback to ffmpeg
    if not success:
        cmd = [
            "ffmpeg", "-y", "-i", filepath, "-vn", "-sn", "-dn",
            "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
            "-ac", "2", "-f", "wav", "-c:a", "pcm_f32le", "-map_metadata", "-1", temp_wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    y, sr = sf.read(temp_wav_path, always_2d=True)
    y = y.T.astype(np.float32)

    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

    return y

def calculate_non_linear_score(mos, bitrate, threshold=4.75):
    # 1. Hard/Exponential penalty if MOS drops below the 4.75 threshold
    threshold_penalty = np.where(mos < threshold, np.exp(threshold - mos) - 1.0, 0.0)
    
    # 2. Logarithmic bitrate penalty (diminishing penalty for higher bitrates)
    # This prevents bloated bitrates from unfairly dominating efficient codecs
    bitrate_cost = 0.04 * np.log(bitrate + 1.0)
    
    # 3. Final non-linear formulation
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
            "Compare a lossy audio file against a reference using Zimtohrli (supports long audio via chunking and skipping)."
        )
    )
    parser.add_argument("lossy", help="Path to the lossy/test audio file")
    parser.add_argument("ref", help="Path to the reference audio file")
    parser.add_argument("--chunk_duration", type=float, default=20.0, help="Chunk duration in seconds (default: 20s)")
    parser.add_argument("--skip_duration", type=float, default=20.0, help="Skip duration after each chunk in seconds (default: 20s)")
    args = parser.parse_args()

    # Load reference and test audio concurrently using threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        ref_future = executor.submit(load_audio_stereo, args.ref)
        test_future = executor.submit(load_audio_stereo, args.lossy)
        ref_stereo = ref_future.result()
        test_stereo = test_future.result()

    ref_l, ref_r = ref_stereo[0], ref_stereo[1]
    test_l, test_r = test_stereo[0], test_stereo[1]

    min_len = min(len(ref_l), len(test_l))
    ref_l, test_l = ref_l[:min_len], test_l[:min_len]
    ref_r, test_r = ref_r[:min_len], test_r[:min_len]

    sr = 48000
    chunk_samples = int(args.chunk_duration * sr)
    skip_samples = int(args.skip_duration * sr)
    step_samples = chunk_samples + skip_samples
    
    scores_l = []
    scores_r = []

    # Process audio in chunks with skipping for long audio optimization
    for start in range(0, min_len, step_samples):
        end = min(start + chunk_samples, min_len)
        
        # Skip trailing fragments smaller than 0.5 seconds
        if end - start < sr * 0.5:
            continue

        chunk_ref_l = ref_l[start:end]
        chunk_test_l = test_l[start:end]
        chunk_ref_r = ref_r[start:end]
        chunk_test_r = test_r[start:end]

        # Evaluate Left and Right channels concurrently for the current chunk
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_l = executor.submit(mos_from_signals, chunk_ref_l, chunk_test_l)
            future_r = executor.submit(mos_from_signals, chunk_ref_r, chunk_test_r)
            scores_l.append(future_l.result())
            scores_r.append(future_r.result())

    scoreL = float(np.mean(scores_l)) if scores_l else 0.0
    scoreR = float(np.mean(scores_r)) if scores_r else 0.0
    score = (scoreL + scoreR) / 2.0
    bitrate = get_bitrate(args.lossy)
    # final = score/bitrate*int(bitrate)
    final = calculate_non_linear_score(score,bitrate)

    print(f"Score:{score:.6f}", end="\t")
    print(f"kbps:{bitrate:.3f}", end="\t")
    print(f"{final:.6f}")

if __name__ == "__main__":
    main()
