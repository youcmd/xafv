import os, time
import subprocess
from glob import glob
from pymediainfo import MediaInfo
import soundfile as sf
import importlib
import math
import uuid
import tempfile
import json
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

def get_adaptive_samples(filename, sample_duration=20, max_samples=8):
    # Scale num_samples based on duration (e.g., 1 sample every 20 seconds)
    # Adjust the divisor (20) if you want samples denser or sparser
    total_duration = float(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(filename)]).stdout)
    calculated_samples = int(total_duration) // 30
    # print(total_duration)
    
    # Ensure it is at least 1, and capped at max_samples (8)
    return max(1, min(max_samples, calculated_samples))

def get_audio_bitrate(filename):
    packets = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                              "-show_entries", "packet=size", "-of", "json", str(filename)]).stdout)
    duration = float(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(filename)]).stdout)
    size = sum(int(p["size"]) for p in packets["packets"] if "size" in p)
    return size * 8 / duration / 1000

# def get_audio_bitrate(filepath: str) -> float:
#     """
#     Return audio bitrate in kbps as a float.
#     - Try ffprobe JSON stream metadata first.
#     - Fallback: extract audio stream to a temp file and compute bytes/duration.
#     - Returns 0.0 on failure.
#     """
#     ffprobe = shutil.which("ffprobe")
#     ffmpeg = shutil.which("ffmpeg")

#     # 1) Try ffprobe JSON
#     if ffprobe:
#         try:
#             cmd = [
#                 ffprobe, "-v", "quiet",
#                 "-print_format", "json",
#                 "-show_streams", "-show_format",
#                 filepath
#             ]
#             out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
#             data = json.loads(out)

#             # stream-level bit_rate (bits/sec)
#             for s in data.get("streams", []):
#                 if s.get("codec_type") == "audio":
#                     if s.get("bit_rate"):
#                         return float(s["bit_rate"]) / 1000.0
#                     tags = s.get("tags") or {}
#                     if "BPS" in tags:
#                         try:
#                             return float(tags["BPS"]) / 1000.0
#                         except Exception:
#                             pass

#             # format-level bit_rate or duration
#             fmt = data.get("format", {}) or {}
#             if fmt.get("bit_rate"):
#                 return float(fmt["bit_rate"]) / 1000.0
#             duration = float(fmt.get("duration", 0.0)) if fmt.get("duration") else 0.0
#         except Exception:
#             duration = 0.0
#     else:
#         duration = 0.0

#     # 2) Fallback: extract audio stream and compute bytes/duration
#     if ffmpeg and duration > 0.0:
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mka")
#         tmp_path = tmp.name
#         tmp.close()
#         try:
#             cmd = [
#                 ffmpeg, "-y", "-i", filepath,
#                 "-vn", "-sn", "-dn", "-map", "0:a:0",
#                 "-c:a", "copy", tmp_path
#             ]
#             subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#             if os.path.exists(tmp_path):
#                 size_bytes = os.path.getsize(tmp_path)
#                 if size_bytes > 0:
#                     kbps = (size_bytes * 8) / (duration * 1000.0)
#                     return float(kbps)
#         except Exception:
#             pass
#         finally:
#             try:
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)
#             except Exception:
#                 pass

#     # 3) Last resort: parse ffmpeg -i stderr for "bitrate: X kb/s"
#     if ffmpeg:
#         try:
#             proc = subprocess.run([ffmpeg, "-i", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
#             text = proc.stderr
#             import re
#             m = re.search(r"bitrate:\s*([\d\.]+)\s*kb/s", text)
#             if m:
#                 return float(m.group(1))
#         except Exception:
#             pass

#     return 0.0

# def extract_multiple_samples(input_file, num_samples=3, duration=15):
#     """Extracts multiple sample timestamps evenly spaced across the audio file."""
#     # Get total duration via ffprobe
#     cmd = [
#         "ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_file
#     ]
#     result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#     total_duration = float(result.stdout.strip())

#     # Avoid picking very edges (start/end silence); slice safely between 10% and 90%
#     start_margin = total_duration * 0.1
#     end_margin = total_duration * 0.9
#     usable_duration = end_margin - start_margin

#     if usable_duration < duration * num_samples:
#         # Fallback for short audio files: just take one sample from the middle
#         return [max(0.0, (total_duration / 2) - (duration / 2))]

#     step = usable_duration / (num_samples + 1)
#     timestamps = [start_margin + (i * step) for i in range(1, num_samples + 1)]
#     return timestamps

def extract_multiple_samples(input_file, num_samples=3, duration=15):
    """Extracts samples evenly spread across the entire audio track."""

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    total_duration = float(result.stdout.strip())

    # Ensure the sample duration does not exceed the audio duration
    if total_duration <= duration:
        return [0.0]
    
    if num_samples == 1:
        return [max(0.0, (total_duration - duration) / 2)]

    # Spread sample start times across the entire track
    timestamps = np.linspace(
        0,
        total_duration - duration,
        num_samples
    )

    return timestamps.tolist()

def _process_single_sample(args_tuple):
    """Worker function to process a single timestamp safely in a thread."""
    i, ts, input_file, bitrate, duration = args_tuple

    # Use unique filenames per index to avoid thread collisions
    ref_sample = f"temp_ref_{i}.wav"
    test_encoded = f"temp_test_{bitrate}k_{i}.opus"

    try:
        # 1. Extract reference chunk
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(ts), "-i", input_file, "-t", str(duration), "-vn", "-sn", "-dn",
            # "-af", "aresample=48000:dither_method=none:osf=flt",
            "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
            "-map_metadata", "-1", "-map_metadata:s:a", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
            "-c:a", "pcm_f32le", ref_sample
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Encode test chunk at target bitrate
        subprocess.run([
            "opusenc", "--quiet", "--bitrate", str(bitrate), ref_sample, test_encoded
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Call your metric script programmatically
        cmd = ["python", "/content/py/xompare.py", test_encoded, ref_sample, "--skip_duration", "0"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse tab-separated output line
        parts = result.stdout.strip().split("\t")
        metrics = {}
        for part in parts:
            if ":" in part:
                key, val = part.split(":")
                metrics[key] = float(val)

        return {
            "score": metrics.get("Score", 0.0),
            "min_score": metrics.get("min", 0.0),
            "kbps": metrics.get("kbps", 0.0),
            "final": metrics.get("final", 0.0)
        }

    finally:
        # Guaranteed cleanup of temp files for this index
        for f in [ref_sample, test_encoded]:
            if os.path.exists(f):
                os.remove(f)

def evaluate_multiple_samples(input_file, bitrate, timestamps, duration=15):
    all_results = {
        "scores": [],
        "min_scores": [],
        "kbps": [],
        "finals": []
    }

    # Package tasks for thread pool
    tasks = [(i, ts, input_file, bitrate, duration) for i, ts in enumerate(timestamps)]

    # Run across threads (set max_workers carefully based on your vCPU count)
    max_workers = min(len(timestamps), os.cpu_count() or 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_sample, task) for task in tasks]

        for future in as_completed(futures):
            res = future.result()
            if res:
                all_results["scores"].append(res["score"])
                all_results["min_scores"].append(res["min_score"])
                all_results["kbps"].append(res["kbps"])
                all_results["finals"].append(res["final"])

    return {
        "mean_score": float(np.mean(all_results["scores"])) if all_results["scores"] else 0.0,
        # "mean_min_score": float(np.mean(all_results["min_scores"])) if all_results["min_scores"] else 0.0,
        "lowest_min_score": float(np.min(all_results["min_scores"])) if all_results["min_scores"] else 0.0,
        "mean_kbps": float(np.mean(all_results["kbps"])) if all_results["kbps"] else 0.0,
        "mean_final": float(np.mean(all_results["finals"])) if all_results["finals"] else 0.0,
    }

def calculate_target_mos(target_mos: float, current_bitrate: int) -> float:
    if current_bitrate <= 48:
        return target_mos + 0.04
    elif current_bitrate <= 56:
        return target_mos + 0.02
    elif current_bitrate <= 64:
        return target_mos + 0.01
    elif current_bitrate >= 128:
        return target_mos - 0.02
    elif current_bitrate >= 96:
        return target_mos - 0.01
    else:
        return target_mos

def find_optimal_bitrate(input_file, target_mos=4.6):
    # bitrate_ladder = list(range(32, 128, 8))
    bitrate_ladder = (list(range(32, 64, 8)) +
                      list(range(64, 96, 4)) +
                      list(range(96, 129, 2)))
                    #   list(range(128, 161, 2)))

    num_samples = get_adaptive_samples(input_file, sample_duration=20)
    timestamps = extract_multiple_samples(input_file, num_samples=num_samples, duration=20)

    results = {}       # bitrate -> mean score
    min_scores = {}    # bitrate -> minimum score
    bitrates = {}    # bitrate -> minimum score

    low = 0
    high = len(bitrate_ladder) - 1
    target_index = high

    # 1. Binary search to find the first bitrate meeting the target MOS
    while low <= high:
        mid = (low + high) // 2
        current_bitrate = bitrate_ladder[mid]

        if current_bitrate not in results:
            print(f"[*] Testing bitrate: {current_bitrate} kbps across {len(timestamps)} samples...", end="\t")

            metrics_dict = evaluate_multiple_samples(input_file, current_bitrate, timestamps)

            min_score = metrics_dict["lowest_min_score"]
            mean_final = metrics_dict["mean_final"]
            mean_score = metrics_dict["mean_score"]
            mean_kbps = metrics_dict["mean_kbps"]

            results[current_bitrate] = mean_score
            min_scores[current_bitrate] = min_score
            bitrates[current_bitrate] = mean_kbps

            print(f"    -> Mean Score: {mean_score:.4f} | Min: {min_score:.4f} | {mean_final:.4f} | {mean_kbps:.3f}")
        else:
            mean_score = results[current_bitrate]
            min_score = min_scores[current_bitrate]

        # Determine effective target based on bitrate tier
        # current_target_mos = calculate_target_mos(target_mos, mean_kbps)
        current_target_mos = target_mos

        # # Minimum score requirement
        # min_target = (4.88 if mean_kbps <= 64 
        #               else 4.87 if mean_kbps < 96
        #               else 4.86)
        min_target = 4.86

        # Minimum score is a hard requirement
        if min_score < min_target:
            low = mid + 1

        # Both mean score and minimum score must pass
        elif mean_score >= current_target_mos:
            target_index = mid
            high = mid - 1

        # Otherwise we need a higher bitrate
        else:
            low = mid + 1

    # 2. Ensure we test the immediate neighbor below the candidate
    # for the plateau / diminishing returns check.
    candidate_bitrate = bitrate_ladder[target_index]
    candidate_idx = target_index

    if candidate_idx > 0:
        prev_bitrate = bitrate_ladder[candidate_idx - 1]

        if prev_bitrate not in results:
            print(f"[*] Testing neighbor bitrate: {prev_bitrate} kbps for plateau check...")

            metrics_dict = evaluate_multiple_samples(input_file, prev_bitrate, timestamps)

            min_score = metrics_dict["lowest_min_score"]
            mean_final = metrics_dict["mean_final"]
            mean_score = metrics_dict["mean_score"]
            mean_kbps = metrics_dict["mean_kbps"]

            results[prev_bitrate] = mean_score
            min_scores[prev_bitrate] = min_score
            bitrates[prev_bitrate] = mean_kbps

            print(f"    -> Mean Score: {mean_score:.4f} | Min: {min_score:.4f} | {mean_final:.4f} | {mean_kbps:.3f}")

    # 3. Sequential plateau / diminishing returns check
    # Only drop down if the previous bitrate also passes both targets.
    tested_bitrates = sorted([b for b in bitrate_ladder if b in results])

    for i in range(1, len(tested_bitrates)):
        prev_b = tested_bitrates[i - 1]
        curr_b = tested_bitrates[i]

        if bitrate_ladder.index(curr_b) - bitrate_ladder.index(prev_b) == 1:
            score_diff = results[curr_b] - results[prev_b]
            prev_bitrate = bitrates.get(prev_b, 0.0)

            # Use the same target calculation as the binary search
            # prev_target = calculate_target_mos(target_mos, prev_bitrate)
            prev_target = target_mos

            prev_min_target = 4.86

            # prev_min_target = (
            #     4.88 if prev_bitrate <=64
            #     else 4.87 if prev_bitrate <96
            #     else 4.86
            # )

            prev_min_score = min_scores.get(prev_b, 0.0)

            # Only accept the plateau fallback if the lower bitrate
            # passes both mean-score and minimum-score requirements.
            if curr_b >= 96 and score_diff < 0.005 and results[prev_b] >= prev_target and prev_min_score >= prev_min_target:
                if prev_b <= candidate_bitrate:
                    print(f"[*] Plateau detected! Diminishing returns after {prev_b} kbps (gain was only {score_diff:.4f}).")
                    print(f"[+] Optimal bitrate found: {prev_b} kbps")
                    return prev_b

    print(f"[+] Optimal bitrate found: {candidate_bitrate} kbps")
    return candidate_bitrate
