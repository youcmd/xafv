import os
import time
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

XOMPARE_SCRIPT = "/content/py/xompare.py"


def run(cmd):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )


def get_audio_duration(filename):
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filename)
    ])

    duration = float(result.stdout.strip())

    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {filename}")

    return duration


def get_adaptive_samples(filename, sample_duration=10, max_samples=12):
    total_duration = get_audio_duration(filename)
    samples = int(total_duration // sample_duration)

    return max(1, min(max_samples, samples))


def extract_multiple_samples(input_file, num_samples=4, duration=10):
    total_duration = get_audio_duration(input_file)
    duration = min(float(duration), total_duration)
    num_samples = max(1, int(num_samples))

    if num_samples == 1 or total_duration <= duration:
        return [max(0.0, (total_duration - duration) / 2)]

    # Distribute samples evenly without exceeding audio boundaries.
    return np.linspace(
        0,
        total_duration - duration,
        num_samples
    ).tolist()


def _process_single_sample(args):
    i, timestamp, input_file, bitrate, duration = args

    ref_sample = f"temp_ref_{os.getpid()}_{i}.wav"
    test_encoded = f"temp_test_{os.getpid()}_{bitrate}k_{i}.opus"

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", input_file,
            "-t", str(duration),
            "-vn", "-sn", "-dn",
            "-af",
            "aresample=48000:resampler=soxr:cutoff=1:"
            "precision=33:dither_method=none:osf=flt",
            "-map_metadata", "-1",
            "-map_metadata:s:a", "-1",
            "-fflags", "+bitexact",
            "-flags:a", "+bitexact",
            "-c:a", "pcm_f32le",
            ref_sample
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        subprocess.run([
            "opusenc", "--quiet",
            "--bitrate", str(bitrate),
            ref_sample,
            test_encoded
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        result = run([
            "python",
            XOMPARE_SCRIPT,
            test_encoded,
            ref_sample,
            "--skip_duration", "0"
        ])

        metrics = {}

        for part in result.stdout.strip().split("\t"):
            if ":" in part:
                key, value = part.split(":", 1)

                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    pass

        if "Score" not in metrics or "min" not in metrics:
            raise ValueError(
                f"Could not parse xompare output: {result.stdout}"
            )

        return {
            "score": metrics["Score"],
            "min_score": metrics["min"],
            "kbps": metrics.get("kbps", 0.0),
            "final": metrics.get("final", 0.0)
        }

    finally:
        for filename in (ref_sample, test_encoded):
            if os.path.exists(filename):
                os.remove(filename)


def evaluate_multiple_samples(input_file, bitrate, timestamps, duration=10):
    if not timestamps:
        raise ValueError("At least one sample timestamp is required.")

    tasks = [
        (i, ts, input_file, bitrate, duration)
        for i, ts in enumerate(timestamps)
    ]

    results = []

    with ThreadPoolExecutor(
        max_workers=min(len(tasks), os.cpu_count() or 2)
    ) as executor:

        futures = [
            executor.submit(_process_single_sample, task)
            for task in tasks
        ]

        for future in as_completed(futures):
            results.append(future.result())

    scores = [r["score"] for r in results]
    min_scores = [r["min_score"] for r in results]
    kbps = [r["kbps"] for r in results]
    finals = [r["final"] for r in results]

    return {
        "mean_score": float(np.mean(scores)),
        "lowest_min_score": float(np.min(min_scores)),
        "mean_kbps": float(np.mean(kbps)),
        "mean_final": float(np.mean(finals))
    }


def calculate_target_mos(target_mos, current_bitrate):
    if current_bitrate <= 48:
        return target_mos + 0.04

    if current_bitrate <= 56:
        return target_mos + 0.02

    if current_bitrate <= 64:
        return target_mos + 0.01

    if current_bitrate >= 128:
        return target_mos - 0.02

    if current_bitrate >= 96:
        return target_mos - 0.01

    return target_mos


def find_optimal_bitrate(
    input_file,
    target_mos=4.75,
    sample_duration=10,
    num_samples=None,
    max_samples=12
):
    bitrate_ladder = (
        list(range(32, 64, 8))
        + list(range(64, 96, 4))
        + list(range(96, 128, 2))
        + list(range(128, 137, 2))
    )

    # Adapt even when num_samples is explicitly provided.
    requested_samples = max_samples if num_samples is None else num_samples

    num_samples = get_adaptive_samples(
        input_file,
        sample_duration,
        requested_samples
    )

    timestamps = extract_multiple_samples(
        input_file,
        num_samples,
        sample_duration
    )

    print(
        f"[*] Duration: {get_audio_duration(input_file):.1f}s | "
        f"Sample duration: {sample_duration}s | "
        f"Samples: {len(timestamps)}"
    )

    results = {}
    min_scores = {}
    bitrates = {}

    def evaluate(bitrate):
        if bitrate not in results:
            print(
                f"[*] Testing {bitrate} kbps "
                f"({len(timestamps)} samples)...",
                end=" ",
                flush=True
            )

            metrics = evaluate_multiple_samples(
                input_file,
                bitrate,
                timestamps,
                sample_duration
            )

            results[bitrate] = metrics["mean_score"]
            min_scores[bitrate] = metrics["lowest_min_score"]
            bitrates[bitrate] = metrics["mean_kbps"]

            print(
                f"Mean: {metrics['mean_score']:.4f} | "
                f"Min: {metrics['lowest_min_score']:.4f} | "
                f"kbps: {metrics['mean_kbps']:.2f} | "
                f"final: {metrics['mean_final']:.4f}"
            )

        return results[bitrate], min_scores[bitrate]

    def passes(bitrate):
        mean_score, min_score = evaluate(bitrate)

        actual_kbps = bitrates[bitrate]

        min_target = (
            4.88 if actual_kbps <= 64
            else 4.87 if actual_kbps < 96
            else 4.86
        )

        current_target = calculate_target_mos(
            target_mos,
            actual_kbps
        )

        return (
            mean_score >= current_target
            and min_score >= min_target
        )

    # Binary search for the lowest bitrate meeting both thresholds.
    low = 0
    high = len(bitrate_ladder) - 1
    candidate = None

    while low <= high:
        mid = (low + high) // 2
        bitrate = bitrate_ladder[mid]

        if passes(bitrate):
            candidate = bitrate
            high = mid - 1
        else:
            low = mid + 1

    # ==========================================================
    # TARGET NOT REACHED: CHECK FOR A PLATEAU
    # ==========================================================
    if candidate is None:
        print("[!] No tested bitrate met both quality thresholds.")

        # Evaluate the maximum bitrate if it has not been tested.
        max_bitrate = bitrate_ladder[-1]
        evaluate(max_bitrate)

        if not passes(max_bitrate):
            print(
                "[!] Maximum bitrate failed. "
                "Checking for plateau..."
            )

            # Only consider bitrates that have actually been tested.
            tested = sorted(results)

            plateau_bitrate = None

            for prev_b, curr_b in zip(tested, tested[1:]):

                # Only compare adjacent entries in the bitrate ladder.
                if (
                    bitrate_ladder.index(curr_b)
                    != bitrate_ladder.index(prev_b) + 1
                ):
                    continue

                # Only check for a plateau at 96 kbps and above.
                if curr_b < 96:
                    continue

                score_diff = (
                    results[curr_b] - results[prev_b]
                )

                # Plateau: mean MOS gain is >= 0 and < 0.005.
                if 0 <= score_diff < 0.005:
                    plateau_bitrate = prev_b

                    print(
                        f"[*] Plateau detected: "
                        f"{prev_b} -> {curr_b} kbps "
                        f"(gain: {score_diff:.4f})"
                    )

                    break

            if plateau_bitrate is not None:
                print(
                    f"[+] Plateau bitrate: "
                    f"{plateau_bitrate} kbps"
                )

                return plateau_bitrate

            # No plateau: choose the best measured mean MOS.
            print(
                "[!] No plateau detected. "
                "Selecting the tested bitrate with the highest mean MOS."
            )

            best_bitrate = max(
                results,
                key=lambda b: results[b]
            )

            print(
                f"[+] Best measured bitrate: {best_bitrate} kbps | "
                f"Mean MOS: {results[best_bitrate]:.4f} | "
                f"Min MOS: {min_scores[best_bitrate]:.4f}"
            )

            return best_bitrate

    # ==========================================================
    # TARGET REACHED: SELECT THE LOWEST PASSING BITRATE
    # ==========================================================
    candidate_index = bitrate_ladder.index(candidate)

    if candidate_index > 0:
        previous = bitrate_ladder[candidate_index - 1]

        if passes(previous):
            candidate = previous

    print(f"[+] Selected bitrate: {candidate} kbps")

    return candidate


def benchmark_sampling(input_file, target_mos=4.91, configs=None):
    if configs is None:
        configs = [
            (duration, samples)
            for duration in (10, 15, 20, 30)
            for samples in (6, 8, 10, 12)
        ]

    benchmark_results = []

    for duration, samples in configs:
        print(
            f"\n=== Benchmark: "
            f"{duration}s x {samples} samples ==="
        )

        start = time.perf_counter()

        bitrate = find_optimal_bitrate(
            input_file,
            target_mos=target_mos,
            sample_duration=duration,
            num_samples=samples
        )

        elapsed = time.perf_counter() - start

        actual_samples = get_adaptive_samples(
            input_file,
            duration,
            samples
        )

        result = {
            "sample_duration": duration,
            "num_samples": samples,
            "actual_samples": actual_samples,
            "total_sample_seconds": duration * actual_samples,
            "bitrate": bitrate,
            "elapsed_seconds": round(elapsed, 2)
        }

        benchmark_results.append(result)

        print(
            f"[BENCHMARK] Bitrate: {bitrate} kbps | "
            f"Time: {elapsed:.2f}s"
        )

    print("\n=== BENCHMARK SUMMARY ===")

    print(
        "Duration | Requested | Actual | "
        "Sample seconds | Bitrate | Time"
    )

    for r in benchmark_results:
        print(
            f"{r['sample_duration']:>8}s | "
            f"{r['num_samples']:>9} | "
            f"{r['actual_samples']:>6} | "
            f"{r['total_sample_seconds']:>14}s | "
            f"{r['bitrate']:>7} kbps | "
            f"{r['elapsed_seconds']:>5}s"
        )

    return benchmark_results
