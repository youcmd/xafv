import os
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

XOMPARE_SCRIPT = "/content/py/xompare.py"
MAX_WORKERS = 6


def run(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, check=True
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


def get_adaptive_samples(filename, sample_duration=10, max_samples=12,
                         total_duration=None):
    if total_duration is None:
        total_duration = get_audio_duration(filename)

    return max(1, min(max_samples, int(total_duration // sample_duration)))


def extract_multiple_samples(input_file, num_samples=4, duration=10,
                             total_duration=None):
    if total_duration is None:
        total_duration = get_audio_duration(input_file)

    duration = min(float(duration), total_duration)
    num_samples = max(1, int(num_samples))

    if num_samples == 1 or total_duration <= duration:
        return [max(0.0, (total_duration - duration) / 2)]

    return np.linspace(0, total_duration - duration, num_samples).tolist()


def _extract_reference(args):
    i, timestamp, input_file, duration, cache_dir = args
    ref_sample = os.path.join(cache_dir, f"ref_{i}.wav")

    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", input_file,
        "-t", str(duration),
        "-vn", "-sn", "-dn",
        "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
        "-map_metadata", "-1",
        "-map_metadata:s:a", "-1",
        "-fflags", "+bitexact",
        "-flags:a", "+bitexact",
        "-c:a", "pcm_f32le",
        ref_sample
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    return ref_sample


def _process_single_sample(args):
    i, ref_sample, bitrate = args
    test_encoded = f"{ref_sample[:-4]}_{bitrate}k.opus"

    try:
        subprocess.run([
            "opusenc", "--quiet",
            "--bitrate", str(bitrate),
            ref_sample,
            test_encoded
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        result = run([
            "python", XOMPARE_SCRIPT,
            test_encoded, ref_sample,
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
            raise ValueError(f"Could not parse xompare output: {result.stdout}")

        return {
            "score": metrics["Score"],
            "min_score": metrics["min"],
            "kbps": metrics.get("kbps", 0.0),
            "final": metrics.get("final", 0.0)
        }

    finally:
        if os.path.exists(test_encoded):
            os.remove(test_encoded)


def evaluate_multiple_samples(input_file, bitrate, timestamps, duration=10,
                              reference_samples=None):
    if not timestamps:
        raise ValueError("At least one sample timestamp is required.")

    if reference_samples is None:
        raise ValueError("Reference samples must be extracted before evaluation.")

    tasks = [
        (i, ref_sample, bitrate)
        for i, ref_sample in enumerate(reference_samples)
    ]

    results = []
    with ThreadPoolExecutor(
        max_workers=min(len(tasks), MAX_WORKERS)
    ) as executor:
        futures = [executor.submit(_process_single_sample, task) for task in tasks]
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

    total_duration = get_audio_duration(input_file)
    requested_samples = max_samples if num_samples is None else num_samples
    num_samples = get_adaptive_samples(
        input_file, sample_duration, requested_samples, total_duration
    )
    timestamps = extract_multiple_samples(
        input_file, num_samples, sample_duration, total_duration
    )

    print(
        f"[*] Duration: {total_duration:.1f}s | "
        f"Sample duration: {sample_duration}s | "
        f"Samples: {len(timestamps)}"
    )

    results, min_scores, bitrates = {}, {}, {}
    cache_dir = f"temp_refs_{os.getpid()}"
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with ThreadPoolExecutor(
            max_workers=min(len(timestamps), MAX_WORKERS)
        ) as executor:
            reference_samples = list(executor.map(
                _extract_reference,
                [
                    (i, ts, input_file, sample_duration, cache_dir)
                    for i, ts in enumerate(timestamps)
                ]
            ))

        def evaluate(bitrate):
            if bitrate not in results:
                print(
                    f"[*] Testing {bitrate} kbps "
                    f"({len(timestamps)} samples)...",
                    end=" ", flush=True
                )

                metrics = evaluate_multiple_samples(
                    input_file, bitrate, timestamps, sample_duration,
                    reference_samples=reference_samples
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
            mean_score, _ = evaluate(bitrate)
            return mean_score >= calculate_target_mos(
                target_mos, bitrates[bitrate]
            )

        def interpolate_bitrate(b1, b2):
            """Estimate the target crossing between a failing and passing bitrate."""
            d1 = results[b1] - calculate_target_mos(target_mos, bitrates[b1])
            d2 = results[b2] - calculate_target_mos(target_mos, bitrates[b2])

            if d1 >= 0 or d2 < 0 or d1 == d2:
                return None

            estimate = b1 - d1 * (b2 - b1) / (d2 - d1)

            choices = [
                b for b in bitrate_ladder
                if b1 < b < b2 and b not in results
            ]

            return min(choices, key=lambda b: abs(b - estimate)) if choices else None

        # ==========================================================
        # SEARCH: INTERPOLATION + BINARY-SEARCH FALLBACK
        # ==========================================================
        fail_idx = None
        pass_idx = None
        candidate = None
        ladder_size = len(bitrate_ladder)

        while True:
            # Keep the measured failing/passing boundaries independently
            # of the next search interval.
            if fail_idx is not None and pass_idx is not None:
                if fail_idx >= pass_idx:
                    # Measurements contradict the monotonic-curve assumption.
                    # Keep the lowest passing bitrate found.
                    candidate = bitrate_ladder[pass_idx]
                    break

                if pass_idx - fail_idx == 1:
                    candidate = bitrate_ladder[pass_idx]
                    break

                low_idx, high_idx = fail_idx + 1, pass_idx - 1

                # Interpolate inside the known failing/passing bracket.
                bitrate = interpolate_bitrate(
                    bitrate_ladder[fail_idx],
                    bitrate_ladder[pass_idx]
                )

            elif pass_idx is None:
                low_idx = fail_idx + 1 if fail_idx is not None else 0
                high_idx = ladder_size - 1
                bitrate = None

            else:
                low_idx, high_idx = 0, pass_idx - 1
                bitrate = None

                if high_idx < low_idx:
                    candidate = bitrate_ladder[pass_idx]
                    break

            # If interpolation cannot choose a useful point, use binary search.
            if bitrate is None:
                untested = [
                    i for i in range(low_idx, high_idx + 1)
                    if bitrate_ladder[i] not in results
                ]

                if not untested:
                    if pass_idx is not None:
                        candidate = bitrate_ladder[pass_idx]
                    break

                midpoint = (low_idx + high_idx) // 2
                idx = min(untested, key=lambda i: abs(i - midpoint))
                bitrate = bitrate_ladder[idx]

            passed = passes(bitrate)
            idx = bitrate_ladder.index(bitrate)

            if passed:
                pass_idx = idx if pass_idx is None else min(pass_idx, idx)
            else:
                fail_idx = idx if fail_idx is None else max(fail_idx, idx)

        # ==========================================================
        # TARGET NOT REACHED: CHECK FOR A PLATEAU
        # ==========================================================
        if candidate is None:
            print("[!] No tested bitrate met both quality thresholds.")
            max_bitrate = bitrate_ladder[-1]

            if not passes(max_bitrate):
                print(
                    "[!] Maximum bitrate failed. "
                    "Checking the upper bitrate range for a plateau..."
                )

                upper_range = bitrate_ladder[-7:]

                for bitrate in upper_range:
                    evaluate(bitrate)

                plateau_bitrate = None

                for prev_b, curr_b in zip(upper_range, upper_range[1:]):
                    score_diff = results[curr_b] - results[prev_b]

                    if 0 <= score_diff < 0.005:
                        plateau_bitrate = prev_b
                        print(
                            f"[*] Plateau detected: {prev_b} -> {curr_b} kbps "
                            f"(gain: {score_diff:.4f})"
                        )
                        break

                if plateau_bitrate is not None:
                    print(
                        f"[+] Plateau bitrate: {plateau_bitrate} kbps | "
                        f"Mean MOS: {results[plateau_bitrate]:.4f} | "
                        f"Min MOS: {min_scores[plateau_bitrate]:.4f}"
                    )
                    return plateau_bitrate

                print(
                    "[!] No plateau detected. "
                    "Selecting the tested bitrate with the highest mean MOS."
                )

                best_bitrate = max(results, key=lambda b: results[b])

                print(
                    f"[+] Fallback selection: {best_bitrate} kbps | "
                    f"Actual: {bitrates[best_bitrate]:.2f} kbps | "
                    f"Mean MOS: {results[best_bitrate]:.4f} | "
                    f"Min MOS: {min_scores[best_bitrate]:.4f}"
                )

                return best_bitrate

            candidate = max_bitrate

        print(f"[+] Selected bitrate: {candidate} kbps")
        return candidate

    finally:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


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
