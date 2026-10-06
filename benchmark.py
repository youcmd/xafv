#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from zimtohrli import mos_from_signals

# Change these if your binaries are elsewhere.
FFMPEG_FDK = "/content/sample_data/ffmpeg"
FFMPEG_NMR = "/root/ffmpeg"
FAAC = "faac"
OPUSENC = "opusenc"

VBR_LEVELS = [2, 3, 4, 5]
ITERATIONS = 12
FAAC_MIN, FAAC_MAX = 6, 1000
NMR_MIN, NMR_MAX = 0.1, 5.0
OPUS_MIN, OPUS_MAX = 6.0, 256.0
AUDIO_EXTENSIONS = {".wav", ".wv", ".flac", ".m4a", ".mp3", ".opus", ".aiff", ".aif"}


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def bitrate(filename):
    packets = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                              "-show_entries", "packet=size", "-of", "json", str(filename)]).stdout)
    duration = float(run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                          "-show_entries", "stream=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(filename)]).stdout)
    size = sum(int(p["size"]) for p in packets["packets"] if "size" in p)
    return size * 8 / duration / 1000


def make_wav(source, output):
    run([FFMPEG_FDK, "-hide_banner", "-y", "-v", "quiet", "-i", str(source),
         "-map", "0:a:0", "-af",
         "lowpass=20000,aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
         "-vn", "-sn", "-dn", "-c:a", "pcm_f32le", "-map_metadata", "-1",
         "-map_metadata:s:a", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
         "-f", "wav", str(output)])


def fdk(wav, output, vbr):
    run([FFMPEG_FDK, "-hide_banner", "-y", "-i", str(wav), "-vn",
         "-c:a", "libfdk_aac", "-profile:a", "aac_low", "-vbr", str(vbr),
         "-cutoff", "20000", "-map_metadata", "-1", "-movflags", "+faststart",
         "-f", "mp4", str(output)])


def faac(wav, output, q, joint):
    run([FAAC, "-q", str(round(q)), "-c", "20000", "--joint", str(joint),
         "--object-type", "lc", "--overwrite", "-o", str(output), str(wav)])


def nmr(wav, output, q):
    run([FFMPEG_NMR, "-hide_banner", "-y", "-i", str(wav), "-vn",
         "-c:a", "aac", "-aac_coder", "nmr", "-q:a", str(q), "-cutoff", "20000",
         "-map_metadata", "-1", "-movflags", "+faststart", "-f", "mp4", str(output)])


def opus(wav, output, target):
    run([OPUSENC, "--bitrate", f"{target:.3f}", "--no-phase-inv", str(wav), str(output)])


def search(name, test, low, high, target):
    low_bitrate = test(low)
    high_bitrate = test(high)
    increasing = high_bitrate > low_bitrate

    best_q = low
    best_bitrate = low_bitrate
    best_error = abs(low_bitrate - target)

    if abs(high_bitrate - target) < best_error:
        best_q, best_bitrate, best_error = high, high_bitrate, abs(high_bitrate - target)

    seen = set()
    crossed = False

    for _ in range(ITERATIONS):
        q = round((low + high) / 2, 3)
        if q == low or q == high:
            break

        value = test(q)
        error = abs(value - target)
        print(f"  {name}: {q:.3f} -> {value:.3f} kbps")

        if error < best_error:
            best_q, best_bitrate, best_error = q, value, error

        key = round(value, 3)
        if key in seen and crossed:
            break
        seen.add(key)

        if increasing:
            if value < target:
                low = q
            else:
                high = q
            crossed |= value >= target
        else:
            if value > target:
                low = q
            else:
                high = q
            crossed |= value <= target

    print(f"  {name}: best={best_q:.3f}, bitrate={best_bitrate:.3f}")
    return best_q, best_bitrate


def calibrate(wav, temp, vbr):
    fdk_test = temp / "fdk.m4a"
    fdk(wav, fdk_test, vbr)
    target = bitrate(fdk_test)
    print(f"FDK VBR {vbr}: {target:.3f} kbps")

    def test_faac_j1(q):
        out = temp / "faac_j1.m4a"
        out.unlink(missing_ok=True)
        try:
            faac(wav, out, q, 1)
            return bitrate(out)
        finally:
            out.unlink(missing_ok=True)

    def test_faac_j3(q):
        out = temp / "faac_j3.m4a"
        out.unlink(missing_ok=True)
        try:
            faac(wav, out, q, 3)
            return bitrate(out)
        finally:
            out.unlink(missing_ok=True)

    def test_nmr(q):
        out = temp / "nmr.m4a"
        out.unlink(missing_ok=True)
        try:
            nmr(wav, out, q)
            return bitrate(out)
        finally:
            out.unlink(missing_ok=True)

    def test_opus(q):
        out = temp / "opus.opus"
        out.unlink(missing_ok=True)
        try:
            opus(wav, out, q)
            return bitrate(out)
        finally:
            out.unlink(missing_ok=True)

    searches = [
        ("FAAC J1", test_faac_j1, FAAC_MIN, FAAC_MAX),
        ("FAAC J3", test_faac_j3, FAAC_MIN, FAAC_MAX),
        ("NMR", test_nmr, NMR_MIN, NMR_MAX),
        ("Opus", test_opus, OPUS_MIN, OPUS_MAX),
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(search, *x, target): x[0] for x in searches}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    faac_j1_q, faac_j1_bitrate = results["FAAC J1"]
    faac_j3_q, faac_j3_bitrate = results["FAAC J3"]
    nmr_q, nmr_bitrate = results["NMR"]
    opus_target, opus_actual = results["Opus"]

    return {
        "fdk_vbr": vbr,
        "target_bitrate": target,
        "faac_j1_q": round(faac_j1_q),
        "faac_j1_bitrate": faac_j1_bitrate,
        "faac_j3_q": round(faac_j3_q),
        "faac_j3_bitrate": faac_j3_bitrate,
        "nmr_q": nmr_q,
        "nmr_bitrate": nmr_bitrate,
        "opus_bitrate": opus_target,
        "opus_bitrate_actual": opus_actual,
    }, fdk_test


def encode_one(source, output_root, vbr):
    stem = source.stem
    song_dir = output_root / stem
    song_dir.mkdir(parents=True, exist_ok=True)

    wav = song_dir / f"{stem}_standardized.wav"
    calibration_file = song_dir / f"calibration_vbr{vbr}.json"

    if not wav.exists():
        print(f"Standardizing {source.name}...")
        make_wav(source, wav)

    outputs = {
        "fdk": song_dir / f"{stem}_fdk_vbr{vbr}.m4a",
        "faac_j1": song_dir / f"{stem}_faac_j1_vbr{vbr}.m4a",
        "faac_j3": song_dir / f"{stem}_faac_j3_vbr{vbr}.m4a",
        "nmr": song_dir / f"{stem}_nmr_vbr{vbr}.m4a",
        "opus": song_dir / f"{stem}_opus_vbr{vbr}.opus",
    }

    if all(p.exists() for p in outputs.values()) and calibration_file.exists():
        print(f"{source.name}: VBR {vbr} already encoded, skipping.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)

        if calibration_file.exists():
            calibration = json.loads(calibration_file.read_text())
            fdk_test = temp / "fdk.m4a"
            fdk(wav, fdk_test, vbr)
        else:
            calibration, fdk_test = calibrate(wav, temp, vbr)
            calibration_file.write_text(json.dumps({
                "source": str(source),
                "standardized_wav": str(wav),
                **calibration
            }, indent=2))

        if not outputs["fdk"].exists():
            shutil.copy2(fdk_test, outputs["fdk"])

        if not outputs["faac_j1"].exists():
            faac(wav, outputs["faac_j1"], calibration["faac_j1_q"], 1)

        if not outputs["faac_j3"].exists():
            faac(wav, outputs["faac_j3"], calibration["faac_j3_q"], 3)

        if not outputs["nmr"].exists():
            nmr(wav, outputs["nmr"], calibration["nmr_q"])

        if not outputs["opus"].exists():
            opus(wav, outputs["opus"], calibration["opus_bitrate"])

    print(f"{source.name}: VBR {vbr} complete.")


def encode(input_dir, output_dir):
    sources = sorted(p for p in Path(input_dir).iterdir()
                     if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not sources:
        raise RuntimeError(f"No audio files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        print(f"\n{'=' * 70}\n{source.name}\n{'=' * 70}")
        for vbr in VBR_LEVELS:
            encode_one(source, output_dir, vbr)


def load_audio(filepath, target_channels=None):
    ext = Path(filepath).suffix.lower()
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp.name
    temp.close()
    success = False

    if ext == ".opus":
        try:
            subprocess.run(["opusdec", "--rate", "48000", "--no-dither", "--float",
                            str(filepath), temp_path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            success = os.path.exists(temp_path) and os.path.getsize(temp_path) > 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    if not success:
        try:
            y, sr = sf.read(filepath, always_2d=True)
            if sr == 48000 and (target_channels is None or y.shape[1] == target_channels):
                os.remove(temp_path)
                return y.T.astype(np.float32)
        except Exception:
            pass

    cmd = [FFMPEG_NMR, "-y", "-i", str(filepath), "-vn", "-sn", "-dn",
           "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
           "-f", "wav", "-c:a", "pcm_f32le", "-map_metadata", "-1"]
    if target_channels is not None:
        cmd += ["-ac", str(target_channels)]
    cmd.append(temp_path)

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    y, sr = sf.read(temp_path, always_2d=True)
    os.remove(temp_path)
    return y.T.astype(np.float32)


def compare_file(lossy, reference):
    ref_audio = load_audio(reference)
    test_audio = load_audio(lossy, ref_audio.shape[0])

    min_len = min(ref_audio.shape[1], test_audio.shape[1])
    ref_audio = ref_audio[:, :min_len]
    test_audio = test_audio[:, :min_len]

    sr = 48000
    chunk = 20 * sr
    skip = 20 * sr
    step = chunk + skip
    channel_scores = [[] for _ in range(ref_audio.shape[0])]

    for start in range(0, min_len, step):
        end = min(start + chunk, min_len)
        if end - start < sr * 0.5:
            continue

        with ThreadPoolExecutor(max_workers=ref_audio.shape[0]) as pool:
            futures = [pool.submit(mos_from_signals, ref_audio[c, start:end],
                                    test_audio[c, start:end])
                       for c in range(ref_audio.shape[0])]
            for c, future in enumerate(futures):
                channel_scores[c].append(future.result())

    means = [float(np.mean(x)) for x in channel_scores if x]
    all_scores = [s for x in channel_scores for s in x]
    score = float(np.mean(means)) if means else 0.0
    min_score = float(np.min(all_scores)) if all_scores else 0.0
    kbps = bitrate(lossy)

    threshold_penalty = np.exp(5 - score) - 1 if score < 5 else 0
    final = score - 0.04 * np.log(kbps + 1) - threshold_penalty
    reencode = int(min_score < 4.75 and kbps >= 64)

    return {
        "score": score,
        "min_score": min_score,
        "bitrate": kbps,
        "final": final,
        "reencode": reencode,
    }


def compare(output_dir):
    output_dir = Path(output_dir)
    result_file = output_dir / "results.json"
    results = json.loads(result_file.read_text()) if result_file.exists() else {}

    for song_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        wavs = list(song_dir.glob("*_standardized.wav"))
        if not wavs:
            continue
        reference = wavs[0]
        stem = reference.stem.replace("_standardized", "")

        for vbr in VBR_LEVELS:
            calibration_file = song_dir / f"calibration_vbr{vbr}.json"
            if not calibration_file.exists():
                continue

            key_prefix = f"{stem}|vbr{vbr}"
            files = {
                "fdk": song_dir / f"{stem}_fdk_vbr{vbr}.m4a",
                "faac_j1": song_dir / f"{stem}_faac_j1_vbr{vbr}.m4a",
                "faac_j3": song_dir / f"{stem}_faac_j3_vbr{vbr}.m4a",
                "nmr": song_dir / f"{stem}_nmr_vbr{vbr}.m4a",
                "opus": song_dir / f"{stem}_opus_vbr{vbr}.opus",
            }

            for encoder, file in files.items():
                key = f"{key_prefix}|{encoder}"
                if key in results:
                    print(f"Skipping {key}")
                    continue
                if not file.exists():
                    print(f"Missing {file}")
                    continue

                print(f"Comparing {stem} / VBR {vbr} / {encoder}...")
                r = compare_file(file, reference)
                results[key] = {
                    "song": stem,
                    "fdk_vbr": vbr,
                    "encoder": encoder,
                    "file": str(file),
                    **r,
                }
                result_file.write_text(json.dumps(results, indent=2))

    save_csv(results, output_dir / "results.csv")
    print(f"Saved {len(results)} comparisons.")


def save_csv(results, filename):
    fields = ["song", "fdk_vbr", "encoder", "bitrate", "score",
              "min_score", "final", "reencode", "file"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results.values())


def graph(output_dir):
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    df = pd.read_csv(output_dir / "results.csv")
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)

    # Average score vs bitrate.
    avg = df.groupby(["encoder", "fdk_vbr"], as_index=False).agg(
        bitrate=("bitrate", "mean"),
        score=("score", "mean"),
        min_score=("min_score", "mean")
    )

    plt.figure(figsize=(10, 6))
    for encoder, g in avg.groupby("encoder"):
        g = g.sort_values("bitrate")
        plt.plot(g.bitrate, g.score, "o-", label=encoder)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("Average Zimtohrli score")
    plt.title("Average quality vs bitrate")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / "score_vs_bitrate.png", dpi=150)
    plt.close()

    # Minimum score vs bitrate.
    plt.figure(figsize=(10, 6))
    for encoder, g in avg.groupby("encoder"):
        g = g.sort_values("bitrate")
        plt.plot(g.bitrate, g.min_score, "o-", label=encoder)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("Average minimum score")
    plt.title("Worst-case quality vs bitrate")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_dir / "min_score_vs_bitrate.png", dpi=150)
    plt.close()

    # NMR advantage over FDK.
    pivot = df.pivot_table(index=["song", "fdk_vbr"], columns="encoder",
                           values="score", aggfunc="mean").reset_index()
    if "nmr" in pivot and "fdk" in pivot:
        pivot["nmr_minus_fdk"] = pivot["nmr"] - pivot["fdk"]
        avg_adv = pivot.groupby("fdk_vbr")["nmr_minus_fdk"].mean()

        plt.figure(figsize=(8, 5))
        plt.axhline(0, color="black", linewidth=1)
        plt.plot(avg_adv.index, avg_adv.values, "o-", linewidth=2)
        plt.xticks(VBR_LEVELS)
        plt.xlabel("FDK VBR level")
        plt.ylabel("NMR score − FDK score")
        plt.title("NMR advantage over FDK")
        plt.grid(alpha=.25)
        plt.tight_layout()
        plt.savefig(graph_dir / "nmr_vs_fdk.png", dpi=150)
        plt.close()

    print(f"Graphs saved to {graph_dir}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("encode")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("benchmark_results"))

    p = sub.add_parser("compare")
    p.add_argument("output", type=Path)

    p = sub.add_parser("graph")
    p.add_argument("output", type=Path)

    p = sub.add_parser("all")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("benchmark_results"))

    args = parser.parse_args()

    if args.command == "encode":
        encode(args.input, args.output)
    elif args.command == "compare":
        compare(args.output)
    elif args.command == "graph":
        graph(args.output)
    elif args.command == "all":
        encode(args.input, args.output)
        compare(args.output)
        graph(args.output)


if __name__ == "__main__":
    main()
