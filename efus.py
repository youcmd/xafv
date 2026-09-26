#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pymediainfo import MediaInfo
from zimtohrli import mos_from_signals


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 48000

DEFAULT_CHUNK_DURATION = 20.0
DEFAULT_SKIP_DURATION = 20.0

BITRATE_LADDER = list(range(56, 129, 8))


# ============================================================
# External executable handling
# ============================================================

def find_executable(name: str) -> str:
    """
    Find an external executable in PATH.

    Works on Windows (.exe), macOS and Linux.
    """
    path = shutil.which(name)

    if path:
        return path

    raise RuntimeError(
        f"Required executable '{name}' was not found in PATH.\n"
        f"Please install it and make sure it is available from the command line."
    )


def run_command(
    command: list[str],
    *,
    capture_output: bool = False,
    quiet: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Cross-platform subprocess runner.

    IMPORTANT:
    We intentionally do NOT use shell=True.
    """

    if not quiet:
        print("$", " ".join(str(x) for x in command))

    try:
        return subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            check=check,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Executable not found: {command[0]}"
        ) from exc

    except subprocess.CalledProcessError as exc:
        if capture_output and exc.stderr:
            print(exc.stderr, file=sys.stderr)

        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}:\n"
            f"{' '.join(command)}"
        ) from exc


def check_dependencies(codec: Optional[str] = None):
    """
    Verify required external tools.
    """

    required = {
        "ffmpeg": find_executable("ffmpeg"),
        "ffprobe": find_executable("ffprobe"),
    }

    if codec == "opus":
        required["opusenc"] = find_executable("opusenc")

    elif codec == "flac":
        required["flac"] = find_executable("flac")

    return required


# ============================================================
# Media information
# ============================================================

def get_audio_info(input_file: str | Path):
    input_file = Path(input_file)

    media_info = MediaInfo.parse(str(input_file))

    audio_track = next(
        (t for t in media_info.tracks if t.track_type == "Audio"),
        None,
    )

    general_track = next(
        (t for t in media_info.tracks if t.track_type == "General"),
        None,
    )

    if not audio_track or not general_track:
        return None

    bit_depth = (
        int(audio_track.bit_depth)
        if audio_track.bit_depth
        else 0
    )

    is_float = (
        audio_track.format_settings_endianness == "Float"
        or audio_track.format_profile == "Float"
    )

    duration_ms = (
        float(general_track.duration)
        if general_track.duration
        else 0
    )

    file_size = (
        int(general_track.file_size)
        if general_track.file_size
        else 0
    )

    stream_size = (
        int(audio_track.stream_size)
        if audio_track.stream_size
        else file_size
    )

    if stream_size and duration_ms > 0:
        calculated_bitrate = (
            stream_size * 8
        ) / (duration_ms / 1000)
    else:
        calculated_bitrate = (
            int(audio_track.bit_rate)
            if audio_track.bit_rate
            else 0
        )

    if is_float:
        sample_fmt = "flt"

    elif bit_depth in (24, 32):
        sample_fmt = f"s{bit_depth}"

    else:
        sample_fmt = (
            f"s{bit_depth}"
            if bit_depth
            else "unknown"
        )

    return {
        "sample_rate": (
            int(audio_track.sampling_rate)
            if audio_track.sampling_rate
            else 0
        ),
        "sample_fmt": sample_fmt,
        "bit_depth": bit_depth,
        "file_size": file_size,
        "kbps": (
            f"{round(calculated_bitrate / 1000)}kbps"
            if calculated_bitrate > 0
            else "N/A"
        ),
    }


def get_base_sample_rate(rate: int) -> int:
    """
    88.2 -> 44.1
    96   -> 48
    176.4 -> 44.1
    192 -> 48
    """

    if rate < 44100:
        return rate

    if rate % 44100 == 0:
        return 44100

    if rate % 48000 == 0:
        return 48000

    return 48000


def db_to_percent(db: float) -> float:
    return round(10 ** (db / 20), 4)


# ============================================================
# Audio conversion
# ============================================================

def make_temp_wav(
    input_file: str | Path,
    *,
    output_rate: int,
    sample_format: str,
    preamp: float = 0.0,
) -> Path:

    input_file = Path(input_file)

    temp_dir = Path(tempfile.mkdtemp(prefix="efus_"))
    output_file = temp_dir / "audio.wav"

    ffmpeg = find_executable("ffmpeg")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-vn",
        "-sn",
        "-dn",
    ]

    if preamp != 0:
        command += [
            "-af",
            f"volume={preamp}dB",
        ]

    command += [
        "-ar",
        str(output_rate),
        "-acodec",
        sample_format,
        "-map_metadata",
        "-1",
        str(output_file),
    ]

    try:
        run_command(command)

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty WAV file.")

        return output_file

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def cleanup_temp_wav(wav: Path):
    shutil.rmtree(wav.parent, ignore_errors=True)


def convert_opus(
    input_file: str | Path,
    output_file: str | Path,
    bitrate: int,
    preamp: float = 0.0,
    phase_inv: str = "scan",
):
    """
    Convert input -> temporary 48 kHz float WAV -> opusenc.

    phase_inv:
        on   = allow phase inversion
        off  = disable phase inversion
        scan = currently treated as off unless a scanner is added
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    check_dependencies("opus")

    info = get_audio_info(input_file)

    if not info:
        raise RuntimeError(
            f"Could not read audio information: {input_file}"
        )

    # Opus always works internally at 48 kHz.
    wav = make_temp_wav(
        input_file,
        output_rate=48000,
        sample_format="pcm_f32le",
        preamp=preamp,
    )

    try:
        opusenc = find_executable("opusenc")

        command = [
            opusenc,
            "--quiet",
            "--bitrate",
            str(bitrate),
        ]

        if phase_inv == "off":
            command.append("--no-phase-inv")

        elif phase_inv == "scan":
            # Your original code had the scanner commented out.
            # To preserve the effective current behavior,
            # scan currently disables phase inversion.
            command.append("--no-phase-inv")

        command += [
            str(wav),
            str(output_file),
        ]

        run_command(command)

    finally:
        cleanup_temp_wav(wav)


def convert_flac(
    input_file: str | Path,
    output_file: str | Path,
    bit_depth: int = 16,
    preamp: float = 0.0,
):
    """
    Convert input -> temporary WAV -> FLAC.

    This intentionally avoids SoX shell pipelines so that the
    program behaves consistently on Windows/macOS/Linux.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    check_dependencies("flac")

    info = get_audio_info(input_file)

    if not info:
        raise RuntimeError(
            f"Could not read audio information: {input_file}"
        )

    target_rate = get_base_sample_rate(
        info["sample_rate"]
    )

    wav = make_temp_wav(
        input_file,
        output_rate=target_rate,
        sample_format=(
            "pcm_s16le"
            if bit_depth == 16
            else "pcm_s24le"
        ),
        preamp=preamp,
    )

    try:
        flac = find_executable("flac")

        command = [
            flac,
            "-8",
            "-s",
            "-V",
            "-f",
            "-o",
            str(output_file),
            str(wav),
        ]

        run_command(command)

    finally:
        cleanup_temp_wav(wav)


def convert_audio(
    codec: str,
    input_file: str | Path,
    output_file: str | Path,
    *,
    bitrate: Optional[int] = None,
    bit_depth: int = 16,
    preamp: float = 0.0,
    phase_inv: str = "scan",
):
    if codec == "opus":
        if bitrate is None:
            raise ValueError(
                "Opus conversion requires --bitrate."
            )

        convert_opus(
            input_file,
            output_file,
            bitrate,
            preamp,
            phase_inv,
        )

    elif codec == "flac":
        convert_flac(
            input_file,
            output_file,
            bit_depth,
            preamp,
        )

    else:
        raise ValueError(
            f"Unsupported codec: {codec}"
        )


# ============================================================
# Audio loading
# ============================================================

def load_audio(
    filepath: str | Path,
    target_channels: Optional[int] = None,
):
    """
    Load audio as:

        channels x samples

    at 48 kHz float32.
    """

    filepath = Path(filepath)

    # Try SoundFile first.
    try:
        y, sr = sf.read(
            str(filepath),
            always_2d=True,
        )

        if (
            sr == SAMPLE_RATE
            and (
                target_channels is None
                or y.shape[1] == target_channels
            )
        ):
            return y.T.astype(np.float32)

    except Exception:
        pass

    ffmpeg = find_executable("ffmpeg")

    temp_dir = Path(
        tempfile.mkdtemp(prefix="efus_decode_")
    )

    temp_wav = temp_dir / "decoded.wav"

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(filepath),
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "aresample=48000:dither_method=none:osf=flt",
        "-f",
        "wav",
        "-c:a",
        "pcm_f32le",
        "-map_metadata",
        "-1",
    ]

    if target_channels is not None:
        command += [
            "-ac",
            str(target_channels),
        ]

    command.append(str(temp_wav))

    try:
        run_command(command, quiet=True)

        y, sr = sf.read(
            str(temp_wav),
            always_2d=True,
        )

        if sr != SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected sample rate: {sr}"
            )

        return y.T.astype(np.float32)

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# Bitrate
# ============================================================

def get_bitrate(filepath: str | Path) -> float:
    ffprobe = find_executable("ffprobe")

    command = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(filepath),
    ]

    try:
        result = run_command(
            command,
            capture_output=True,
            quiet=True,
        )

        data = json.loads(result.stdout)

        bitrate = (
            data
            .get("format", {})
            .get("bit_rate")
        )

        if bitrate:
            return float(bitrate) / 1000.0

    except Exception:
        pass

    return 0.0


# ============================================================
# Zimtohrli comparison
# ============================================================

def calculate_non_linear_score(
    mos,
    bitrate,
    threshold=4.75,
):
    threshold_penalty = np.where(
        mos < threshold,
        np.exp(threshold - mos) - 1.0,
        0.0,
    )

    bitrate_cost = (
        0.04 * np.log(bitrate + 1.0)
    )

    return (
        mos
        - bitrate_cost
        - threshold_penalty
    )


def compare_audio(
    lossy: str | Path,
    reference: str | Path,
    *,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    skip_duration: float = DEFAULT_SKIP_DURATION,
):
    reference_audio = load_audio(reference)

    num_channels = reference_audio.shape[0]

    test_audio = load_audio(
        lossy,
        target_channels=num_channels,
    )

    min_len = min(
        reference_audio.shape[1],
        test_audio.shape[1],
    )

    reference_audio = (
        reference_audio[:, :min_len]
    )

    test_audio = (
        test_audio[:, :min_len]
    )

    chunk_samples = int(
        chunk_duration * SAMPLE_RATE
    )

    skip_samples = int(
        skip_duration * SAMPLE_RATE
    )

    step_samples = (
        chunk_samples + skip_samples
    )

    channel_scores = [
        []
        for _ in range(num_channels)
    ]

    max_workers = max(
        1,
        min(
            num_channels,
            os.cpu_count() or 2,
        ),
    )

    for start in range(
        0,
        min_len,
        step_samples,
    ):

        end = min(
            start + chunk_samples,
            min_len,
        )

        if (
            end - start
            < SAMPLE_RATE * 0.5
        ):
            continue

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            futures = {
                executor.submit(
                    mos_from_signals,
                    reference_audio[
                        c,
                        start:end,
                    ],
                    test_audio[
                        c,
                        start:end,
                    ],
                ): c
                for c in range(num_channels)
            }

            for future in as_completed(
                futures
            ):
                channel = futures[future]
                channel_scores[channel].append(
                    float(future.result())
                )

    mean_channel_scores = [
        float(np.mean(scores))
        if scores
        else 0.0
        for scores in channel_scores
    ]

    score = (
        float(np.mean(mean_channel_scores))
        if mean_channel_scores
        else 0.0
    )

    all_scores = [
        score
        for scores in channel_scores
        for score in scores
    ]

    min_score = (
        float(np.min(all_scores))
        if all_scores
        else 0.0
    )

    bitrate = get_bitrate(lossy)

    final = calculate_non_linear_score(
        score,
        bitrate,
    )

    reencode = (
        1
        if min_score < 4.75
        else 0
    )

    return {
        "score": score,
        "min_score": min_score,
        "kbps": bitrate,
        "final": float(final),
        "reencode": reencode,
    }


def print_comparison(result):
    print(
        f"Score:{result['score']:.6f}\t"
        f"min:{result['min_score']:.6f}\t"
        f"kbps:{result['kbps']:.3f}\t"
        f"final:{result['final']:.6f}\t"
        f"reencode:{result['reencode']}"
    )


# ============================================================
# Sample extraction
# ============================================================

def get_duration(filepath: str | Path) -> float:
    ffprobe = find_executable("ffprobe")

    command = [
        ffprobe,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(filepath),
    ]

    result = run_command(
        command,
        capture_output=True,
        quiet=True,
    )

    return float(result.stdout.strip())


def extract_multiple_samples(
    input_file: str | Path,
    num_samples: int = 8,
    duration: float = 20.0,
):
    total_duration = get_duration(
        input_file
    )

    if total_duration <= duration:
        return [
            max(
                0.0,
                (total_duration / 2)
                - (duration / 2),
            )
        ]

    start_margin = (
        total_duration * 0.1
    )

    end_margin = (
        total_duration * 0.9
    )

    usable_duration = (
        end_margin - start_margin
    )

    if usable_duration < (
        duration * num_samples
    ):
        return [
            max(
                0.0,
                (total_duration / 2)
                - (duration / 2),
            )
        ]

    step = (
        usable_duration
        / (num_samples + 1)
    )

    return [
        start_margin + i * step
        for i in range(
            1,
            num_samples + 1,
        )
    ]


# ============================================================
# Optimization
# ============================================================

def calculate_target_mos(
    target_mos: float,
    current_bitrate: int,
) -> float:

    if current_bitrate <= 48:
        return target_mos + 0.02

    if current_bitrate <= 56:
        return target_mos + 0.015

    if current_bitrate <= 64:
        return target_mos + 0.01

    if current_bitrate >= 128:
        return target_mos - 0.02

    if current_bitrate >= 96:
        return target_mos - 0.01

    return target_mos


def evaluate_sample(
    input_file: str | Path,
    timestamp: float,
    bitrate: int,
    duration: float,
    index: int,
):
    """
    Encode one sample and compare it directly.

    No subprocess call to another Python script.
    """

    ffmpeg = find_executable("ffmpeg")
    opusenc = find_executable("opusenc")

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"efus_sample_{index}_"
        )
    )

    reference_wav = (
        temp_dir / "reference.wav"
    )

    encoded_opus = (
        temp_dir / "encoded.opus"
    )

    try:
        # ----------------------------------------------------
        # Extract reference
        # ----------------------------------------------------

        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(input_file),
            "-t",
            str(duration),
            "-vn",
            "-sn",
            "-dn",
            "-af",
            (
                "aresample="
                "48000:"
                "resampler=soxr:"
                "cutoff=1:"
                "precision=33:"
                "dither_method=none:"
                "osf=flt"
            ),
            "-map_metadata",
            "-1",
            "-map_metadata:s:a",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-c:a",
            "pcm_f32le",
            str(reference_wav),
        ]

        run_command(
            command,
            quiet=True,
        )

        # ----------------------------------------------------
        # Encode Opus
        # ----------------------------------------------------

        command = [
            opusenc,
            "--no-phase-inv",
            "--quiet",
            "--bitrate",
            str(bitrate),
            str(reference_wav),
            str(encoded_opus),
        ]

        run_command(
            command,
            quiet=True,
        )

        # ----------------------------------------------------
        # Compare directly
        # ----------------------------------------------------

        result = compare_audio(
            encoded_opus,
            reference_wav,
            chunk_duration=duration,
            skip_duration=0,
        )

        return result

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


def evaluate_multiple_samples(
    input_file: str | Path,
    bitrate: int,
    timestamps,
    duration: float = 20.0,
):
    scores = []
    min_scores = []
    kbps = []
    finals = []

    max_workers = max(
        1,
        min(
            len(timestamps),
            os.cpu_count() or 2,
        ),
    )

    tasks = [
        (
            i,
            timestamp,
        )
        for i, timestamp
        in enumerate(timestamps)
    ]

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                evaluate_sample,
                input_file,
                timestamp,
                bitrate,
                duration,
                i,
            )
            for i, timestamp in tasks
        ]

        for future in as_completed(
            futures
        ):
            result = future.result()

            scores.append(
                result["score"]
            )

            min_scores.append(
                result["min_score"]
            )

            kbps.append(
                result["kbps"]
            )

            finals.append(
                result["final"]
            )

    return {
        "mean_score": (
            float(np.mean(scores))
            if scores
            else 0.0
        ),

        "lowest_min_score": (
            float(np.min(min_scores))
            if min_scores
            else 0.0
        ),

        "mean_kbps": (
            float(np.mean(kbps))
            if kbps
            else 0.0
        ),

        "mean_final": (
            float(np.mean(finals))
            if finals
            else 0.0
        ),
    }


def find_optimal_bitrate(
    input_file: str | Path,
    target_mos: float = 4.72,
    num_samples: int = 8,
    sample_duration: float = 20.0,
):
    timestamps = extract_multiple_samples(
        input_file,
        num_samples=num_samples,
        duration=sample_duration,
    )

    results = {}

    min_scores = {}

    low = 0
    high = len(BITRATE_LADDER) - 1

    target_index = high

    # --------------------------------------------------------
    # Binary search
    # --------------------------------------------------------

    while low <= high:

        mid = (low + high) // 2

        bitrate = BITRATE_LADDER[mid]

        if bitrate not in results:

            print(
                f"[*] Testing bitrate: "
                f"{bitrate} kbps "
                f"across {len(timestamps)} samples..."
            )

            metrics = evaluate_multiple_samples(
                input_file,
                bitrate,
                timestamps,
                sample_duration,
            )

            mean_score = metrics[
                "mean_final"
            ]

            min_score = metrics[
                "lowest_min_score"
            ]

            results[bitrate] = mean_score
            min_scores[bitrate] = min_score

            print(
                f"    -> Mean: {mean_score:.4f} "
                f"Min: {min_score:.4f}"
            )

        else:

            mean_score = results[
                bitrate
            ]

            min_score = min_scores[
                bitrate
            ]

        target = calculate_target_mos(
            target_mos,
            bitrate,
        )

        min_target = (
            4.88
            if bitrate <= 64
            else 4.87
        )

        if min_score < min_target:

            low = mid + 1

        elif (
            mean_score >= target
            and min_score >= min_target
        ):

            target_index = mid
            high = mid - 1

        else:

            low = mid + 1

    candidate = BITRATE_LADDER[
        target_index
    ]

    # --------------------------------------------------------
    # Neighbor check
    # --------------------------------------------------------

    if target_index > 0:

        previous = BITRATE_LADDER[
            target_index - 1
        ]

        if previous not in results:

            print(
                f"[*] Testing neighbor bitrate: "
                f"{previous} kbps..."
            )

            metrics = evaluate_multiple_samples(
                input_file,
                previous,
                timestamps,
                sample_duration,
            )

            results[previous] = metrics[
                "mean_final"
            ]

            min_scores[previous] = metrics[
                "lowest_min_score"
            ]

            print(
                f"    -> Mean: "
                f"{metrics['mean_final']:.4f} "
                f"Min: "
                f"{metrics['lowest_min_score']:.4f}"
            )

    # --------------------------------------------------------
    # Plateau check
    # --------------------------------------------------------

    tested = sorted(
        results.keys()
    )

    for i in range(
        1,
        len(tested),
    ):

        previous = tested[i - 1]
        current = tested[i]

        previous_index = (
            BITRATE_LADDER.index(
                previous
            )
        )

        current_index = (
            BITRATE_LADDER.index(
                current
            )
        )

        if (
            current_index
            - previous_index
            != 1
        ):
            continue

        score_difference = (
            results[current]
            - results[previous]
        )

        previous_target = (
            target_mos - 0.01
            if previous >= 96
            else target_mos
        )

        previous_min_target = (
            4.88
            if previous <= 64
            else 4.87
        )

        previous_min_score = (
            min_scores.get(
                previous,
                0.0,
            )
        )

        if (
            current >= 96
            and score_difference < 0.01
            and results[previous]
            >= previous_target
            and previous_min_score
            >= previous_min_target
            and previous <= candidate
        ):

            print(
                f"[*] Plateau detected after "
                f"{previous} kbps "
                f"(gain {score_difference:.4f})"
            )

            candidate = previous
            break

    print(
        f"[+] Optimal bitrate: "
        f"{candidate} kbps"
    )

    return candidate


# ============================================================
# Batch processing
# ============================================================

def get_output_path(
    input_path: Path,
    suffix: str,
) -> Path:

    return (
        input_path.parent
        / f"{input_path.stem}{suffix}"
    )


def batch_process(
    input_directory: str | Path,
    *,
    pattern: str = "*.wav",
    target_mos: float = 4.72,
    preamp: float = 0.0,
    phase_inv: str = "off",
    output_suffix: str = "_EF",
):
    input_directory = Path(
        input_directory
    )

    if not input_directory.is_dir():
        raise RuntimeError(
            f"Directory does not exist: "
            f"{input_directory}"
        )

    files = sorted(
        p
        for p in input_directory.rglob(
            pattern
        )
        if p.is_file()
    )

    print(
        f"Found {len(files)} audio files."
    )

    for index, input_file in enumerate(
        files,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"[{index}/{len(files)}] "
            f"{input_file}"
        )
        print("=" * 70)

        try:
            bitrate = find_optimal_bitrate(
                input_file,
                target_mos,
            )

            output_file = get_output_path(
                input_file,
                output_suffix,
            ).with_suffix(
                ".opus"
            )

            print(
                f"[*] Encoding -> "
                f"{output_file}"
            )

            convert_opus(
                input_file,
                output_file,
                bitrate,
                preamp,
                phase_inv,
            )

            print(
                "[*] Comparing final encode..."
            )

            result = compare_audio(
                output_file,
                input_file,
                chunk_duration=20,
                skip_duration=0,
            )

            print_comparison(result)

        except Exception as exc:

            print(
                f"[ERROR] {input_file}: "
                f"{exc}",
                file=sys.stderr,
            )


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "EFUS Audio Tool - "
            "cross-platform audio conversion, "
            "comparison and bitrate optimization"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # convert
    # --------------------------------------------------------

    convert = subparsers.add_parser(
        "convert",
        help="Convert an audio file.",
    )

    convert.add_argument(
        "-c",
        "--codec",
        choices=["opus", "flac"],
        required=True,
    )

    convert.add_argument(
        "-i",
        "--input",
        required=True,
    )

    convert.add_argument(
        "-o",
        "--output",
        required=True,
    )

    convert.add_argument(
        "-b",
        "--bitrate",
        type=int,
        help="Opus bitrate in kbps.",
    )

    convert.add_argument(
        "-d",
        "--bitdepth",
        type=int,
        choices=[16, 24],
        default=16,
    )

    convert.add_argument(
        "--preamp",
        "-v",
        type=float,
        default=0.0,
        help="Preamp in dB.",
    )

    convert.add_argument(
        "--phase-inv",
        choices=["on", "scan", "off"],
        default="scan",
    )

    # --------------------------------------------------------
    # compare
    # --------------------------------------------------------

    compare = subparsers.add_parser(
        "compare",
        help="Compare lossy audio against reference.",
    )

    compare.add_argument(
        "lossy",
    )

    compare.add_argument(
        "reference",
    )

    compare.add_argument(
        "--chunk-duration",
        type=float,
        default=20.0,
    )

    compare.add_argument(
        "--skip-duration",
        type=float,
        default=20.0,
    )

    # --------------------------------------------------------
    # optimize
    # --------------------------------------------------------

    optimize = subparsers.add_parser(
        "optimize",
        help="Find optimal Opus bitrate.",
    )

    optimize.add_argument(
        "input",
    )

    optimize.add_argument(
        "--target",
        type=float,
        default=4.72,
    )

    optimize.add_argument(
        "--samples",
        type=int,
        default=8,
    )

    optimize.add_argument(
        "--sample-duration",
        type=float,
        default=20.0,
    )

    # --------------------------------------------------------
    # batch
    # --------------------------------------------------------

    batch = subparsers.add_parser(
        "batch",
        help="Optimize, encode and compare files.",
    )

    batch.add_argument(
        "directory",
    )

    batch.add_argument(
        "--pattern",
        default="*.wav",
    )

    batch.add_argument(
        "--target",
        type=float,
        default=4.72,
    )

    batch.add_argument(
        "--preamp",
        "-v",
        type=float,
        default=0.0,
    )

    batch.add_argument(
        "--phase-inv",
        choices=["on", "scan", "off"],
        default="off",
    )

    batch.add_argument(
        "--suffix",
        default="_EF",
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    try:

        if args.command == "convert":

            convert_audio(
                args.codec,
                args.input,
                args.output,
                bitrate=args.bitrate,
                bit_depth=args.bitdepth,
                preamp=args.preamp,
                phase_inv=args.phase_inv,
            )

        elif args.command == "compare":

            result = compare_audio(
                args.lossy,
                args.reference,
                chunk_duration=args.chunk_duration,
                skip_duration=args.skip_duration,
            )

            print_comparison(result)

        elif args.command == "optimize":

            bitrate = find_optimal_bitrate(
                args.input,
                args.target,
                args.samples,
                args.sample_duration,
            )

            print(
                f"bitrate:{bitrate}"
            )

        elif args.command == "batch":

            batch_process(
                args.directory,
                pattern=args.pattern,
                target_mos=args.target,
                preamp=args.preamp,
                phase_inv=args.phase_inv,
                output_suffix=args.suffix,
            )

    except KeyboardInterrupt:

        print(
            "\nInterrupted.",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
