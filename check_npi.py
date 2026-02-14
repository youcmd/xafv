import subprocess
import tempfile
import shutil
import re
import math
from pathlib import Path
from typing import Dict, Optional

def analyze_phase(file_path: str,
                  ffmpeg_cmd: str = "ffmpeg",
                  side_threshold: float = 0.20) -> Dict[str, Optional[object]]:
    """
    Analyze a stereo file for mid/side energy and recommend whether to use --no-phase-inv.

    Returns a dict with keys:
      - file: input path
      - mid_db: mean_volume (dB) of mid channel (float)
      - side_db: mean_volume (dB) of side channel (float)
      - side_pct: fraction of energy in side (0.0-1.0)
      - side_pct_percent: formatted percent string (e.g., "12.3%")
      - correlation: numeric correlation if found (float) or None
      - recommendation: "use --no-phase-inv" or "allow phase inversion"
      - error: error message if something failed (None on success)

    Requires ffmpeg available on PATH.
    """
    tmpdir = tempfile.mkdtemp(prefix="ms_analyze_")
    try:
        src_path = Path(file_path).resolve()
        temp_input = Path(tmpdir) / "input_src.wav"
        mid_path = Path(tmpdir) / "mid.wav"
        side_path = Path(tmpdir) / "side.wav"

        shutil.copy2(src_path, temp_input)

        # 1) create mid and side mono files
        cmd_pan = [
            ffmpeg_cmd, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(temp_input),
            "-filter_complex",
            "[0:a]pan=mono|c0=0.5*c0+0.5*c1[mid];[0:a]pan=mono|c0=0.5*c0-0.5*c1[side]",
            "-map", "[mid]", str(mid_path),
            "-map", "[side]", str(side_path)
        ]
        subprocess.run(cmd_pan, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

        # helper to run volumedetect and extract mean_volume
        def mean_volume(path: Path) -> Optional[float]:
            cmd = [ffmpeg_cmd, "-hide_banner", "-nostats", "-y", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
            proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = proc.stderr + proc.stdout
            # common volumedetect line: "mean_volume: -21.0 dB"
            m = re.search(r"mean_volume\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*dB", out, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
            # fallback: try to find "mean_volume" with any numeric token
            m2 = re.search(r"mean_volume.*?([-+]?\d+(?:\.\d+)?)", out, re.IGNORECASE)
            if m2:
                try:
                    return float(m2.group(1))
                except ValueError:
                    return None
            return None

        mid_db = mean_volume(mid_path)
        side_db = mean_volume(side_path)

        if mid_db is None or side_db is None:
            return {
                "file": file_path,
                "mid_db": mid_db,
                "side_db": side_db,
                "side_pct": None,
                "side_pct_percent": None,
                "correlation": None,
                "recommendation": None,
                "error": "Failed to extract mean_volume from ffmpeg volumedetect output."
            }

        # 2) compute side percent (linear energy share)
        mid_lin = 10 ** (mid_db / 10.0)
        side_lin = 10 ** (side_db / 10.0)
        side_pct = side_lin / (mid_lin + side_lin) if (mid_lin + side_lin) > 0 else 0.0
        side_pct_percent = f"{100 * side_pct:.1f}%"

        # 3) try to extract correlation via astats (optional)
        corr = None
        try:
            cmd_astats = [
                ffmpeg_cmd, "-hide_banner", "-nostats", "-y", "-i", str(temp_input),
                "-af", "astats=measure_perchannel=1:reset=0", "-f", "null", "-"
            ]
            proc = subprocess.run(cmd_astats, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            out = proc.stderr + proc.stdout
            # look for "Overall.Correlation: <num>" or "Correlation: <num>"
            m_corr = re.search(r"(?:Overall\.)?Correlation\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)", out, re.IGNORECASE)
            if m_corr:
                corr = float(m_corr.group(1))
        except Exception:
            corr = None

        # 4) recommendation logic
        recommend_no_phase = False
        if side_pct >= side_threshold:
            recommend_no_phase = True
        if corr is not None and corr <= 0:
            recommend_no_phase = True

        recommendation = "use --no-phase-inv (true stereo or anti-phase detected)" if recommend_no_phase else "allow phase inversion (likely double-mono or low side energy)"

        return {
            "file": file_path,
            "mid_db": mid_db,
            "side_db": side_db,
            "side_pct": side_pct,
            "side_pct_percent": side_pct_percent,
            "correlation": corr,
            "recommendation": recommendation,
            "error": None
        }

    except subprocess.CalledProcessError as e:
        return {
            "file": file_path,
            "mid_db": None,
            "side_db": None,
            "side_pct": None,
            "side_pct_percent": None,
            "correlation": None,
            "recommendation": None,
            "error": f"ffmpeg failed: {e}"
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def isnophaseinv(file_path: str, Verbose: bool = False):
    """
    Simple decision: return True if --no-phase-inv is recommended,
    False if phase inversion is allowed, or None on error.
    """
    analysed = analyze_phase(file_path)
    if analysed.get("error"):
        if Verbose:
            print("error:", analysed["error"])
        return None

    side_pct = analysed.get("side_pct", 0.0) or 0.0
    corr = analysed.get("correlation", None)

    if Verbose:
        pct_str = analysed.get("side_pct_percent") or f"{100*side_pct:.1f}%"
        print("Side energy:", pct_str)
        print("Correlation:", corr if corr is not None else "(not found)")

    # decision: recommend no-phase-inv when side energy is high or correlation <= 0
    SIDE_THRESHOLD = 0.20
    if side_pct >= SIDE_THRESHOLD:
        return True
    if corr is not None:
        try:
            if float(corr) <= 0.0:
                return True
        except Exception:
            pass
    return False
