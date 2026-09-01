#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from zimtohrli import mos_from_signals

def load_audio_stereo(filepath):
    temp_wav = f"/dev/shm/temp_decode_{os.getpid()}_{np.random.randint(10000)}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", filepath, "-vn", "-sn", "-dn",
        "-af", "aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt",
        "-ac", "2", "-f", "wav", "-c:a", "pcm_f32le", "-map_metadata", "-1", temp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    y, sr = sf.read(temp_wav, always_2d=True)
    y = y.T.astype(np.float32)

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    return y

def get_bitrate(filepath):
  cmd = [
      "ffprobe",
      "-v",
      "quiet",
      "-print_format",
      "json",
      "-show_format",
      filepath,
  ]
  result = subprocess.run(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
  )
  data = json.loads(result.stdout)
  return (
      float(data["format"]["bit_rate"]) / 1000.0
      if "bit_rate" in data["format"]
      else 0.0
  )


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Compare a lossy audio file against a reference using Zimtohrli."
      )
  )
  parser.add_argument("lossy", help="Path to the lossy/test audio file")
  parser.add_argument("ref", help="Path to the reference audio file")
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

  # Evaluate Left and Right channels concurrently
  with ThreadPoolExecutor(max_workers=2) as executor:
    future_l = executor.submit(mos_from_signals, ref_l, test_l)
    future_r = executor.submit(mos_from_signals, ref_r, test_r)
    scoreL = future_l.result()
    scoreR = future_r.result()

  score = (scoreL + scoreR) / 2.0
  bitrate = get_bitrate(args.lossy)

#   print("\n")
#   print("\n--- Zimtohrli Audio Quality Comparison ---")
#   print(f"File:     {os.path.basename(args.lossy)}", end="\t")
  print(f"Score:    {score:.6f}", end="\t")
  print(f"Left:     {scoreL:.6f}", end="\t")
  print(f"Right:    {scoreR:.6f}", end="\t")
  print(f"Bitrate:  {bitrate:.3f} kbps")


if __name__ == "__main__":
  main()
