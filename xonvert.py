import subprocess
import argparse
import sys
import json

from pymediainfo import MediaInfo

import check_npi

def get_audio_info(input_file):
    media_info = MediaInfo.parse(input_file)
    
    # Separate tracks
    audio_track = next((t for t in media_info.tracks if t.track_type == "Audio"), None)
    general_track = next((t for t in media_info.tracks if t.track_type == "General"), None)
    
    if not audio_track or not general_track:
        return None

    # Extraction with fallbacks
    bit_depth = int(audio_track.bit_depth) if audio_track.bit_depth else 0
    is_float = audio_track.format_settings_endianness == "Float" or audio_track.format_profile == "Float"
    duration_ms = float(general_track.duration) if general_track.duration else 0
    file_size = int(general_track.file_size) if general_track.file_size else 0
    
    # Calculate Bitrate (bps)
    stream_size = int(audio_track.stream_size) if audio_track.stream_size else file_size
    if stream_size and duration_ms > 0:
        # duration is in ms, so (size * 8) / (ms / 1000)
        calc_bitrate = (stream_size * 8) / (duration_ms / 1000)
    else:
        calc_bitrate = int(audio_track.bit_rate) if audio_track.bit_rate else 0

    # Determine sample_fmt
    if is_float:
        sample_fmt = 'flt'
    elif bit_depth in [24, 32]:
        sample_fmt = f's{bit_depth}'
    else:
        sample_fmt = f"s{bit_depth}" if bit_depth else "unknown"

    return {
        'sample_rate': int(audio_track.sampling_rate) if audio_track.sampling_rate else 0,
        'sample_fmt': sample_fmt,
        'bit_depth': bit_depth,
        'file_size': file_size,
        'kbps': f'{round(calc_bitrate / 1000)}kbps' if calc_bitrate > 0 else 'N/A'
    }

def run_command(command):
    """Executes the shell command."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    use_shell = isinstance(command, str)
    try:
        subprocess.run(command, shell=use_shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")

def get_base_sample_rate(rate: int) -> int: # 88.2 > 44.1 or 96 > 48
    if rate < 44100:
        return rate
    if rate % 44100 == 0:
        return 44100
    elif rate % 48000 == 0:
        return 48000
    else:
        return 48000

def process_audio(codec, bit_depth, input_path, output_path, bitrate=None, preamp=0, phase_inv_mode="scan", show_log=True):
    info = get_audio_info(input_path)
    sr = info['sample_rate']
    fmt = info['sample_fmt']
    bd = info['bit_depth']
    
    # Logic for target sample rate
    target_sr = get_base_sample_rate(sr)
    # print(f"{fmt}_{bd}:{sr} > {bit_depth}:{target_sr}")

    needs_ffmpeg = False
    dither = "none"
    vol_filter = f"volume={preamp}dB," if preamp and float(preamp) != 0.0 else ""
    logs = []

    if codec == 'flac': #WIP
        # Check if we need resampling or bit depth change
        resample_needed = sr != target_sr
        bit_depth_mismatch = (bit_depth == 16 and bd != 16) or (bit_depth == 24 and bd > 24)
        
        # if resample_needed or bit_depth_mismatch or (bd >= 24 and bd != bit_depth):
        if resample_needed or bit_depth_mismatch or 'flt' in fmt or 's32' in fmt:
            if bit_depth != bd: dither = "shibata" if bit_depth == 16 else "triangular"
            osf = "s16" if bit_depth == 16 else "s32"
            pcm = "pcm_s16le" if bit_depth == 16 else "pcm_s24le"
            
            cmd = (f'/content/ffmpeg -hide_banner -v quiet -i "{input_path}" '
                   f'-af "{vol_filter}aresample={target_sr}:resampler=soxr:cutoff=1:precision=33:dither_method={dither}:osf={osf}" '
                   f'-c:a {pcm} -f wav - | flac -s -V -f -o "{output_path}" -')
            run_command(cmd)
        else:
            run_command(['flac', '-s', '-V', '-f', '-o', output_path, input_path])
        
        # Log results
        out_info = get_audio_info(output_path)
        ratio = (out_info['file_size'] / info['file_size']) * 100 if info['file_size'] else 0
        logs.append(f"flac: b:{bit_depth} s:{target_sr} d:{dither} ({ratio:.1f}% of source).")

    elif codec == 'opus':
        br_arg = f"--bitrate {bitrate}" if bitrate else ""
        # --- NPI Logic ---
        opus_npi = ""
        npi_status = "on"

        if phase_inv_mode == "on":
            opus_npi = ""
            npi_status = "forced-off"
        elif phase_inv_mode == "scan":
            isnophaseinv = check_npi.isnophaseinv(input_path)
            opus_npi = "--no-phase-inv" if isnophaseinv else ""
            npi_status = "scanned:" + ("on" if isnophaseinv else "off")
        else: # false
            opus_npi = "--no-phase-inv"
            npi_status = "forced-on"

        # Opus strictly handles float; if it's already float or simple enough, opusenc handles it
        if fmt != "s32" and sr == 48000:
            cmd = (f'opusenc --quiet {br_arg} {opus_npi} "{input_path}" "{output_path}"')
            run_command(cmd)
            # run_command(['opusenc',"--quiet", br_arg, opus_npi, input_path, output_path])
        elif target_sr != 48000 or sr > 48000 or 's32' in fmt:
            cmd = (f'/content/ffmpeg -hide_banner -v quiet -i "{input_path}" '
                   f'-af "{vol_filter}aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt" '
                   f'-c:a pcm_f32le -f wav - | opusenc --quiet {br_arg} {opus_npi} - "{output_path}"')
            run_command(cmd)
        else:
            cmd = (f'opusenc --quiet {br_arg} {opus_npi} "{input_path}" "{output_path}"')
            run_command(cmd)
            # run_command(['opusenc',"--quiet", f"{br_arg}", f"{opus_npi}", input_path, output_path])
        
        out_info = get_audio_info(output_path)
        logs.append(f"opus: {out_info['kbps']}kbps npi:{npi_status}.")
    
    logs.append(f"b:{bd} s:{sr} preamp:{preamp}.")

    if show_log:
        print(" ".join(logs))

def main():
    parser = argparse.ArgumentParser(description="Custom Audio Converter Wrapper")
    parser.add_argument('-a:c', '--codec', choices=['flac', 'opus'], required=True, help="Output codec")
    parser.add_argument('-a:b', '-a:bd', '--bitdepth', type=int, choices=[16, 24], default=16, help="Bit depth (FLAC only)")
    parser.add_argument('-b', '-a:br', '--bitrate', type=int, help="Bitrate in kbps (Opus only)")
    parser.add_argument('-i', '--input', required=True, help="Input file path")
    parser.add_argument('-o', '--output', required=True, help="Output file path")
    parser.add_argument('-vol', '--preamp', type=float, default=0.0, help="Volume adjustment in dB (e.g., -3 or 1.5)")
    parser.add_argument('-pi', '--phase-inv', choices=['on', 'scan', 'off'], default='scan', help="Control Opus phase inversion (default: scan)")
                            
    args = parser.parse_args()
    # process_audio(args.codec, args.bitdepth, args.input, args.output, args.preamp)
    process_audio(args.codec, args.bitdepth, args.input, args.output, args.bitrate, args.preamp, args.phase_inv)

if __name__ == "__main__":
    main()
