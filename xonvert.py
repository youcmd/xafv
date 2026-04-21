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

def get_base_sample_rate(rate: int) -> int: # 88.2 > 44.1 or 96 > 48
    if rate < 44100:
        return rate
    if rate % 44100 == 0:
        return 44100
    elif rate % 48000 == 0:
        return 48000
    else:
        return 48000

def db_to_percent(db):
    return  round(10 ** (db / 20),4)

def run_command(command):
    """Executes the shell command."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    use_shell = isinstance(command, str)
    try:
        subprocess.run(command, shell=use_shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")

def process_audio(codec, bit_depth, input_path, output_path, bitrate=None, preamp=0, phase_inv_mode="scan", show_log=True):
    info = get_audio_info(input_path)
    sr = info['sample_rate']
    fmt = info['sample_fmt']
    bd = info['bit_depth']
    
    target_sr = get_base_sample_rate(sr)
    # print(f"{fmt}_{bd}:{sr} > {bit_depth}:{target_sr}")
    
    vol_filter = f'-af "volume={preamp}dB" ' if preamp and float(preamp) != 0.0 else ""
    preamp_percent = db_to_percent(preamp)
    vol = f"-v {preamp_percent}" if preamp and float(preamp) != 0.0 else ""
    
    logs = []

    if codec == 'flac': #WIP
        # Check if we need resampling or bit depth change
        resample_needed = sr != target_sr
        bit_depth_mismatch = (bit_depth == 16 and bd != 16) or (bit_depth == 24 and bd > 24)
        dither = "dither" if (bit_depth == 24 and bd > 24) else ("dither -s" if (bit_depth == 16 and bd > 16) else "")
        rate_arg = f"rate -v {target_sr} {dither}" if (sr != target_sr) else dither

        if bd > 32 or 'flt' in fmt:
            if float(preamp) == 0.0:
                cmd = (f'sox {vol} "{input_path}" -e signed-integer -b {bit_depth} -t wav -L - {rate_arg} | '
                       f'flac -8 -p -s -V -f -o "{output_path}" -')
            else:
                cmd = (f'ffmpeg -hide_banner -v quiet -i "{input_path}" {vol_filter}'
                   f'-f sox - | sox -p -e signed-integer -b {bit_depth} -t wav -L - {rate_arg} | flac -8 -p -s -V -f -o "{output_path}" -')
            run_command(cmd)
        elif resample_needed or bit_depth_mismatch or 'flt' in fmt or 's32' in fmt or float(preamp) != 0.0:
            cmd = (f'sox {vol} "{input_path}" -e signed-integer -b {bit_depth} -t wav -L - {rate_arg} | '
                   f'flac -8 -p -s -V -f -o "{output_path}" -')
            run_command(cmd)
        else:
            run_command(['flac', '-8', '-p', '-s', '-V', '-f', '-o', output_path, input_path])
        
        # Log results
        out_info = get_audio_info(output_path)
        ratio = (out_info['file_size'] / info['file_size']) * 100 if info['file_size'] else 0
        logs.append(f"flac: b:{bit_depth} s:{target_sr} d:{dither} ({ratio:.1f}% of source).")

    elif codec == 'opus':
        br_arg = f"--bitrate {bitrate}" if bitrate else ""
        rate_arg = "rate -v 48000" if (44100 < sr != 48000) else ""
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
        if fmt != "s32" and sr <= 48000 and bd <= 32 and float(preamp) == 0.0 :
            cmd = (f'opusenc --quiet {br_arg} {opus_npi} "{input_path}" "{output_path}"')
            run_command(cmd)
            # run_command(['opusenc',"--quiet", br_arg, opus_npi, input_path, output_path])
        elif target_sr != 48000 or sr > 48000 or 's32' in fmt or bd > 32 or float(preamp) != 0.0:
            if 'flt' in fmt:
                cmd = (f'ffmpeg -hide_banner -v quiet -i "{input_path}" {vol_filter}'
                   f'-f sox - | sox -p -D -e floating-point -b 32 -L -t wav - {rate_arg} | opusenc --quiet {br_arg} {opus_npi} - "{output_path}"')
            elif 's32' in fmt:
                cmd = (f'sox {vol} "{input_path}" -D -e floating-point -b 32 -L -t wav - {rate_arg} | '
                   f'opusenc --quiet {br_arg} {opus_npi} - "{output_path}"')
            else:
                cmd = (f'sox {vol} "{input_path}" -D -L -t wav - {rate_arg} | '
                   f'opusenc --quiet {br_arg} {opus_npi} - "{output_path}"')
            run_command(cmd)
        else:
            cmd = (f'opusenc --quiet {br_arg} {opus_npi} "{input_path}" "{output_path}"')
            run_command(cmd)
        
        out_info = get_audio_info(output_path)
        logs.append(f"opus: {out_info['kbps']}kbps npi:{npi_status}.")
    
    logs.append(f"b:{bd} s:{sr} preamp:{preamp}.")

    if show_log:
        print(" ".join(logs))

def main():
    parser = argparse.ArgumentParser(description="Custom Audio Converter Wrapper")
    parser.add_argument('-c', '--codec', choices=['flac', 'opus'], required=True, help="Output codec")
    parser.add_argument('-d', '--bitdepth', type=int, choices=[16, 24], default=16, help="Bit depth (FLAC only)")
    parser.add_argument('-b', '--bitrate', type=int, help="Bitrate in kbps (Opus only)")
    parser.add_argument('-i', '--input', required=True, help="Input file path")
    parser.add_argument('-o', '--output', required=True, help="Output file path")
    parser.add_argument('-vol', '--preamp', type=float, default=0.0, help="Volume adjustment in dB (e.g., -3 or 1.5)")
    parser.add_argument('-pi', '--phase-inv', choices=['on', 'scan', 'off'], default='scan', help="Control Opus phase inversion (default: scan)")
    
    args = parser.parse_args()
    process_audio(args.codec, args.bitdepth, args.input, args.output, args.bitrate, args.preamp, args.phase_inv)

if __name__ == "__main__":
    main()
