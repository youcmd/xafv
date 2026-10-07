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

def get_strict_opus_rate(rate: int) -> int:
    if rate <= 8000:
        return 8000
    elif rate <= 12000:
        return 12000
    elif rate <= 16000:
        return 16000
    elif rate <= 24000:
        return 24000
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
        if isinstance(command, str) and command.startswith("sox "):
            fallback = "/usr/bin/sox"

            if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
                print(f"Retrying with {fallback}...")
                command = fallback + command[3:]

                try:
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Fallback SoX also failed: {e}")
                    raise
            else:
                print(f"Fallback SoX not found: {fallback}")
                raise

def process_audio(codec, bit_depth, input_path, output_path, bitrate=None, preamp=0, show_log=True):
    info = get_audio_info(input_path)
    sr = info['sample_rate']
    fmt = info['sample_fmt']
    bd = info['bit_depth']
    
    target_sr = get_base_sample_rate(sr)
    
    gain = f'gain {preamp}' if preamp and float(preamp) != 0.0 else ""
    
    logs = []

    if codec == 'flac':
        resample_needed = sr != target_sr
        bit_depth_mismatch = (bit_depth == 16 and bd != 16) or (bit_depth == 24 and bd > 24)
        # dither = "dither" if (bit_depth == 24 and bd > 24) else ("dither -s" if (bit_depth == 16 and bd > 16) else "")
        dither = (
            "dither -s" if bit_depth == 16 and bd > 16 and sr >= 32000
            else "dither" if (bit_depth == 16 and bd > 16) or (bit_depth == 24 and bd > 24)
            else ""
        )
        
        # Combine gain and rate/dither natively using sox-vulkan
        rate_args = []
        if gain:
            rate_args.append(gain)
        if resample_needed:
            rate_args.append(f"rate -v {target_sr}")
        if dither:
            rate_args.append(dither)
        
        rate_arg = " ".join(rate_args)
        no_dither = "-D" if dither == "" else ""

        if bd == 16 and sr <= 48000 and float(preamp) == 0.0 and not resample_needed:
            run_command(['flac', '-8', '-p', '-s', '-V', '-f', '-o', output_path, input_path])
        else:
            cmd = (f'sox "{input_path}" {no_dither} -G -e signed-integer -b {bit_depth} -t wav -L - {rate_arg} | '
                   f'flac -8 -p -s -V -f -o "{output_path}" -')
            run_command(cmd)
        
        # Log results
        out_info = get_audio_info(output_path)
        ratio = (out_info['file_size'] / info['file_size']) * 100 if info['file_size'] else 0
        logs.append(f"flac: b:{bit_depth} s:{target_sr} d:{dither} ({ratio:.1f}% of source).")

    elif codec == 'opus':
        br_arg = f"--bitrate {bitrate}" if bitrate else ""
        # rate_arg = f"rate -v {output_sr}" if (44100 < sr != 48000) else ""
        rate_arg = f"rate -v {get_strict_opus_rate(sr)}" if sr not in {8000, 12000, 16000, 24000, 48000} else ""

        sox_effects = []
        if gain:
            sox_effects.append(gain)
        if rate_arg:
            sox_effects.append(rate_arg)
        effects_str = " ".join(sox_effects)

        if fmt != "s32" and sr <= 48000 and bd <= 32 and float(preamp) == 0.0 :
            cmd = (f'opusenc --quiet {br_arg} "{input_path}" "{output_path}"')
        else: #use sox if wav is s32 or preamp !=0
            cmd = (f'sox "{input_path}" -D -G -e floating-point -b 32 -L -t wav - {effects_str} | '
                   f'opusenc --quiet {br_arg} - "{output_path}"')
        
        run_command(cmd)
        
        out_info = get_audio_info(output_path)
        logs.append(f"opus: {out_info['kbps']}kbps")
    
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
    
    args = parser.parse_args()
    process_audio(args.codec, args.bitdepth, args.input, args.output, args.bitrate, args.preamp)

if __name__ == "__main__":
    main()
