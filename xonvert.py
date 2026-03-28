import subprocess
import argparse
import sys
import json

#replace with mediainfo or sox
def get_audio_info(input_file): 
    """Uses ffprobe to get sample rate, sample format, and bit depth."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'a:0', input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error reading file: {input_file}")
        sys.exit(1)
    
    data = json.loads(result.stdout)['streams'][0]
    return {
        'sample_rate': int(data.get('sample_rate', 0)),
        'sample_fmt': data.get('sample_fmt', ''),
        'bits_per_raw_sample': int(data.get('bits_per_raw_sample', 0)) if data.get('bits_per_raw_sample') else 0
    }

def run_command(command):
    """Executes the shell command."""
    print(f"Executing: {' '.join(command) if isinstance(command, list) else command}")
    try:
        if "|" in str(command):
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")

def get_base_sample_rate(rate: int) -> int: # 88.2 > 44.1 or 96 > 48
    if rate < 44100:
        return rate
    if rate % 44100 == 0:
        return 44100
    elif rate % 48000 == 0:
        return 48000
    else:
        return 48000

def process_audio(codec, bit_depth, input_path, output_path):
    info = get_audio_info(input_path)
    sr = info['sample_rate']
    fmt = info['sample_fmt']
    
    # Logic for target sample rate
    target_sr = get_base_sample_rate(sr)

    needs_ffmpeg = False
    
    if codec == 'flac': #WIP
        # Check if we need resampling or bit depth change
        resample_needed = sr != target_sr
        bit_depth_mismatch = (bit_depth == 16 and fmt != 's16') or (bit_depth == 24 and fmt != 's32' and fmt != 's24')
        
        if resample_needed or bit_depth_mismatch or 'flt' in fmt or 's32' in fmt:
            dither = "shibata" if bit_depth == 16 else "triangular"
            osf = "s16" if bit_depth == 16 else "s32"
            pcm = "pcm_s16le" if bit_depth == 16 else "pcm_s24le"
            
            cmd = (f'/content/ffmpeg -hide_banner -v verbose -i "{input_path}" '
                   f'-af "volume=0dB,aresample={target_sr}:resampler=soxr:cutoff=1:precision=33:dither_method={dither}:osf={osf}" '
                   f'-c:a {pcm} -f wav - | flac -f -o "{output_path}" -')
            run_command(cmd)
        else:
            run_command(['flac', '-f', '-o', output_path, input_path])

    elif codec == 'opus':
        # Opus strictly handles float; if it's already float or simple enough, opusenc handles it
        if target_sr != 48000 or 's32' in fmt:
            cmd = (f'/content/ffmpeg -hide_banner -v verbose -i "{input_path}" '
                   f'-af "volume=0dB,aresample=48000:resampler=soxr:cutoff=1:precision=33:dither_method=none:osf=flt" '
                   f'-c:a pcm_f32le -f wav - | opusenc - "{output_path}"')
            run_command(cmd)
        elif 'flt' in fmt and target_sr == 48000:
            run_command(['opusenc', input_path, output_path])
        else:
            run_command(['opusenc', input_path, output_path])

def main():
    parser = argparse.ArgumentParser(description="Custom Audio Converter Wrapper")
    parser.add_argument('-a:c', '--codec', choices=['flac', 'opus'], required=True, help="Output codec")
    parser.add_argument('-a:b', '--bitdepth', type=int, choices=[16, 24], default=16, help="Bit depth (FLAC only)")
    parser.add_argument('-i', '--input', required=True, help="Input file path")
    parser.add_argument('-o', '--output', required=True, help="Output file path")

    args = parser.parse_args()
    process_audio(args.codec, args.bitdepth, args.input, args.output)

if __name__ == "__main__":
    main()
