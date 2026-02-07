import subprocess
import json
import sys
import os
import shutil
from pathlib import Path

def check_dependencies():
    """Ensure required binaries are in the PATH."""
    deps = ['ffmpeg', 'ffprobe', 'opustags']
    missing = [d for d in deps if shutil.which(d) is None]
    if missing:
        # opustags is only strictly needed for opus; we check here for simplicity
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        sys.exit(1)

def extract_audio(input_file):
    src_path = Path(input_file).resolve()
    
    if not src_path.exists():
        print(f"Error: File {src_path} not found.")
        return

    try:
        # 1. Get stream info
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(src_path)]
        result = subprocess.check_output(cmd)
        info = json.loads(result)
        
        audio_count = 0
        for stream in info.get('streams', []):
            if stream['codec_type'] == 'audio':
                codec = stream['codec_name']
                
                # Extension mapping
                ext_map = {
                    'opus': 'opus', 'vorbis': 'ogg', 'aac': 'm4a', 
                    'mp3': 'mp3', 'flac': 'flac', 'alac': 'm4a', 
                    'pcm_s16le': 'wav', 'pcm_s24le': 'wav', 'ac3': 'ac3', 'dts': 'dts'
                }
                ext = ext_map.get(codec, 'mka')
                
                suffix = f"_track{audio_count}" if audio_count > 0 else ""
                output_name = f"{src_path.stem}{suffix}.{ext}"
                output_path = src_path.parent / output_name
                cover_path = src_path.parent / f"{src_path.stem}_cover.png"

                print(f"--- Processing Track {audio_count}: {codec} ---")

                # 2. Extract Audio (Normal behavior for most formats)
                if ext not in ['m4a', 'opus']:
                    subprocess.run([
                        'ffmpeg', '-hide_banner', '-y', '-i', str(src_path),
                        '-map', f'0:a:{audio_count}', '-vn', '-c:a', 'copy', str(output_path)
                    ], check=True)
                
                else:
                    # 3. Handle Cover Extraction (Only for M4A/Opus)
                    # Try extracting at 10:00, then 1:00 if that's too far
                    for timestamp in ["00:10:00", "00:01:00"]:
                        subprocess.run([
                            'ffmpeg', '-hide_banner', '-y', '-ss', timestamp, '-i', str(src_path),
                            '-vf', r"select=eq(pict_type\,I)", '-vframes', '1', str(cover_path)
                        ], capture_output=True)
                        if cover_path.exists() and cover_path.stat().st_size > 0:
                            break

                    # 4. M4A Specific: Extract and attach cover in one go
                    if ext == 'm4a':
                        print("Applying cover to M4A...")
                        subprocess.run([
                            'ffmpeg', '-hide_banner', '-y', '-i', str(src_path), '-i', str(cover_path),
                            '-map', f'0:a:{audio_count}', '-map', '1', '-c', 'copy',
                            '-disposition:v:0', 'attached_pic', '-movflags', '+faststart', str(output_path)
                        ], check=True)

                    # 5. Opus Specific: Extract audio then use opustags
                    elif ext == 'opus':
                        print("Applying cover to Opus via opustags...")
                        # Extract temp audio first
                        temp_opus = src_path.parent / f"temp_{output_name}"
                        subprocess.run([
                            'ffmpeg', '-hide_banner', '-y', '-i', str(src_path),
                            '-map', f'0:a:{audio_count}', '-vn', '-c:a', 'copy', str(temp_opus)
                        ], check=True)
                        
                        # Apply tags
                        subprocess.run([
                            'opustags', '--set-cover', str(cover_path), str(temp_opus), '-o', str(output_path)
                        ], check=True)
                        
                        if temp_opus.exists():
                            os.remove(temp_opus)

                # Clean up cover image after track is done
                if cover_path.exists():
                    os.remove(cover_path)
                
                audio_count += 1
                
        if audio_count == 0:
            print("No audio streams found.")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg/Opustags error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_dependencies()
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <video_file>")
    else:

        extract_audio(sys.argv[1])
