#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from playwright.sync_api import sync_playwright
import sys
import os
import time

def get_download_link(page):
    download_link = None
    try:
        inputs = page.locator('input[type="text"], input[readonly]').all()
        for inp in inputs:
            value = inp.input_value()
            if value and 'transfer.it/t/' in value:
                download_link = value
                break
    except:
        pass
    
    if not download_link:
        try:
            clipboard_button = page.locator('button.js-copy-to-clipboard')
            if clipboard_button.is_visible():
                clipboard_button.click()
                time.sleep(0.5)
                download_link = page.evaluate('navigator.clipboard.readText()')
        except:
            pass
    
    if not download_link:
        current_url = page.url
        if 'transfer.it/t/' in current_url:
            download_link = current_url
    
    return download_link.strip() if download_link else None


def wait_for_upload(page, start_time, timeout=7200):
    last_progress = 0
    last_displayed_progress = 0
    last_progress_time = time.time()
    
    def handle_console(msg):
        nonlocal last_progress, last_progress_time, last_displayed_progress
        text = msg.text
        
        if 'ul-progress' in text:
            parts = text.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0 and parts[i-1] == 'ul_2048':
                    progress = int(part)
                    if progress > last_progress:
                        last_progress = progress
                        last_progress_time = time.time()
                        
                        if progress % 10 == 0 and progress != last_displayed_progress:
                            last_displayed_progress = progress
                            elapsed = int(time.time() - start_time)
                            print(f"   [*] Progress: {progress}% | Elapsed: {elapsed // 60}m {elapsed % 60}s")
                    break
    
    page.on('console', handle_console)
    
    while (time.time() - start_time) < timeout:
        try:
            if last_progress >= 100:
                print("   [+] Progress 100% - Upload completed!")
                return True
            
            if page.locator('text=Completed!').is_visible():
                print("   [+] 'Completed!' signal found!")
                return True
            
            if last_progress > 0 and (time.time() - last_progress_time) > 300:
                print(f"   [!] Progress stuck at {last_progress}% for 5 minutes!")
            
            time.sleep(5)
        except:
            time.sleep(5)
    
    print("[!] Timeout! Upload took too long.")
    return False


def upload_single_file(page, file_path):
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    file_name = os.path.basename(file_path)
    
    print(f"\n{'-'*60}")
    print(f"[*] File: {file_name}")
    print(f"[*] Size: {file_size_mb:.2f} MB")
    print(f"{'-'*60}")
    
    try:
        print("[*] Connecting to Transfer.it...")
        page.goto('https://transfer.it', wait_until='networkidle', timeout=30000)
        
        print("[*] Selecting file...")
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(file_path)
        time.sleep(1)
        
        print("[*] Starting transfer...")
        transfer_button = page.locator('button.js-get-link-button')
        transfer_button.wait_for(state='visible', timeout=10000)
        transfer_button.click()
        
        print("[*] Uploading...")
        start_time = time.time()
        
        if not wait_for_upload(page, start_time):
            return None
        
        print("[+] Upload complete!")
        print("[*] Waiting for link button...")
        
        copy_button = page.locator('button.js-copy-link')
        copy_button.wait_for(state='visible', timeout=0)
        print("[*] Retrieving link...")
        copy_button.click()
        page.wait_for_load_state('networkidle', timeout=0)
        
        download_link = get_download_link(page)
        
        if download_link:
            print(f"[+] Link retrieved!")
            print(f"[*] {download_link}")
            return download_link
        else:
            print("[!] Failed to retrieve link!")
            return None
            
    except Exception as e:
        print(f"[-] Error: {e}")
        return None


def upload_multiple_files(page, file_paths):
    total_size = sum(os.path.getsize(f) for f in file_paths)
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    
    print(f"\n{'-'*60}")
    print(f"[*] Uploading {len(file_paths)} files together")
    print(f"[*] Total size: {total_size_gb:.2f} GB ({total_size_mb:.2f} MB)")
    print(f"{'-'*60}")
    
    for i, fp in enumerate(file_paths, 1):
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {i}. {os.path.basename(fp)} ({size_mb:.2f} MB)")
    
    print(f"{'-'*60}")
    
    try:
        print("\n[*] Connecting to Transfer.it...")
        page.goto('https://transfer.it', wait_until='networkidle', timeout=30000)
        
        print("[*] Selecting files...")
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(file_paths)
        time.sleep(2)
        
        print("[*] Starting transfer...")
        transfer_button = page.locator('button.js-get-link-button')
        transfer_button.wait_for(state='visible', timeout=10000)
        transfer_button.click()
        
        print("[*] Uploading...")
        print("   (This may take a while for multiple files)\n")
        start_time = time.time()
        
        if not wait_for_upload(page, start_time):
            return None
        
        print("[+] Upload complete!")
        print("[*] Waiting for link button...")
        
        copy_button = page.locator('button.js-copy-link')
        copy_button.wait_for(state='visible', timeout=0)
        print("[*] Retrieving link...")
        copy_button.click()
        page.wait_for_load_state('networkidle', timeout=0)
        
        download_link = get_download_link(page)
        
        if download_link:
            print(f"[+] Link retrieved!")
            print(f"[*] {download_link}")
            return download_link
        else:
            print("[!] Failed to retrieve link!")
            return None
            
    except Exception as e:
        print(f"[-] Error: {e}")
        return None


def upload_files(file_paths, together=True):
    print(f"\n{'='*60}")
    print(f"[*] Transfer.it Upload Tool")
    print(f"{'='*60}")
    print(f"[*] Total files: {len(file_paths)}")
    
    if together and len(file_paths) > 1:
        print(f"[*] Mode: Single link for all files")
    else:
        print(f"[*] Mode: Separate links per file")
    
    print(f"{'='*60}")
    
    results = []
    
    with sync_playwright() as p:
        print("\n[*] Launching browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(permissions=['clipboard-read', 'clipboard-write'])
        page = context.new_page()
        
        if together and len(file_paths) > 1:
            valid_files = [fp for fp in file_paths if os.path.exists(fp)]
            for fp in file_paths:
                if not os.path.exists(fp): print(f"[-] File not found: {fp}")
            
            if valid_files:
                link = upload_multiple_files(page, valid_files)
                results.append((valid_files, link))
        else:
            for i, file_path in enumerate(file_paths, 1):
                print(f"\n{'='*60}")
                print(f"[*] File {i}/{len(file_paths)}")
                print(f"{'='*60}")
                
                if not os.path.exists(file_path):
                    print(f"[-] File not found: {file_path}")
                    results.append((file_path, None))
                    continue
                
                link = upload_single_file(page, file_path)
                results.append((file_path, link))
                
                if i < len(file_paths):
                    time.sleep(2)
        
        browser.close()
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"[*] SUMMARY")
    print(f"{'='*60}")
    
    for item, link in results:
        status = "[+]" if link else "[-]"
        name = f"{len(item)} files" if isinstance(item, list) else os.path.basename(item)
        print(f"  {status} {name} -> {link if link else 'FAILED'}")
    
    print(f"{'='*60}\n")
    return results


def main():
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("[*] Transfer.it CLI - Usage")
        print("="*60)
        print("\nSingle file:")
        print("  py transferit.py <file_path>")
        print("\nMultiple files (single link):")
        print("  py transferit.py <file1> <file2>")
        print("\nOptions:")
        print("  --separate         Upload files individually")
        print("  --output <file>    Save links to a text file")
        print("\nExamples:")
        print("  py transferit.py movie.mp4 --output results.txt")
        print("  py transferit.py --separate file1.zip file2.zip")
        print("="*60 + "\n")
        sys.exit(1)
    
    args = sys.argv[1:]
    together = True
    output_file = None
    
    if '--separate' in args:
        together = False
        args.remove('--separate')
        
    if '--output' in args or '-o' in args:
        flag = '--output' if '--output' in args else '-o'
        idx = args.index(flag)
        if idx + 1 < len(args):
            output_file = args[idx + 1]
            args.pop(idx + 1)
            args.pop(idx)
        else:
            print(f"[-] Error: No filename provided for {flag}")
            sys.exit(1)
            
    file_paths = args
    if not file_paths:
        print("[-] Error: No files specified!")
        sys.exit(1)
    
    results = upload_files(file_paths, together=together)
    
    if output_file:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Upload Session: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            for item, link in results:
                if link:
                    name = f"{len(item)} files" if isinstance(item, list) else os.path.basename(item)
                    f.write(f"{name}: {link}\n")
        print(f"[+] Results saved to {output_file}")


if __name__ == '__main__':
    main()