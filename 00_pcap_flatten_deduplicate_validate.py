import os
import shutil
import hashlib
from pathlib import Path

SOURCE = "../pcaps2"
REFERENCE = "../pcaps"

def calculate_sha256(file_path, block_size=65536):
    """
    Calculates the SHA-256 hash of a file using a buffered approach
    to handle large PCAP files efficiently.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                sha256.update(block)
        return sha256.hexdigest()
    except (OSError, IOError) as e:
        print(f"[!] Error hashing {file_path}: {e}")
        return None

def process_pcaps(source_dir_name, reference_dir_name):
    pcaps2_root = Path(source_dir_name).resolve()
    pcaps_ref_root = Path(reference_dir_name).resolve()
    
    if not pcaps2_root.exists() or not pcaps_ref_root.exists():
        print(f"[!] Error: One or both directories do not exist.")
        return

    # Added .cap to the supported extensions
    extensions = ('.pcap', '.pcapng', '.cap')

    # Phase 1: Recursive Flattening
    print(f"[*] Starting Phase 1: Flattening {pcaps2_root}...")
    for file_path in list(pcaps2_root.rglob('*')):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            destination = pcaps2_root / file_path.name
            
            if file_path.parent == pcaps2_root:
                continue
                
            if destination.exists():
                # Use a small hash of the path to avoid filename collisions during flattening
                path_suffix = hashlib.md5(str(file_path).encode()).hexdigest()[:6]
                destination = pcaps2_root / f"{file_path.stem}_{path_suffix}{file_path.suffix}"
            
            shutil.move(str(file_path), str(destination))

    # Phase 2: Cleanup Empty Subfolders
    for root, dirs, files in os.walk(pcaps2_root, topdown=False):
        for name in dirs:
            dir_path = Path(root) / name
            try:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
            except OSError:
                pass

    # Phase 3: Counting consolidated files
    current_files = [f for f in pcaps2_root.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    print(f"[+] Total files consolidated in {pcaps2_root}: {len(current_files)}")

    # Phase 4: Indexing Reference Folder with Caching
    print(f"[*] Starting Phase 2: Indexing reference files in {pcaps_ref_root}...")
    cache_file = pcaps_ref_root / ".hash_cache.txt"
    reference_hashes = set()
    
    # Load existing hashes from cache if it exists
    if cache_file.exists():
        with open(cache_file, "r") as f:
            reference_hashes = set(line.strip() for line in f if line.strip())
        print(f"[*] Loaded {len(reference_hashes)} existing hashes from cache.")

    # Find files and update cache for missing ones
    updated_cache = False
    for ref_file in pcaps_ref_root.rglob('*'):
        if ref_file.is_file() and ref_file.suffix.lower() in extensions:
            # Check if file modification time or existence needs re-hashing logic
            # For simplicity, we hash if not in current set
            file_hash = calculate_sha256(ref_file)
            if file_hash and file_hash not in reference_hashes:
                reference_hashes.add(file_hash)
                updated_cache = True
    
    # Save back to cache if we found new files
    if updated_cache:
        with open(cache_file, "w") as f:
            for h in reference_hashes:
                f.write(f"{h}\n")
        print(f"[*] Cache updated and saved to {cache_file}.")

    # Phase 5: Internal and External Deduplication
    print(f"[*] Starting Phase 3: Deduplicating {pcaps2_root}...")
    seen_in_pcaps2 = set()
    removed_external = 0  # Exists in 'pcaps'
    removed_internal = 0  # Exists multiple times within 'pcaps2'

    for pcap_file in current_files:
        current_hash = calculate_sha256(pcap_file)
        if not current_hash:
            continue

        # Check if it exists in the reference folder (pcaps)
        if current_hash in reference_hashes:
            print(f"[-] Deleting {pcap_file.name}: Matches reference in {reference_dir_name}")
            pcap_file.unlink()
            removed_external += 1
        
        # Check if we have already encountered this file within pcaps2 itself
        elif current_hash in seen_in_pcaps2:
            print(f"[-] Deleting {pcap_file.name}: Internal duplicate within {source_dir_name}")
            pcap_file.unlink()
            removed_internal += 1
        
        else:
            # First time seeing this unique file in pcaps2
            seen_in_pcaps2.add(current_hash)

    print(f"\n[✔] Execution Summary:")
    print(f"    - Removed (Already in {reference_dir_name}): {removed_external}")
    print(f"    - Removed (Duplicate within {source_dir_name}): {removed_internal}")
    print(f"    - Final unique files kept in {source_dir_name}: {len(list(pcaps2_root.glob('*')))}")

    # -----------------------------
    # Added Phase 6: Corruption Validation & Cleanup (post-processing)
    # -----------------------------
    import gc  # Garbage Collector to force handle release
    from scapy.utils import RawPcapReader, Scapy_Exception # type: ignore

    def is_pcap_corrupt(file_path):
        """
        Attempts to validate the PCAP. 
        Returns True if corrupt, False if valid.
        """
        reader = None
        try:
            reader = RawPcapReader(str(file_path))
            # Try to read the first packet to check internal structure
            next(reader, None)
            return False
        except (Scapy_Exception, EOFError, OSError, StopIteration):
            return True
        finally:
            # CRITICAL: Explicitly close the file handle
            if reader is not None:
                reader.close()
            del reader # Remove the reference

    def cleanup_corrupted_pcaps(target_folder):
        target_path = Path(target_folder).resolve()
        
        extensions = ('.pcap', '.pcapng', '.cap')
        files_deleted = 0

        print(f"[*] Analyzing files in: {target_path}...")

        # We convert to a list to avoid issues while iterating and deleting
        for file in list(target_path.iterdir()):
            if file.is_file() and file.suffix.lower() in extensions:
                
                corrupt = is_pcap_corrupt(file)
                
                if corrupt:
                    print(f"[!] Corrupted file detected: {file.name}. Attempting deletion...")
                    
                    # Small trick for Windows: force garbage collection to release 
                    # any lingering lazy handles from Scapy
                    gc.collect() 

                    try:
                        file.unlink()
                        print(f"    [✔] Successfully deleted.")
                        files_deleted += 1
                    except PermissionError:
                        print(f"    [✘] Permission Denied: File is locked by another app (Wireshark or System).")
                    except Exception as e:
                        print(f"    [✘] Error: {e}")

        print(f"\n[✔] Done. Deleted {files_deleted} files.")

    # Validate and delete corrupted files after all dedup operations
    cleanup_corrupted_pcaps(source_dir_name)

if __name__ == "__main__":
    process_pcaps(SOURCE, REFERENCE)
