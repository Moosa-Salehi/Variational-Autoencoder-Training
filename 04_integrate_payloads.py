import os
import json
import random
from collections import defaultdict

# --- Configuration ---
INPUT_DIR = "./industrial_messages/"
OUTPUT_FILE = "04_z_integrated_vae_dataset.json"
STATS_FILE = "04_z_dataset_info.txt"

MIN_PAYLOAD_LEN = 4       # Bytes (Filter out tiny packets)
MIN_PER_PROTOCOL = 100    # Prune protocols with fewer than this many unique samples
MAX_PER_PROTOCOL = 200000  # Cap protocols for balancing

def integrate_payloads():
    unique_data = defaultdict(set)
    total_raw_count = 0

    # 1. Extraction & Deduplication
    if not os.path.exists(INPUT_DIR):
        print(f"[!] Error: {INPUT_DIR} not found.")
        return

    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"[*] Analyzing {len(json_files)} JSON files...")

    for filename in json_files:
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                messages = json.load(f)
                for msg in messages:
                    total_raw_count += 1
                    proto = msg['protocol']
                    payload = msg['payload_hex']
                    # Filter by length (hex string length is 2x byte length)
                    if len(payload) >= (MIN_PAYLOAD_LEN * 2):
                        unique_data[proto].add(payload)
        except Exception as e:
            print(f"    [!] Error reading {filename}: {e}")

    # 2. Filtering and Balancing Logic
    final_collection = []
    final_stats = []   # List of (proto, count) for Table 2
    
    # Sort protocols by the size of their payload set (Descending)
    # unique_data.items() returns (protocol_name, set_of_payloads)
    sorted_raw_items = sorted(unique_data.items(), key=lambda x: len(x[1]), reverse=True)

    # 3. Generate Report Strings
    report = []
    report.append("="*60)
    report.append("PHASE 1: RAW DATA & DEDUPLICATION (Sorted by Count)")
    report.append(f"{'Protocol':<25} | {'Unique Payloads':<15}")
    report.append("-"*60)
    
    for proto, payloads_set in sorted_raw_items:
        unique_count = len(payloads_set)
        report.append(f"{proto:<25} | {unique_count:<15}")
        
        # Apply Thresholds
        if unique_count >= MIN_PER_PROTOCOL:
            payload_list = list(payloads_set)
            if unique_count > MAX_PER_PROTOCOL:
                payload_list = random.sample(payload_list, MAX_PER_PROTOCOL)
            
            final_count = len(payload_list)
            final_stats.append((proto, final_count))
            
            for p in payload_list:
                final_collection.append({"protocol": proto, "payload_hex": p})

    # Sort Table 2 by count descending
    final_stats.sort(key=lambda x: x[1], reverse=True)
    random.shuffle(final_collection)

    report.append("\n" + "="*60)
    report.append(f"PHASE 2: FINAL DATASET (Min: {MIN_PER_PROTOCOL}, Max: {MAX_PER_PROTOCOL})")
    report.append(f"{'Protocol':<25} | {'Final Count':<15}")
    report.append("-"*60)
    for proto, count in final_stats:
        report.append(f"{proto:<25} | {count:<15}")
    
    report.append("-"*60)
    report.append(f"TOTAL SAMPLES IN INTEGRATED DATASET: {len(final_collection)}")
    report.append("="*60)

    # Output and Persistence
    full_report_text = "\n".join(report)
    print(full_report_text)
    
    with open(STATS_FILE, 'w') as f_stats:
        f_stats.write(full_report_text)

    with open(OUTPUT_FILE, 'w') as f_out:
        json.dump(final_collection, f_out, indent=2)

    print(f"\n[+] Integrated dataset saved: {OUTPUT_FILE}")
    print(f"[+] Technical stats saved: {STATS_FILE}")

if __name__ == "__main__":
    integrate_payloads()
