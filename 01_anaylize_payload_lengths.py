import json
import numpy as np
from collections import Counter
from tqdm import tqdm

DATASET_PATH = "dataset.json"

if __name__ == "__main__":
    print(f"Loading data from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        exit()

    print(f"Processing {len(raw_data)} items...")
    
    payload_lengths = []
    for item in tqdm(raw_data, desc="Calculating lengths"):
        if 'payload_hex' in item and isinstance(item['payload_hex'], str):
            # Corrected: Hex string length / 2 = Byte length
            byte_len = len(item['payload_hex']) // 2
            payload_lengths.append(byte_len)
        else:
            continue

    if not payload_lengths:
        print("No valid payload lengths found. Exiting.")
        exit()

    lengths_np = np.array(payload_lengths)

    # --- Basic Statistics ---
    print("\n--- Payload Length Statistics (Bytes) ---")
    print(f"Total Payloads: {len(lengths_np)}")
    print(f"Min: {np.min(lengths_np)} bytes")
    print(f"Max: {np.max(lengths_np)} bytes")
    print(f"Average: {np.mean(lengths_np):.2f} bytes")
    print(f"Median: {np.median(lengths_np)} bytes")
    print(f"Standard Deviation: {np.std(lengths_np):.2f} bytes")

    # --- Coverage Analysis ---
    pct_128 = (np.sum(lengths_np <= 128) / len(lengths_np)) * 100
    pct_256 = (np.sum(lengths_np <= 256) / len(lengths_np)) * 100
    lost_256 = len(lengths_np[lengths_np > 256])

    print("\n--- Coverage Analysis ---")
    print(f"Coverage at 128 bytes: {pct_128:.2f}%")
    print(f"Coverage at 256 bytes: {pct_256:.2f}%")
    print(f"Packets lost (truncated) > 256 bytes: {lost_256}")

    # --- Top 10 Distribution ---
    length_counts = Counter(payload_lengths)
    most_common = length_counts.most_common(10)

    print("\n--- Top 10 Most Common Payload Lengths ---")
    for length, count in most_common:
        print(f"Length: {length:<4} bytes | Count: {count:<8} | Percentage: {(count/len(lengths_np))*100:.2f}%")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns # type: ignore

        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(lengths_np, bins=50, kde=True, color='skyblue')
        plt.title('Payload Length Distribution (Bytes)')
        plt.xlabel('Bytes')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(1, 2, 2)
        top_lengths = [l for l, c in most_common]
        top_counts = [c for l, c in most_common]
        sns.barplot(x=top_lengths, y=top_counts, hue=top_lengths, palette='viridis', legend=False)
        plt.title('Top 10 Most Common Lengths')
        plt.xlabel('Length (Bytes)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("01_payload_length_distribution.png")
        print("\n[+] Plot saved as '01_payload_length_distribution.png'")

    except ImportError:
        print("\n[!] Matplotlib/Seaborn not found. Skipping plot.")
