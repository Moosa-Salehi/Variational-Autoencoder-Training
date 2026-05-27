import os
import csv
import sys
from collections import Counter

# Increase the CSV field size limit to handle large nDPI metadata fields
# This prevents _csv.Error: field larger than field limit
max_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_limit)
        break
    except OverflowError:
        # Reduce size until it fits into a C long
        max_limit = int(max_limit / 10)

CSV_DIR = "./ndpi_csv/"
OUTPUT_FILE = "02_z_detected_protocols.txt"

def analyze_protocols():
    protocol_counter = Counter()
    
    # Verify if the target directory exists before proceeding
    if not os.path.exists(CSV_DIR):
        print(f"Error: Directory {CSV_DIR} does not exist.")
        return

    # Using os.scandir for better performance with large numbers of files
    try:
        for entry in os.scandir(CSV_DIR):
            if entry.is_file() and entry.name.endswith(".csv"):
                # Open with utf-8 and ignore errors to handle non-ASCII characters in network payloads
                with open(entry.path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f, delimiter='|')
                    try:
                        for row in reader:
                            proto = row.get('ndpi_proto')
                            if proto:
                                protocol_counter[proto] += 1
                                # Optional: Debug specific protocols like DNP3
                                # if proto == 'DNP3':
                                #     print(f"Found DNP3 in: {entry.name}")
                    except csv.Error as e:
                        print(f"Parsing error in {entry.name}: {e}")
                        
    except Exception as e:
        print(f"OS error during directory traversal: {e}")

    with open(OUTPUT_FILE, 'a') as f:
        # Output formatted results
        s = "\n--- Protocol Flow Counts ---"
        print(s)
        f.write(s + '\n')
        # sorted by frequency (most common first)
        for proto, count in protocol_counter.most_common():
            s = f"{proto:25}: {count} flows"
            print(s)
            f.write(s + '\n')

def main():
    analyze_protocols()

if __name__ == "__main__":
    main()
