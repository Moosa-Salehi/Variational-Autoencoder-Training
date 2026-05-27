import os
import subprocess
import concurrent.futures

INPUT_DIR = "../pcaps/"
CSV_DIR = "./ndpi_csv/"
NDPI_READER = "ndpiReader"

def process_pcap(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    csv_path = os.path.join(CSV_DIR, f"{filename}_ndpi.csv")
    if os.path.exists(csv_path):
        return f"Skipped: {filename} (CSV already exists)"
    try:
        subprocess.run([NDPI_READER, "-i", input_path, "-C", csv_path], check=True)
        return f"Successfully processed {filename}"
    except subprocess.CalledProcessError as e:
        return f"Error processing {filename}: {e}"

def main():
    for directory in [CSV_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    pcaps = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.pcap', '.pcapng'))]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_pcap, pcaps)
    
    print("[+] All Done.")

if __name__ == "__main__":
    main()
