import os, sys, json, time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from scapy.all import PcapReader, IP, TCP, UDP, defragment # type: ignore

# --- Configuration ---
INPUT_DIR = "../pcaps/"
CSV_DIR = "./ndpi_csv/"
OUTPUT_DIR = "./industrial_messages/"
CONCURRENT_FILES = 10
GLOBAL_MAX_PAYLOADS = 200000 

TARGET_PROTOCOLS = {
    "Modbus",               # The "de facto" standard for serial/TCP industrial comms
    "Modbus.UMAS",          # Schneider Electric proprietary extension for Modbus
    "IEC60870",             # Telecontrol protocol (e.g., IEC 104) used in power grids
    "DNP3",                 # Distributed Network Protocol (Power/Water utilities)
    "BACnet",               # Building Automation and Control networks
    "S7Comm",               # Siemens S7-300/400 communication
    "S7CommPlus",           # Siemens S7-1200/1500 (TIA Portal) secured communication
    "CIP",                  # Common Industrial Protocol (Common layer for EtherNet/IP)
    "EthernetIP",           # Industrial Ethernet (Rockwell Automation/ODVA)
    "IEC62056",             # DLMS/COSEM (Smart Metering)
    "ISO9506-1-MMS",        # Manufacturing Message Specification (used in IEC 61850)
    "OPC-UA",               # Open Platform Communications Unified Architecture
    "COAP",                 # Constrained Application Protocol (often used in IIoT)
    "MQTT",                 # Message Queuing Telemetry Transport (IIoT/SCADA)
    "RTPS",                 # Real-Time Publish-Subscribe (Backbone for Data Distribution Service - DDS)
    "IEEE-C37118",          # Synchrophasor data transfer (Power grid monitoring)
    "HART-IP",              # Highway Addressable Remote Transducer over IP
    "FINS",                 # Omron Factory Interface Network Service
    "KNXnet_IP",            # Building automation standard for control of lighting/HVAC
    "MELSEC",               # Mitsubishi Electric PLC protocol
    "BeckhoffADS",          # Beckhoff Automation Device Specification
    "ANSI_C1222",           # Smart grid protocol for interfacing data communication networks
    "DCERPC.PROFINET_IO",   # Profinet IO (Industrial Ethernet for Siemens/others)
    "EtherSIO",             # Ethernet Serial I/O
    "Ether-S-Bus",          # Saia-Burgess Controls S-Bus
    "TriStation",           # Triconex (Schneider Electric) Safety Instrumented Systems
    "PTPv2",                # Precision Time Protocol (IEEE 1588) crucial for industrial sync
    "SOMEIP",               # Scalable service-Oriented MiddlewarE over IP (Automotive Ethernet)
    "RMCP",                 # Remote Management and Control Protocol (used in IPMI/industrial hardware)
    "CNP-IP",               # CEA-709.1 (LonWorks over IP) for building automation
}

def load_flow_index(pcap_filename):
    csv_path = os.path.join(CSV_DIR, f"{pcap_filename}_ndpi.csv")
    if not os.path.exists(csv_path):
        return {}

    flow_index = {}
    try:
        df = pd.read_csv(csv_path, sep='|', low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        proto_col = 'ndpi_proto'
        
        for _, row in df.iterrows():
            proto = str(row.get(proto_col, ''))
            if proto in TARGET_PROTOCOLS:
                src_ip, dst_ip = str(row['src_ip']).strip(), str(row['dst_ip']).strip()
                sport, dport = int(row['src_port']), int(row['dst_port'])
                flow_index[(src_ip, dst_ip, sport, dport)] = proto
                flow_index[(dst_ip, src_ip, dport, sport)] = proto
        return flow_index
    except:
        return {}

def process_single_pcap(filename, global_counts, lock):
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
    if os.path.exists(output_path):
        return # already exists.
    
    flow_index = load_flow_index(filename)
    if not flow_index:
        return # no csv or industrial flow detected.

    # Optimization: know which protocols we are looking for in this specific file
    protos_in_this_file = set(flow_index.values())
    
    input_path = os.path.join(INPUT_DIR, filename)
    
    flow_sequences = {}
    fragments = []
    stats = {p: 0 for p in TARGET_PROTOCOLS}
    local_count = 0

    try:
        with open(output_path, 'w') as f_out:
            f_out.write("[\n")
            first_entry = True

            with PcapReader(input_path) as reader:
                for pkt in reader:
                    # PERFORMANCE CHECK: If all protocols in this file reached global limit, stop reading PCAP
                    if all(global_counts.get(p, 0) >= GLOBAL_MAX_PAYLOADS for p in protos_in_this_file):
                        break

                    # 1. IP Fragmentation Reassembly
                    if pkt.haslayer(IP) and (pkt[IP].flags == "MF" or pkt[IP].frag > 0):
                        fragments.append(pkt)
                        if len(fragments) > 50:
                            defrag = defragment(fragments)
                            for rp in [p for p in defrag if not (p.haslayer(IP) and (p[IP].flags == "MF" or p[IP].frag > 0))]:
                                if not (rp.haslayer(TCP) or rp.haslayer(UDP)): continue
                                f_id = (rp[IP].src, rp[IP].dst, rp.sport, rp.dport)
                                p_name = flow_index.get(f_id)
                                if p_name and global_counts.get(p_name, 0) < GLOBAL_MAX_PAYLOADS:
                                    pay = bytes(rp[TCP].payload if rp.haslayer(TCP) else rp[UDP].payload)
                                    if not pay: continue
                                    with lock:
                                        if global_counts.get(p_name, 0) >= GLOBAL_MAX_PAYLOADS: continue
                                        global_counts[p_name] += 1
                                    
                                    stats[p_name] += 1
                                    entry = {"timestamp": float(rp.time), "protocol": p_name, "src": f"{f_id[0]}:{f_id[2]}", "dst": f"{f_id[1]}:{f_id[3]}", "payload_hex": pay.hex()}
                                    if not first_entry: f_out.write(",\n")
                                    json.dump(entry, f_out); first_entry = False; local_count += 1
                            fragments = [p for p in defrag if (p.haslayer(IP) and (p[IP].flags == "MF" or p[IP].frag > 0))]
                        continue

                    # 2. Standard Packet Processing
                    if not (pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP))):
                        continue
                    
                    flow_id = (pkt[IP].src, pkt[IP].dst, pkt.sport, pkt.dport)
                    proto_name = flow_index.get(flow_id)
                    
                    if proto_name:
                        # Skip if this specific protocol reached its limit
                        if global_counts.get(proto_name, 0) >= GLOBAL_MAX_PAYLOADS:
                            continue

                        payload = bytes(pkt[TCP].payload if pkt.haslayer(TCP) else pkt[UDP].payload)
                        if not payload: continue

                        # TCP De-duplication
                        if pkt.haslayer(TCP):
                            seq = pkt[TCP].seq
                            if flow_sequences.get(flow_id, 0) > 0 and seq < flow_sequences[flow_id]:
                                continue
                            flow_sequences[flow_id] = seq + len(payload)

                        # Counter increment
                        with lock:
                            if global_counts.get(proto_name, 0) >= GLOBAL_MAX_PAYLOADS:
                                continue
                            global_counts[proto_name] += 1
                        
                        stats[proto_name] += 1
                        local_count += 1

                        msg_entry = {
                            "timestamp": float(pkt.time),
                            "protocol": proto_name,
                            "src": f"{pkt[IP].src}:{pkt.sport}",
                            "dst": f"{pkt[IP].dst}:{pkt.dport}",
                            "payload_hex": payload.hex()
                        }
                        if not first_entry: f_out.write(",\n")
                        json.dump(msg_entry, f_out)
                        first_entry = False
            
            f_out.write("\n]")

        # Cleanup and Output Printing
        if local_count == 0:
            if os.path.exists(output_path): os.remove(output_path)
        else:
            # Construct the block
            output_lines = [f"[+] Finished: {filename}"]
            active_stats = {k: v for k, v in stats.items() if v > 0}
            for proto, count in active_stats.items():
                is_capped = global_counts.get(proto, 0) >= GLOBAL_MAX_PAYLOADS
                limit_msg = " (MAX REACHED)" if is_capped else ""
                output_lines.append(f"    - {proto}: {count} messages{limit_msg}")
            
            # Use the existing multiprocessing lock to ensure exclusive access to stdout
            full_message = "\n".join(output_lines) + "\n"
            with lock:
                sys.stdout.write(full_message)
                sys.stdout.flush()

    except Exception as e:
        with lock:
            sys.stdout.write(f"[!] Error processing {filename}: {e}\n")
            sys.stdout.flush()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    pcaps = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.pcap', '.pcapng'))])
    if not pcaps: 
        print(f"No files in {INPUT_DIR}")
        return

    print(f"[*] Starting extraction for {len(pcaps)} files...")
    print(f"[*] Global Limit: {GLOBAL_MAX_PAYLOADS} payloads per protocol.")
    print(f"[*] Extracting Protocols: {", ".join(TARGET_PROTOCOLS)}")
    
    with Manager() as manager:
        global_counts = manager.dict({p: 0 for p in TARGET_PROTOCOLS})
        lock = manager.Lock()
        
        print(f"[*] Spawning {CONCURRENT_FILES} workers...")
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=CONCURRENT_FILES) as executor:
            futures = [executor.submit(process_single_pcap, f, global_counts, lock) for f in pcaps]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"[CRASH] Worker failed: {e}")
        print(f"\n[DONE] Extraction finished in {time.time() - start_time:.2f} seconds.")
        print("Final Global Protocol Counts:")
        for proto, count in global_counts.items():
            print(f" - {proto}: {count}")

if __name__ == "__main__":
    main()
