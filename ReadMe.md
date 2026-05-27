# Industrial Protocol Analysis & VAE Training Pipeline

A multi-stage pipeline for DPI-based protocol discovery, payload extraction, and Variational Autoencoder (VAE) training for industrial traffic.

## Pipeline Flow

0. `00_pcap_flatten_deduplicate_validate.py`: Flattens, deduplicates based on `SOURCE` and `REFERENCE` pcaps, and validates pcaps. Usefull if more pcaps are added to pipeline over time. The `REFERENCE` folder is the folder containing the existing pcaps in pipeline, and the `SOURCE` folder contains the new pcaps. After setting the `SOURCE` and `REFERENCE` folders and running the script, the remaining pcaps in `SOURCE` folder can be added to `REFERENCE` folder as actuall new pcaps.
1. `01_run_dpi.py`: Orchestrates `ndpiReader` to process PCAP files. Generates structured CSV metadata. Set the folder containing the pcaps in `INPUT_DIR`.
2. `02_get_protocol_names.py`: Aggregates and reports detected protocol distributions and flow counts.
3. `03_extract_industrial_messages.py`: High-performance industrail protocols payload extraction using `Scapy`. Handles TCP reassembly (segmentation/retransmission) and IP fragmentation. Exports up to 200K payloads per protocol to JSON. Need to set `INPUT_DIR`.
4. `04_integrate_payloads.py`: Dataset curation: Deduplication, class balancing, filtering, and shuffling into `integrated_vae_dataset.json`.
5. `05_analyze_message_lengths.py`: Statistical analysis of payload length distributions to inform VAE input dimensions.
6. `06_VAE_training_gpu.py`: VAE model training. Optimized for NVIDIA RTX 3050 (Ampere architecture).
6. `07_visualize_latent_space.py`: Extracts latent embeddings from the trained ConvVAE, applies t-SNE dimensionality reduction, and generates publication-grade visualizations.

## System Requirements

| Stage | OS Recommended | Hardware |
| :--- | :--- | :--- |
| **Stage 1 (DPI)** | Linux (Ubuntu/Debian) | High I/O throughput |
| **Stages 2-5** | Linux / Windows | 16GB+ RAM recommended |
| **Stage 6-7 (Training/Visualization)**| Windows/Linux | NVIDIA GPU (RTX 3050 4GB/8GB) |

## Installation & Setup

### Python Environment
```bash
# 1. Install GPU-accelerated PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest of the pipeline dependencies
pip install -r requirements.txt

# if failed, try this:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### nDPI Installation (Linux)
Required for Stage 1.
```bash
# 1. Install build dependencies (Required for compiling from source)
sudo apt-get update
sudo apt-get install -y build-essential git bison flex libpcap-dev libtool libtool-bin autoconf pkg-config automake gettext

# 2. Download and build the stable v5.0 release
wget https://github.com/ntop/nDPI/archive/refs/tags/5.0.tar.gz
tar -zxvf 5.0.tar.gz
cd nDPI-5.0

# 3. Compile and Install
./autogen.sh
./configure
make
sudo make install

# 4. Update shared library cache (Crucial step after installation)
sudo ldconfig
```
