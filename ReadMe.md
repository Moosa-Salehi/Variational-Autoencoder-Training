# VAE Training Scripts

A multi-stage pipeline for a Convolutional Beta Variational Autoencoder (Conv-βVAE) training for industrial traffic.

## Pipeline Flow

1. `01_analyze_payload_lengths.py`: Statistical analysis of payload length distributions to inform VAE input dimensions.
2. `02_VAE_training_gpu.py`: VAE model training. Optimized for NVIDIA RTX 3050 (Ampere architecture).
3. `03_visualize_latent_space.py`: Extracts latent embeddings from the trained ConvVAE, applies t-SNE dimensionality reduction, and generates visualizations.

## System Requirements

| OS Recommended | Hardware |
| :--- | :--- |
| Windows/Linux | NVIDIA GPU (RTX 3050 4GB/8GB) - 16GB+ RAM recommended |

## Installation & Setup

```bash
# 1. Install GPU-accelerated PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest of the pipeline dependencies
pip install -r requirements.txt

# if failed, try this:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## How To Use

- prepare dataset as `dataset.josn` in repo root, then run scripts in order.
