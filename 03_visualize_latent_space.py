import os
import json
import random
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from tqdm import tqdm
from sklearn.manifold import TSNE

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "02_z_industrial_vae_full.pth"  # Ensure this matches your saved VAE model name
DATA_PATH = "dataset.json"
MAX_LEN = 256                # Must match training script
LATENT_DIM = 32              # Must match training script
MAX_SAMPLES_PER_PROTO = 300  # Cap to prevent t-SNE hairballs and speed up computation
RANDOM_SEED = 42

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Determine processing device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        return device, torch.cuda.get_device_name(0)
    else:
        return torch.device("cpu"), "CPU"
device, device_name = get_device()
print(f"[*] Executing on device: {device_name}")

# ==========================================
# 2. DEFINE MODEL ARCHITECTURE (Must match 06_VAE_training_gpu.py)
# ==========================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32, max_len=256):
        super(ConvVAE, self).__init__()
        
        self.compressed_len = max_len // 8
        self.flattened_size = 128 * self.compressed_len
        self.latent_dim = latent_dim
        
        # Encoder: Must match training script exactly
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten() 
        )
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder: Must include Unflatten at index 0 to match .pth indices
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, self.compressed_len)), # Index 0
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # Index 1
            nn.BatchNorm1d(64), # Index 2
            nn.ReLU(),          # Index 3
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # Index 4
            nn.BatchNorm1d(32), # Index 5
            nn.ReLU(),          # Index 6
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)  # Index 7
        )

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        # Matches the flow in training: Linear -> Sequential
        logits = self.decoder(self.decoder_input(z))
        return logits.squeeze(1)


# ==========================================
# 3. BALANCED CAPPED DATA LOAD & PREPROCESSING
# ==========================================
def load_and_sample_dataset(data_path, max_len, max_samples_per_proto):
    print(f"[*] Loading dataset from {data_path}...")
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    
    # Group samples by protocol
    proto_groups = {}
    for item in raw_data:
        proto = item["protocol"]
        if proto not in proto_groups:
            proto_groups[proto] = []
        proto_groups[proto].append(item["payload_hex"])
        
    print(f"[+] Total unique protocols detected: {len(proto_groups)}")
    
    selected_payloads = []
    selected_labels = []
    
    # Apply balanced capping
    for proto, payloads in proto_groups.items():
        sample_size = min(len(payloads), max_samples_per_proto)
        sampled = random.sample(payloads, sample_size)
        
        selected_payloads.extend(sampled)
        selected_labels.extend([proto] * sample_size)
        print(f"    - {proto:20} : Sampled {sample_size}/{len(payloads)}")

    # Preprocess payloads to float tensors normalized to [0, 1]
    print("[*] Encoding and normalizing payloads to tensor format...")
    processed_tensors = []
    for payload_hex in tqdm(selected_payloads, desc="Normalizing bytes"):
        # Convert hex string back to integer list
        byte_list = [int(payload_hex[i:i+2], 16) for i in range(0, len(payload_hex), 2)]
        
        # Pad or truncate to MAX_LEN
        if len(byte_list) < max_len:
            byte_list += [0] * (max_len - len(byte_list))
        else:
            byte_list = byte_list[:max_len]
            
        # Normalize range to [0.0, 1.0]
        norm_arr = np.array(byte_list, dtype=np.float32) / 255.0
        processed_tensors.append(norm_arr)
        
    x_data = torch.tensor(np.array(processed_tensors)).unsqueeze(1) # Shape: [B, 1, MAX_LEN]
    return x_data, selected_labels

# ==========================================
# 4. EMBEDDING EXTRACTION
# ==========================================
def extract_embeddings(model, data_tensor, batch_size=128):
    model.eval()
    embeddings = []
    
    print("[*] Extracting latent mean coordinates (mu)...")
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size].to(device)
            mu, _ = model.encode(batch)
            embeddings.append(mu.cpu().numpy())
            
    return np.vstack(embeddings)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # 1. Load data
    x_data, labels = load_and_sample_dataset(DATA_PATH, MAX_LEN, MAX_SAMPLES_PER_PROTO)
    
    # 2. Reconstruct Model and Load Weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[-] Saved model weights not found at {MODEL_PATH}. Make sure your training script completed.")
        
    print(f"[*] Initializing ConvVAE and loading weights from '{MODEL_PATH}'...")
    model = ConvVAE(max_len=MAX_LEN, latent_dim=LATENT_DIM).to(device)    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    print("[+] Model loaded successfully.")
    
    # 3. Extract Embeddings
    latent_space = extract_embeddings(model, x_data)
    print(f"[+] Extraction complete. Shape of Latent Matrix: {latent_space.shape}")
    
    # 4. Dimensionality Reduction (t-SNE)
    print(f"[*] Starting t-SNE dimensionality reduction (16D -> 2D)...")
    print("    [Info] Perplexity=30, Iterations=1000. Please wait...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        learning_rate="auto",
        init="pca",  # PCA initialization stabilizes global structures in t-SNE
        random_state=RANDOM_SEED,
        n_jobs=-1     # Use all available CPU cores for speed
    )
    tsne_results = tsne.fit_transform(latent_space)
    print("[+] t-SNE reduction complete.")
    
    # 5. Build Visualization DataFrame
    df = pd.DataFrame({
        "t-SNE Component 1": tsne_results[:, 0],
        "t-SNE Component 2": tsne_results[:, 1],
        "Protocol": labels
    })
    
    # Sort protocols alphabetically so colors match reliably across plots
    df = df.sort_values(by="Protocol")
    
    # ==========================================
    # 5. PUBLICATION-QUALITY PLOTTING
    # ==========================================
    print("[*] Generating publication-grade visualization plots...")
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid") # Clean white grid for engineering theses
    
    # Create an rich, distinct 21-color palette
    palette = sns.color_palette("husl", len(df["Protocol"].unique()))
    
    # Plot Scatter
    scatter = sns.scatterplot(
        x="t-SNE Component 1", 
        y="t-SNE Component 2",
        hue="Protocol",
        palette=palette,
        data=df,
        alpha=0.85,
        s=25,          # Size of markers
        edgecolor="none"
    )
    
    plt.title(
        f"2D t-SNE Projection of VAE Latent Space ($z \\in \\mathbb{{R}}^{{{LATENT_DIM}}}$)\n"
        f"Model: ConvVAE (Input Max Length: {MAX_LEN} Bytes)",
        fontsize=16, fontweight="bold", pad=20
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=12, labelpad=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=12, labelpad=10)
    
    # Place Legend cleanly on the right hand side
    plt.legend(
        bbox_to_anchor=(1.04, 1), 
        loc="upper left", 
        borderaxespad=0, 
        title="Industrial Protocols",
        title_fontsize=12,
        fontsize=10,
        markerscale=2.0
    )
    
    plt.tight_layout()
    
    # Save Outputs
    png_filename = "03_z_latent_space_tsne.png"
    pdf_filename = "03_z_latent_space_tsne.pdf"
    
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    
    print(f"[+] Success! Plot saved as raster image: '{png_filename}' (300 DPI)")
    print(f"[+] Success! Plot saved as vector graphics: '{pdf_filename}'")
    plt.close()

if __name__ == "__main__":
    main()
