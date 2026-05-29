import json, os, time
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler # type: ignore
from tqdm import tqdm
from torch.amp import autocast, GradScaler # type: ignore
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

# --- Hyperparameters ---
MAX_LEN = 256     # according to payload length statistics
LATENT_DIM = 32
BETA = 4.0
BATCH_SIZE = 256  # Increased for GPU efficiency (RTX 3050 can handle this with AMP)
EPOCHS = 20
LEARNING_RATE = 1e-3
SAMPLER_POWER = 0.5  # 0.0 = natural distribution, 1.0 = fully balanced, 0.5 = square-root balanced (optimal)

# --- File Path's ---
DATASET_PATH = "dataset.json"
OUTPUT_MODEL_NAME = "02_z_industrial_vae_full.pth"
OUTPUT_ENCODER_NAME = "02_z_industrial_encoder_only.pth"

# --- Device Optimization ---
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        return device, torch.cuda.get_device_name(0)
    else:
        return torch.device("cpu"), "CPU"

# --- 1. Optimized Dataset ---
class IndustrialDataset(Dataset):
    def __init__(self, json_file, max_len=128):
        self.max_len = max_len
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Memory optimization: Store as uint8 (0-255) to save 75% RAM
        # We only convert to float32 [0, 1] during __getitem__
        self.features = np.zeros((len(raw_data), max_len), dtype=np.uint8)
        self.labels = []
        
        for i, item in enumerate(tqdm(raw_data, desc="Preprocessing Payloads")):
            raw_bytes = bytes.fromhex(item['payload_hex'])
            # Truncate/Pad bytes
            numeric_msg = np.frombuffer(raw_bytes, dtype=np.uint8)
            length = min(len(numeric_msg), max_len)
            self.features[i, :length] = numeric_msg[:length]
            self.labels.append(item.get('protocol', 'unknown'))
            
        del raw_data # Clear memory immediately
        print(f"[+] Dataset ready: {len(self.features)} samples.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to float and normalize on-the-fly (saves RAM)
        x = torch.from_numpy(self.features[idx]).float() / 255.0
        return x, self.labels[idx]

# --- 2. Convolutional VAE Architecture ---
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32, max_len=256):
        super(ConvVAE, self).__init__()
        
        # Calculate compressed length: 3 layers of stride 2 = divide by 8
        self.compressed_len = max_len // 8
        self.flattened_size = 128 * self.compressed_len
        
        # Encoder: (Batch, 1, 128) -> Extract spatial byte patterns
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
        
        # Latent Space Mapping
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder: Latent -> Reconstructed Logits
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, self.compressed_len)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.unsqueeze(1) 
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(self.decoder_input(z))
        return logits.squeeze(1), mu, logvar

# --- 3. Loss Function ---
def loss_function(logits, x, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (BETA * kld_loss)

# --- 4. Training Loop (Optimized for GPU) ---
def train_model():
    # Pass MAX_LEN to dataset
    full_dataset = IndustrialDataset(DATASET_PATH, max_len=MAX_LEN)
    
    # Stratified Train-Validation Split (90/10) to preserve minority protocol representation
    indices = np.arange(len(full_dataset))
    labels = np.array(full_dataset.labels)
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=labels
    )
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    DEVICE, device_name = get_device()
    print(f"[+] Training on: {device_name}")
    
    # Calculate balanced sample weights for training set to handle extreme class imbalance
    train_labels = [full_dataset.labels[i] for i in train_idx]
    class_counts = Counter(train_labels)
    
    # Sub-linear inverse scaling: count^SAMPLER_POWER mitigates massive repetition of rare classes (e.g., Modbus.UMAS)
    class_weights = {cls: 1.0 / (count ** SAMPLER_POWER) for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # pin_memory=True speeds up transfer to GPU. 
    # num_workers > 0 uses multi-process data loading.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,  # Implements Balanced Batch Sampling
        num_workers=4, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Pass MAX_LEN to model
    model = ConvVAE(latent_dim=LATENT_DIM, max_len=MAX_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Initialize Gradient Scaler for Mixed Precision
    scaler = GradScaler()
    
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for data, _ in pbar:
            data = data.to(DEVICE, non_blocking=True) # non_blocking works with pin_memory
            optimizer.zero_grad(set_to_none=True)     # Slightly faster than zero_grad()
            
            # Autocast handles the mixed precision operations
            with autocast(device_type=DEVICE.type):
                logits, mu, logvar = model(data)
                loss = loss_function(logits, data, mu, logvar)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'Loss': total_train_loss / ((pbar.n + 1) * BATCH_SIZE)})
        
        # Validation Phase
        model.eval()
        total_val_loss = 0
        
        # Diagnostics: Track raw reconstruction error per protocol type
        val_proto_recon = defaultdict(float)
        val_proto_counts = defaultdict(int)
        
        with torch.no_grad():
            for v_data, v_labels in val_loader:
                v_data = v_data.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    v_logits, v_mu, v_logvar = model(v_data)
                    v_loss = loss_function(v_logits, v_data, v_mu, v_logvar)
                    
                    # Compute reconstruction loss element-wise (no reduction) and sum across sequence length (dim=1)
                    recon_per_sample = F.binary_cross_entropy_with_logits(
                        v_logits, v_data, reduction='none'
                    ).sum(dim=1)
                    
                total_val_loss += v_loss.item()
                
                # Assign losses to their respective protocol labels
                for loss_val, proto in zip(recon_per_sample, v_labels):
                    val_proto_recon[proto] += loss_val.item()
                    val_proto_counts[proto] += 1
        
        avg_val_loss = total_val_loss / len(val_dataset)
        print(f"\n[*] Epoch {epoch}: Train Loss: {total_train_loss/len(train_idx):.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Display Reconstruction Loss Diagnostics Per Protocol (Sorted from worst to best reconstruction)
        print(f"\n[D] --- Protocol Reconstruction Diagnostics (Epoch {epoch}) ---")
        print(f"{'Protocol':<20} | {'Val Samples':<12} | {'Avg Reconstruction Loss':<25}")
        print("-" * 65)
        sorted_protos = sorted(
            val_proto_recon.keys(),
            key=lambda k: val_proto_recon[k] / val_proto_counts[k],
            reverse=True
        )
        for proto in sorted_protos:
            avg_recon = val_proto_recon[proto] / val_proto_counts[proto]
            print(f"{proto:<20} | {val_proto_counts[proto]:<12} | {avg_recon:.4f}")
        print("-" * 65 + "\n")
        
        # Step scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_MODEL_NAME)
            torch.save(model.encoder.state_dict(), OUTPUT_ENCODER_NAME)
            print(f"    [+] Best model updated at epoch {epoch}")
         
    # --- 5. Model Persistence ---
    # Final save (already handled by 'best model' logic above, but kept for consistency)
    print(f"\n[+] Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        start_time = time.time()
        train_model()
        print(f"[+] Total time: {time.time() - start_time:.2f} seconds.")
    else:
        print(f"[!] {DATASET_PATH} not found.")
