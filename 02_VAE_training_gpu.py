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

try:
    from sklearn.metrics import silhouette_score  # used for collapse-aware model selection
    _SILHOUETTE_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _SILHOUETTE_AVAILABLE = False

# --- Hyperparameters ---
MAX_LEN = 256     # according to payload length statistics
# NOTE: LATENT_DIM and the encoder/decoder topology below are kept byte-compatible
# with src/protocol_re/neural/model_loader.py:ConvVAE so the trained .pth loads in the
# pipeline unchanged. The clustering quality gains here come from a much stronger
# *training objective* (anti-collapse KL + discriminative losses), not from resizing
# the latent. Do NOT change LATENT_DIM/MAX_LEN without also updating the loader and the
# latent_dim=32 call sites in clustering/{hybrid,enhanced}_features.py + clearing caches.
LATENT_DIM = 32
BATCH_SIZE = 256  # Increased for GPU efficiency (RTX 3050 can handle this with AMP)
EPOCHS = 40
LEARNING_RATE = 1e-3
SAMPLER_POWER = 0.5  # 0.0 = natural distribution, 1.0 = fully balanced, 0.5 = square-root balanced (optimal)

# --- Loss weighting (the heart of the anti-collapse + discriminative redesign) ---
# Posterior collapse (every payload mapping to ~the same latent) was the root cause of
# the clustering breakdown. The old objective used BETA=4.0 with a pure reconstruction
# term, which both over-penalised the KL and gave the latent zero incentive to separate
# protocols. We now:
#   * anneal a SMALL beta from 0 -> BETA_MAX (warm-up) so the decoder learns to use z
#     before any KL pressure is applied, and
#   * apply a per-dimension FREE_BITS floor so each latent dim is forced to carry a
#     minimum amount of information (cannot be switched off), and
#   * add supervised-contrastive + classification losses on mu using the `protocol`
#     labels already present in the dataset, directly shaping the latent for clustering.
BETA_MAX = 1.0          # was 4.0 — high beta drove posterior collapse
KL_WARMUP_EPOCHS = 10   # linearly ramp beta 0 -> BETA_MAX over the first N epochs
FREE_BITS = 0.5         # nats/dim KL floor; prevents individual latent dims collapsing
W_SUPCON = 2.0          # weight of the supervised contrastive term on mu
W_CLS = 1.0             # weight of the auxiliary protocol-classification term on mu
SUPCON_TEMP = 0.1       # temperature for supervised contrastive loss
PROJ_DIM = 64           # projection head width for contrastive learning (train-only)
LABEL_SMOOTHING = 0.05  # regularises the auxiliary classifier
GRAD_CLIP = 5.0         # gradient-norm clip for training stability

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
# IMPORTANT: layer names/shapes MUST stay in sync with model_loader.py:ConvVAE.
# The discriminative power is added via *separate* train-only heads (see DiscriminativeHeads)
# whose parameters are deliberately NOT saved, so the persisted state_dict stays identical.
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


# --- 2b. Train-only discriminative heads (NOT persisted) ---
# These operate on mu to make the latent space cluster-friendly. They are excluded from
# the saved checkpoint so the pipeline keeps loading the plain ConvVAE state_dict.
class DiscriminativeHeads(nn.Module):
    def __init__(self, latent_dim, num_classes, proj_dim=PROJ_DIM):
        super().__init__()
        # Non-linear projection head for supervised contrastive learning (SimCLR/SupCon style)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim),
        )
        # Auxiliary linear classifier providing a discriminative gradient into mu
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, mu):
        z = F.normalize(self.projection(mu), dim=1)  # unit-norm for cosine contrastive
        logits = self.classifier(mu)
        return z, logits


# --- 3. Loss Functions ---
def kld_free_bits(mu, logvar, free_bits=FREE_BITS):
    """KL(q(z|x) || N(0,1)) with a per-dimension 'free bits' floor.

    Standard VAE KL lets the optimiser drive whole latent dimensions to the prior
    (posterior collapse), which is exactly what destroyed the clustering. Free-bits
    clamps the *batch-averaged* KL of each dimension to a minimum, so every dimension is
    forced to retain information. Returns a per-sample-scale scalar (sum over dims).
    """
    # (B, D)
    kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Average over the batch, then floor each dimension, then sum across dimensions.
    kld_dim = kld_per_dim.mean(dim=0)
    kld = torch.clamp(kld_dim, min=free_bits).sum()
    return kld


def supervised_contrastive_loss(features, labels, temperature=SUPCON_TEMP):
    """SupCon loss (Khosla et al. 2020) on L2-normalised projections.

    Pulls samples sharing a protocol label together and pushes different protocols apart
    directly in the embedding space — the discriminative signal the old reconstruction-only
    objective completely lacked. Samples with no in-batch positive are skipped.
    """
    device = features.device
    batch_size = features.shape[0]
    labels = labels.view(-1, 1)

    # positives: same label (excluding the diagonal / self-comparison)
    positive_mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(batch_size, device=device)

    similarity = torch.matmul(features, features.T) / temperature
    # numerical stability
    sim_max, _ = similarity.max(dim=1, keepdim=True)
    similarity = similarity - sim_max.detach()

    exp_sim = torch.exp(similarity) * (1.0 - self_mask)  # drop self term from denominator
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    positive_mask = positive_mask * (1.0 - self_mask)
    positives_per_row = positive_mask.sum(dim=1)
    valid = positives_per_row > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positives_per_row[valid]
    return -mean_log_prob_pos.mean()


def vae_loss_terms(logits, x, mu, logvar):
    """Reconstruction (BCE) + free-bits KL, both at per-sample scale (mean over batch)."""
    recon_loss = F.binary_cross_entropy_with_logits(
        logits, x, reduction='none'
    ).sum(dim=1).mean()
    kld_loss = kld_free_bits(mu, logvar)
    return recon_loss, kld_loss

# --- 4. Training Loop (Optimized for GPU) ---
def train_model():
    # Pass MAX_LEN to dataset
    full_dataset = IndustrialDataset(DATASET_PATH, max_len=MAX_LEN)

    # Stratified Train-Validation Split (90/10) to preserve minority protocol representation
    indices = np.arange(len(full_dataset))
    labels = np.array(full_dataset.labels)

    # Build a stable label -> index map for the discriminative (contrastive + classifier) losses
    class_names = sorted(set(full_dataset.labels))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"[+] {num_classes} protocol classes for discriminative training: {class_names}")

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

    # Calculate balanced sample weights for training set to handle extreme class imbalance.
    # Balanced batches are doubly important now: supervised contrastive learning needs
    # multiple classes (and multiple same-class samples) co-present in each batch.
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
    # Train-only discriminative heads (excluded from the saved checkpoint)
    heads = DiscriminativeHeads(LATENT_DIM, num_classes).to(DEVICE)

    # Optimise the VAE and the auxiliary heads jointly
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()),
        lr=LEARNING_RATE
    )
    # Schedule on the clustering objective (silhouette: higher is better -> mode='max')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    class_ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Initialize Gradient Scaler for Mixed Precision
    scaler = GradScaler()

    # Model selection is now driven by latent separability, not by reconstruction loss.
    # (Reconstruction loss is minimised by collapse, so it is the wrong thing to track.)
    best_val_silhouette = -float('inf')
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        # Linear KL warm-up: let the decoder learn to use z before applying KL pressure.
        beta = BETA_MAX * min(1.0, epoch / max(1, KL_WARMUP_EPOCHS))

        # Training Phase
        model.train()
        heads.train()
        total_train_loss = 0.0
        running = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train] beta={beta:.2f}")

        for data, lbls in pbar:
            data = data.to(DEVICE, non_blocking=True) # non_blocking works with pin_memory
            targets = torch.tensor(
                [label_to_idx[l] for l in lbls], dtype=torch.long, device=DEVICE
            )
            optimizer.zero_grad(set_to_none=True)     # Slightly faster than zero_grad()

            # Autocast handles the mixed-precision forward pass; the custom contrastive /
            # KL math is done in float32 below for numerical stability.
            with autocast(device_type=DEVICE.type):
                logits, mu, logvar = model(data)

            logits = logits.float()
            mu = mu.float()
            logvar = logvar.float()

            recon_loss, kld_loss = vae_loss_terms(logits, data, mu, logvar)

            proj, class_logits = heads(mu)
            supcon = supervised_contrastive_loss(proj, targets)
            cls = class_ce(class_logits, targets)

            loss = recon_loss + beta * kld_loss + W_SUPCON * supcon + W_CLS * cls

            # Scaled backward pass + gradient clipping for stability
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(heads.parameters()), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            running['recon'] += recon_loss.item()
            running['kld'] += kld_loss.item()
            running['supcon'] += supcon.item()
            running['cls'] += cls.item()
            pbar.set_postfix({
                'L': f"{loss.item():.2f}",
                'rec': f"{recon_loss.item():.1f}",
                'kld': f"{kld_loss.item():.2f}",
                'sc': f"{supcon.item():.2f}",
                'ce': f"{cls.item():.2f}",
            })

        # Validation Phase
        model.eval()
        heads.eval()
        total_val_loss = 0.0

        # Diagnostics: Track raw reconstruction error per protocol type
        val_proto_recon = defaultdict(float)
        val_proto_counts = defaultdict(int)

        # Collect mu for collapse-aware diagnostics (active units + silhouette)
        val_mu_chunks = []
        val_label_chunks = []

        with torch.no_grad():
            for v_data, v_labels in val_loader:
                v_data = v_data.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    v_logits, v_mu, v_logvar = model(v_data)

                v_logits = v_logits.float()
                v_mu = v_mu.float()
                v_logvar = v_logvar.float()

                v_recon, v_kld = vae_loss_terms(v_logits, v_data, v_mu, v_logvar)
                # Track the same composite objective (reported for reference)
                total_val_loss += (v_recon + beta * v_kld).item() * v_data.size(0)

                # Per-sample reconstruction for the protocol diagnostics
                recon_per_sample = F.binary_cross_entropy_with_logits(
                    v_logits, v_data, reduction='none'
                ).sum(dim=1)

                for loss_val, proto in zip(recon_per_sample, v_labels):
                    val_proto_recon[proto] += loss_val.item()
                    val_proto_counts[proto] += 1

                val_mu_chunks.append(v_mu.cpu().numpy())
                val_label_chunks.extend(v_labels)

        avg_val_loss = total_val_loss / len(val_dataset)

        # --- Collapse-aware latent diagnostics ---
        val_mu = np.concatenate(val_mu_chunks, axis=0)
        per_dim_std = val_mu.std(axis=0)
        active_units = int((per_dim_std > 0.01).sum())  # dims that actually carry signal

        val_silhouette = -1.0
        if _SILHOUETTE_AVAILABLE and num_classes > 1:
            # Silhouette of mu against the TRUE protocol labels = how well-separated the
            # latent already is by protocol. This is the metric we optimise model selection
            # for, because it tracks clustering quality directly (recon loss does not).
            y = np.array([label_to_idx[l] for l in val_label_chunks])
            try:
                if val_mu.shape[0] > 2000:
                    rng = np.random.default_rng(42)
                    sel = rng.choice(val_mu.shape[0], 2000, replace=False)
                    val_silhouette = float(silhouette_score(val_mu[sel], y[sel]))
                else:
                    val_silhouette = float(silhouette_score(val_mu, y))
            except Exception:
                val_silhouette = -1.0

        avg_run = {k: v / max(1, len(train_loader)) for k, v in running.items()}
        print(
            f"\n[*] Epoch {epoch}: ValLoss {avg_val_loss:.2f} | "
            f"Silhouette {val_silhouette:.4f} | ActiveDims {active_units}/{LATENT_DIM} | "
            f"train(rec {avg_run['recon']:.1f}, kld {avg_run['kld']:.2f}, "
            f"supcon {avg_run['supcon']:.2f}, ce {avg_run['cls']:.2f})"
        )

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

        # Step scheduler on the clustering objective (silhouette). Fall back to the
        # negative validation loss when silhouette is unavailable.
        scheduler.step(val_silhouette if val_silhouette > -1.0 else -avg_val_loss)

        # Save best model by latent separability (only the base ConvVAE is persisted, so
        # the pipeline's model_loader.ConvVAE loads it with no architecture changes).
        if val_silhouette > best_val_silhouette:
            best_val_silhouette = val_silhouette
            best_epoch = epoch
            torch.save(model.state_dict(), OUTPUT_MODEL_NAME)
            torch.save(model.encoder.state_dict(), OUTPUT_ENCODER_NAME)
            print(f"    [+] Best model updated at epoch {epoch} (silhouette={val_silhouette:.4f})")

    # --- 5. Model Persistence ---
    # Final save (already handled by 'best model' logic above, but kept for consistency)
    print(f"\n[+] Training Complete. Best Silhouette: {best_val_silhouette:.4f} (epoch {best_epoch}).")

if __name__ == "__main__":
    if os.path.exists(DATASET_PATH):
        start_time = time.time()
        train_model()
        print(f"[+] Total time: {time.time() - start_time:.2f} seconds.")
    else:
        print(f"[!] {DATASET_PATH} not found.")
