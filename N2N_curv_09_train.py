# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import time
import warnings
import chardet
warnings.filterwarnings('ignore')

# ==================== Global Configuration ====================
class Config:
    # Data related parameters
    data_root = "./Data/"  # Data root directory
    data_suffix = ".dat"   # Data file suffix
    file_encoding = None   # File encoding, if None auto-detect
    
    # Training related parameters
    num_pairs = 5400        # Number of noise-noise pairs
    batch_size = 1         # Batch size must be 1 for variable length
    num_epochs = 500        # Number of epochs
    learning_rate = 1e-3   # Initial learning rate
    # Dataset split ratios, train : validation : test = 70 : 15 : 15,
    # the same ratio used for the Noise2Clean benchmark
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Model saving
    model_save_path = "./models/denoise_1d.pth"  # Best model save path
    final_model_save_path = "./models/denoise_1d_final.pth"  # Final model save path
    
    # Learning rate scheduler parameters
    lr_scheduler_type = "cosine"  # LR scheduler type
    lr_min = 1e-5  # Minimum learning rate
    lr_patience = 10  # Patience for LR reduction
    lr_factor = 0.5  # LR reduction factor
    
    # Early Stop parameters
    early_stop_patience = 100  # Early stopping patience
    early_stop_min_delta = 1e-6  # Minimum improvement delta
    
    # Data preprocessing
    clip_negative = True     # Clip negative values to 0
    use_log1p = True         # Apply log1p transformation
    
    # System parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    num_workers = 0        # Data loader workers
    random_seed = 42       # Random seed
    
# Initialize configuration
config = Config()
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)
random.seed(config.random_seed)

# ==================== Early Stop Class ====================
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

# ==================== Data Preprocessing Functions ====================
def preprocess_data(data):
    """Preprocess data: clip negative values and apply log1p"""
    data = data.astype(np.float32)
    
    # 1. Clip negative values to 0
    if config.clip_negative:
        data = np.clip(data, 0, None)
    
    # 2. Apply log1p transformation if enabled
    if config.use_log1p:
        data = np.log1p(data + 1e-8)
    
    return data

# ==================== Sample-level splitting ====================
def split_folders_by_sample(data_root, train_ratio, val_ratio, random_seed):
    """Partition the sample folders into training, validation and test sets.

    Each folder holds the consecutive frames of one static sample, so splitting
    on folders rather than on pairs keeps every frame of a given sample inside a
    single subset. The validation and test pairs then share no sample with the
    training pairs, which is what makes the held-out loss an honest estimate of
    the performance on an unseen sample.
    """
    folders = sorted([f for f in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, f))])

    rng = random.Random(random_seed)
    rng.shuffle(folders)

    total = len(folders)
    n_train = int(train_ratio * total)
    n_val = int(val_ratio * total)

    return (folders[:n_train],
            folders[n_train:n_train + n_val],
            folders[n_train + n_val:])


# ==================== Dataset Class ====================
class FullCurveDataset(Dataset):
    """Dataset using full curves without cropping"""

    def __init__(self, data_root, num_pairs=500, folders=None):
        self.data_root = data_root
        self.num_pairs = num_pairs

        # Get all folders, or keep only the subset of samples passed in
        all_folders = sorted([f for f in os.listdir(data_root)
                             if os.path.isdir(os.path.join(data_root, f))])
        if folders is None:
            self.folders = all_folders
        else:
            selected = set(folders)
            self.folders = [f for f in all_folders if f in selected]
        
        # Collect all valid files
        self.folder_files = {}
        self.file_data = {}  # Cache file data to avoid repeated reading
        for folder in self.folders:
            folder_path = os.path.join(data_root, folder)
            files = sorted(glob.glob(os.path.join(folder_path, f"*{config.data_suffix}")))
            valid_files = []
            for file in files:
                try:
                    data = self._read_curve_file(file)
                    if len(data) > 10:  # Minimum length requirement
                        valid_files.append(file)
                        self.file_data[file] = data  # Cache the data
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
                    continue
            
            if len(valid_files) >= 2:
                self.folder_files[folder] = valid_files
        
        # Generate fixed number of pairs
        self.pairs = self._generate_pairs()
        print(f"Dataset initialized: {len(self.pairs)} pairs")

        # Calculate statistics
        if len(self.pairs) == 0:
            print("Warning: no pairs could be generated from the assigned samples")
            return

        lengths = [len(self.file_data[p[0]]) for p in self.pairs]
        print(f"Curve length range: {min(lengths)} to {max(lengths)} points")
        print(f"Average curve length: {np.mean(lengths):.0f} points")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        file1, file2 = self.pairs[idx]
        
        # Get data from cache
        data1 = self.file_data[file1]
        data2 = self.file_data[file2]
        
        # Apply preprocessing
        data1 = preprocess_data(data1)
        data2 = preprocess_data(data2)
        
        # Ensure both curves have the same length (use minimum length)
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # Convert to tensor and add channel dimension
        data1 = torch.FloatTensor(data1).unsqueeze(0)  # Shape: [1, L]
        data2 = torch.FloatTensor(data2).unsqueeze(0)  # Shape: [1, L]
        
        return data1, data2, min_len
    
    def _generate_pairs(self):
        """Generate fixed number of noise-noise pairs"""
        pairs = []
        
        # Get all folders with at least 2 files
        valid_folders = [folder for folder in self.folders 
                        if folder in self.folder_files and len(self.folder_files[folder]) >= 2]
        
        if not valid_folders:
            print("Error: No valid folders found")
            return pairs
        
        # Generate pairs
        attempts = 0
        max_attempts = self.num_pairs * 10
        
        while len(pairs) < self.num_pairs and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a folder
            folder = random.choice(valid_folders)
            files = self.folder_files[folder]
            
            # Randomly select two different files
            if len(files) >= 2:
                file1, file2 = random.sample(files, 2)
                pairs.append((file1, file2))
        
        # If we couldn't generate enough unique pairs, duplicate some
        if len(pairs) < self.num_pairs:
            print(f"Warning: Could only generate {len(pairs)} unique pairs, duplicating some")
            needed = self.num_pairs - len(pairs)
            for i in range(needed):
                pairs.append(pairs[i % len(pairs)])
        
        return pairs[:self.num_pairs]
    
    def _detect_encoding(self, filepath):
        """Detect file encoding"""
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            
            if encoding is None or result['confidence'] < 0.7:
                encodings_to_try = ['utf-8', 'gbk', 'latin-1']
                for enc in encodings_to_try:
                    try:
                        with open(filepath, 'r', encoding=enc) as test_f:
                            test_f.read(1024)
                        return enc
                    except:
                        continue
                return 'utf-8'
            return encoding
    
    def _read_curve_file(self, filepath):
        """Read single curve file, extract second column data"""
        if config.file_encoding:
            encodings_to_try = [config.file_encoding]
        else:
            detected_encoding = self._detect_encoding(filepath)
            encodings_to_try = [detected_encoding, 'utf-8', 'gbk']
        
        for encoding in encodings_to_try:
            try:
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    lines = f.readlines()
                
                # Find data start
                data_start = 0
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line or line.startswith(('#', '//', '%')):
                        continue
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            float(parts[0])
                            float(parts[1])
                            data_start = i
                            break
                    except:
                        continue
                
                # Extract second column
                data = []
                for line in lines[data_start:]:
                    line = line.strip()
                    if not line or line.startswith(('#', '//', '%')):
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            val = float(parts[1])
                            data.append(val)
                    except:
                        continue
                
                return np.array(data, dtype=np.float32)
                
            except UnicodeDecodeError:
                continue
        
        print(f"Error: Cannot read file {filepath}")
        return np.zeros(100, dtype=np.float32)

# ==================== Dynamic Padding Collate Function ====================
def variable_length_collate(batch):
    """Custom collate function for variable length sequences"""
    # Unzip the batch
    data1_list, data2_list, lengths = zip(*batch)
    
    # Find max length in this batch
    max_len = max(lengths)
    
    # Pad sequences to max length
    padded_data1 = []
    padded_data2 = []
    
    for data1, data2, length in zip(data1_list, data2_list, lengths):
        # Pad data1
        if data1.shape[1] < max_len:
            padding = torch.zeros(1, max_len - data1.shape[1])
            data1 = torch.cat([data1, padding], dim=1)
        padded_data1.append(data1)
        
        # Pad data2
        if data2.shape[1] < max_len:
            padding = torch.zeros(1, max_len - data2.shape[1])
            data2 = torch.cat([data2, padding], dim=1)
        padded_data2.append(data2)
    
    # Stack padded sequences
    padded_data1 = torch.stack(padded_data1, dim=0)
    padded_data2 = torch.stack(padded_data2, dim=0)
    
    return padded_data1, padded_data2

# ==================== Adaptive Network Model ====================
class AdaptiveUNet1D_Denoising(nn.Module):
    """Adaptive 1D U-Net that adjusts to input curve dimensions"""
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, in_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # Store original size for later upsampling
        original_size = x.shape[2]
        
        # Encoder
        e1 = self.enc1(x)      # [B, 32, L]
        e2 = self.enc2(e1)     # [B, 64, L/2]
        e3 = self.enc3(e2)     # [B, 128, L/4]
        
        # Bottleneck
        b = self.bottleneck(e3)  # [B, 256, L/8]
        
        # Decoder with skip connections
        d3 = self.dec3(b)  # [B, 128, L/4]
        
        # Upsample d3 and concatenate with e3
        d3_up = F.interpolate(d3, size=e3.shape[2], mode='linear', align_corners=False)
        d2_input = torch.cat([d3_up, e3], dim=1)  # [B, 256, L/4]
        d2 = self.dec2(d2_input)  # [B, 64, L/2]
        
        # Upsample d2 and concatenate with e2
        d2_up = F.interpolate(d2, size=e2.shape[2], mode='linear', align_corners=False)
        d1_input = torch.cat([d2_up, e2], dim=1)  # [B, 128, L/2]
        d1 = self.dec1(d1_input)  # [B, 1, L]
        
        # Ensure output has same length as input
        d1_up = F.interpolate(d1, size=original_size, mode='linear', align_corners=False)
        
        # Residual connection
        return x + d1_up

# ==================== Training Functions ====================
def train_noise2noise():
    """Train Noise2Noise denoising model"""
    print("="*60)
    print("Training Noise2Noise Denoising Model")
    print("="*60)
    
    # Check data directory
    if not os.path.exists(config.data_root):
        print(f"Error: Data directory {config.data_root} does not exist!")
        return None, None, None, None, None
    
    # Split the samples (folders) into training, validation and test sets, so that
    # no sample contributes frames to more than one subset
    try:
        train_folders, val_folders, test_folders = split_folders_by_sample(
            config.data_root, config.train_ratio, config.val_ratio, config.random_seed
        )
    except Exception as e:
        print(f"Error listing samples: {e}")
        return None, None, None, None, None

    n_samples = len(train_folders) + len(val_folders) + len(test_folders)
    if not (train_folders and val_folders and test_folders):
        print(f"Error: {n_samples} samples are not enough to fill all three splits!")
        return None, None, None, None, None

    print(f"Samples (folders): {len(train_folders)} training, "
          f"{len(val_folders)} validation, {len(test_folders)} test")

    # One dataset per subset, each drawing its pairs only from its own samples
    try:
        train_dataset = FullCurveDataset(config.data_root,
                                         num_pairs=int(config.num_pairs * config.train_ratio),
                                         folders=train_folders)
        val_dataset = FullCurveDataset(config.data_root,
                                       num_pairs=int(config.num_pairs * config.val_ratio),
                                       folders=val_folders)
        test_dataset = FullCurveDataset(config.data_root,
                                        num_pairs=int(config.num_pairs * config.test_ratio),
                                        folders=test_folders)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None, None, None, None

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: one of the three splits contains no pairs!")
        return None, None, None, None, None

    print(f"Training set: {len(train_dataset)} pairs")
    print(f"Validation set: {len(val_dataset)} pairs")
    print(f"Test set: {len(test_dataset)} pairs")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=variable_length_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=variable_length_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=variable_length_collate
    )
    
    # Create model
    model = AdaptiveUNet1D_Denoising(in_channels=1)
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture Summary:")
    print("-"*40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Parameter breakdown: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    
    # L1 Loss only (as requested)
    criterion = nn.L1Loss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    if config.lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.num_epochs, 
            eta_min=config.lr_min
        )
    elif config.lr_scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.lr_factor, 
            patience=config.lr_patience,
            min_lr=config.lr_min
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta
    )
    
    # Training records
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    # Create model save directory
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    
    print("\nStarting training...")
    print("-"*60)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (noisy1, noisy2) in enumerate(train_loader):
            noisy1 = noisy1.to(config.device)
            noisy2 = noisy2.to(config.device)
            
            optimizer.zero_grad()
            
            # Noise2Noise: use noisy1 to predict noisy2
            denoised = model(noisy1)
            loss = criterion(denoised, noisy2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for noisy1, noisy2 in val_loader:
                noisy1 = noisy1.to(config.device)
                noisy2 = noisy2.to(config.device)
                
                denoised = model(noisy1)
                loss = criterion(denoised, noisy2)
                
                epoch_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else 0
        
        # Update learning rate scheduler
        if config.lr_scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        # Record loss
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Clear line and print progress in place
        print(f'\rEpoch {epoch+1:3d}/{config.num_epochs} | '
              f'Time: {epoch_time:5.1f}s | '
              f'Train Loss: {avg_train_loss:8.6f} | '
              f'Val Loss: {avg_val_loss:8.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}', end='')
        
        # Save best model
        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config.__dict__
            }, config.model_save_path)
            
            print(f"\n✅ Saved best model, validation loss: {avg_val_loss:.6f}")
        
        # Early Stopping check
        if early_stopping(avg_val_loss):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break
    
    print()  # New line after training completes
    print("-"*60)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Restore the weights selected on the validation set, so that what is returned
    # and evaluated below is the selected model and not the last epoch
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final model
    if best_model_state is not None:
        torch.save({
            'epoch': config.num_epochs,
            'model_state_dict': best_model_state,
            'train_loss': train_losses[-1] if train_losses else 0,
            'val_loss': best_val_loss,
            'config': config.__dict__
        }, config.final_model_save_path)
        print(f"Final model saved to: {config.final_model_save_path}")

    # Evaluate the selected model on the held-out test set
    model.eval()
    epoch_test_loss = 0
    test_batches = 0

    with torch.no_grad():
        for noisy1, noisy2 in test_loader:
            noisy1 = noisy1.to(config.device)
            noisy2 = noisy2.to(config.device)

            denoised = model(noisy1)
            loss = criterion(denoised, noisy2)

            epoch_test_loss += loss.item()
            test_batches += 1

    test_loss = epoch_test_loss / test_batches if test_batches > 0 else 0
    print(f"Test loss on {len(test_dataset)} held-out pairs: {test_loss:.6f}")

    return model, train_losses, val_losses, best_val_loss, test_loss

# ==================== Main Function ====================
def main():
    """Main function: train model"""
    
    print("="*60)
    print("1D Curve Noise2Noise Denoising - Training Module")
    print("="*60)
    
    # Print configuration information
    print("Training Configuration:")
    print("-"*30)
    print(f"Number of noise-noise pairs: {config.num_pairs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size} (for variable length)")
    print(f"Epochs: {config.num_epochs}")
    print(f"Split (train:val:test): {config.train_ratio:.2f} : {config.val_ratio:.2f} : {config.test_ratio:.2f}")
    print(f"Clip negative values: {config.clip_negative}")
    print(f"Use log1p transform: {config.use_log1p}")
    print(f"Loss function: L1 Loss only")
    print("-"*30)
    
    # Check data directory
    if not os.path.exists(config.data_root):
        print(f"Error: Data directory {config.data_root} does not exist!")
        return
    
    # Train model
    result = train_noise2noise()
    
    if result[0] is None:
        print("Training failed, please check data and configuration")
        return
    
    print("\n" + "="*60)
    print("Training Completed!")
    print(f"Best model saved to: {config.model_save_path}")
    print(f"Final model saved to: {config.final_model_save_path}")
    print(f"Best validation loss: {result[3]:.6f}")
    print(f"Test loss: {result[4]:.6f}")
    print("="*60)

# ==================== Run Main Program ====================
if __name__ == "__main__":
    main()