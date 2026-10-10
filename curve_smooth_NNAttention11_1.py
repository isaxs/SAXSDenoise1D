import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import random

# Set device (automatically detect GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def extract_sample_id(file_path):
    """Recover which sample a paired curve belongs to.

    The pairing script records the source path in the header ('# Data from ...'),
    and that path ends with the folder of the sample the curve was cut from, so
    the sample can be read back from there. Files written without that header fall
    back to the part of the file name before '_paired_with_'.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('# Data from'):
                    source = line.split('from', 1)[1].strip()
                    folder = os.path.basename(os.path.dirname(source))
                    return folder if folder else source
                if not line.startswith('#'):
                    break
    except OSError:
        pass

    name = os.path.basename(file_path)
    return name.split('_paired_with_')[0]

# 1. Custom Dataset Class - Modified to read SmoothFactor from headers and support log transformation
class CurveDataset(Dataset):
    def __init__(self, folder_path, use_log_coordinates=False):
        """
        Initialize curve dataset
        
        Parameters:
            folder_path: Path to folder containing dat files
            use_log_coordinates: Whether to transform data to log coordinates
        """
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
        self.use_log_coordinates = use_log_coordinates
        
        # Store data for each curve
        self.curves = []
        self.curve_lengths = []
        self.smooth_factors = []  # New: Store smooth factors from headers
        self.samples = []         # New: Sample each curve belongs to, used for splitting
        
        print("Loading curve data...")
        for file_name in self.file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read smooth factor from header first
                smooth_factor = 16.0  # Default value if not found
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('# # SmoothFactor:'):
                            # Extract the numeric value
                            match = re.search(r'# # SmoothFactor:\s*([0-9.]+)', line)
                            if match:
                                smooth_factor = float(match.group(1))
                            break
                
                # Read data, skip comment lines - use raw string to handle escape characters
                df = pd.read_csv(file_path, comment='#', sep=r'\s+', header=None)
                
                # Ensure file has at least 4 columns
                if df.shape[1] < 4:
                    continue
                
                # Extract second and fourth columns as a complete curve
                X_curve = df.iloc[:, 1].values.astype(np.float32)  # I_data (rough curve)
                y_curve = df.iloc[:, 3].values.astype(np.float32)  # I_smo (smooth curve)
                
                # Remove negative values by filtering them out
                # Create a mask for positive values in both X and y
                valid_mask = (X_curve > 0) & (y_curve > 0)
                
                # Apply the mask to both curves
                X_curve = X_curve[valid_mask]
                y_curve = y_curve[valid_mask]
                
                # Skip if no valid data points remain
                if len(X_curve) == 0:
                    continue
                
                # Apply log transformation if requested
                if self.use_log_coordinates:
                    X_curve = np.log1p(X_curve)  # log(1+x) transformation
                    y_curve = np.log1p(y_curve)  # log(1+x) transformation
                
                # Store curve data and smooth factor
                self.curves.append((X_curve, y_curve))
                self.curve_lengths.append(len(X_curve))
                self.smooth_factors.append(smooth_factor)
                self.samples.append(extract_sample_id(file_path))
                    
            except Exception as e:
                # Silently handle errors, just skip the file
                continue
        
        # Check if any data was found
        if len(self.curves) == 0:
            raise ValueError("No valid data found")
        
        # Find maximum length
        self.max_length = max(self.curve_lengths)
        print(f"\nMaximum curve length: {self.max_length} data points")
        
        # Pad all curves to maximum length with symmetric padding
        self.padded_curves = []
        for i, (X_curve, y_curve) in enumerate(self.curves):
            length = len(X_curve)
            if length < self.max_length:
                # Use symmetric padding instead of zero padding
                pad_size = self.max_length - length
                # Pad X with symmetric padding
                X_padded = np.pad(X_curve, (0, pad_size), mode='symmetric')
                # Pad y with symmetric padding
                y_padded = np.pad(y_curve, (0, pad_size), mode='symmetric')
                
                self.padded_curves.append((X_padded, y_padded))
            else:
                self.padded_curves.append((X_curve, y_curve))
        
        print(f"\nSuccessfully loaded and processed {len(self.padded_curves)} curves")
        n_samples = len(set(self.samples))
        print(f"Samples the curves come from: {n_samples}")
        if n_samples == len(self.padded_curves):
            print("Warning: every curve was treated as its own sample. Check that the paired files "
                  "carry a '# Data from ...' header identifying their source folder.")
        print(f"Log coordinates: {self.use_log_coordinates}")
    
    def __len__(self):
        return len(self.padded_curves)
    
    def __getitem__(self, idx):
        X_padded, y_padded = self.padded_curves[idx]
        smooth_factor = self.smooth_factors[idx]  # Get the smooth factor for this curve
        # Return CPU tensors, move to GPU in training loop
        return (torch.tensor(X_padded, dtype=torch.float32), 
                torch.tensor(y_padded, dtype=torch.float32),
                torch.tensor(smooth_factor, dtype=torch.float32))
    
    def get_max_length(self):
        return self.max_length
    
    def get_use_log_coordinates(self):
        return self.use_log_coordinates


# Gated Convolution Unit - Specialized for signal processing
class GatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(GatedConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels * 2,  # Double output channels for gating
            kernel_size, 
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        # Split into two parts: one for feature extraction, one for gating
        gate, feature = torch.chunk(x, 2, dim=1)
        return self.sigmoid(gate) * feature

# Residual Gated Block - Specialized for signal smoothing
class ResidualGatedBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super(ResidualGatedBlock, self).__init__()
        self.gated_conv1 = GatedConv1D(channels, channels, kernel_size, dilation)
        self.gated_conv2 = GatedConv1D(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.gated_conv1(x)
        out = self.relu(out)
        out = self.gated_conv2(out)
        out += residual
        return self.relu(out)

# Multi-Scale Receptive Field Module - For capturing features at different frequencies
class MultiScaleReceptiveField(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleReceptiveField, self).__init__()
        
        # Different scale convolution kernels
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=9, padding=4)
        
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # Parallel multi-scale convolution
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out4 = self.relu(self.conv4(x))
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        return out

# Flat Region Detection Module - NEW: Specifically for detecting flat regions
class FlatRegionDetector(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(FlatRegionDetector, self).__init__()
        
        # Multi-scale gradient analysis for flat region detection
        self.gradient_analysis = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU()
        )
        
        # Flat region classification
        self.flat_classifier = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Local variance computation for flatness measurement
        self.variance_conv = nn.Conv1d(1, out_channels, kernel_size=11, padding=5, groups=1)
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len)
        
        # Compute local variance as flatness indicator
        local_mean = F.avg_pool1d(x, kernel_size=11, stride=1, padding=5)
        local_var = F.avg_pool1d((x - local_mean) ** 2, kernel_size=11, stride=1, padding=5)
        
        # Normalize variance
        var_norm = torch.sigmoid(-local_var * 100)  # Low variance -> high flatness
        
        # Gradient-based flat region detection
        grad_features = self.gradient_analysis(x)
        flat_mask = self.flat_classifier(grad_features)
        
        # Combine variance and gradient information
        combined_flatness = (flat_mask + var_norm) / 2
        
        return combined_flatness

# Enhanced Self-Attention Module with flat region awareness
class SelfAttention1D(nn.Module):
    def __init__(self, in_channels, edge_enhance=True, num_heads=8, flat_region_enhance=True):
        super(SelfAttention1D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.flat_region_enhance = flat_region_enhance
        
        # Multi-head attention projections
        self.query = nn.Conv1d(in_channels, in_channels, 1)
        self.key = nn.Conv1d(in_channels, in_channels, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv1d(in_channels, in_channels, 1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Edge enhancement parameters
        self.edge_enhance = edge_enhance
        if edge_enhance:
            self.edge_weight = nn.Parameter(torch.ones(1, in_channels, 1))
        
        # Frequency analysis components for noise detection
        self.freq_analysis = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Peak detection components
        self.peak_detection = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # Flat region detection - NEW
        if flat_region_enhance:
            self.flat_detection = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 4, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(in_channels // 4, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        batch_size, C, seq_len = x.size()
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        
        # Compute attention scores
        attention_scores = torch.einsum('bhdn,bhdm->bhnm', Q, K) / (self.head_dim ** 0.5)
        
        # Frequency analysis to identify noise regions
        freq_weights = self.freq_analysis(x)
        freq_attention = torch.mean(freq_weights, dim=1, keepdim=True)  # (batch_size, 1, seq_len)
        
        # Peak detection to identify scattering peaks
        peak_mask = self.peak_detection(x)  # (batch_size, 1, seq_len)
        
        # Flat region detection - NEW
        if self.flat_region_enhance:
            flat_mask = self.flat_detection(x)  # (batch_size, 1, seq_len)
            # Enhance attention in flat regions for better smoothing
            flat_enhancement = flat_mask * 1.5  # Boost attention in flat regions
        else:
            flat_mask = torch.ones_like(peak_mask)
            flat_enhancement = torch.ones_like(peak_mask)
        
        # Combine frequency, peak, and flat information
        content_aware_mask = (1 - peak_mask) * freq_attention * flat_enhancement
        
        # Apply content-aware masking to attention scores
        content_aware_mask = content_aware_mask.view(batch_size, 1, 1, seq_len)
        attention_scores = attention_scores * content_aware_mask
        
        # Edge enhancement
        if self.edge_enhance and seq_len > 10:
            edge_mask = torch.ones_like(attention_scores)
            edge_size = max(5, seq_len // 20)  # Edge region size
            
            # Create edge weight mask - enhance attention at edges
            edge_mask[:, :, :, :edge_size] *= 2.0  # Left edge
            edge_mask[:, :, :, -edge_size:] *= 2.0  # Right edge
            
            attention_scores = attention_scores * edge_mask
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attention_weights, V)
        out = out.contiguous().view(batch_size, C, seq_len)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        # Further edge enhancement
        if self.edge_enhance:
            out = out * self.edge_weight
            
        return out, attention_weights, peak_mask.squeeze(1), flat_mask.squeeze(1)

# Enhanced Adaptive Smoothing Network with improved flat region handling
class AdaptiveSmoothingNetV2(nn.Module):
    def __init__(self, base_channels=32):
        super(AdaptiveSmoothingNetV2, self).__init__()
        
        # Enhanced smooth factor processing with more layers
        self.smooth_factor_fc = nn.Sequential(
            nn.Linear(1, base_channels * 2),
            nn.ReLU(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.ReLU(),
            nn.Linear(base_channels * 2, base_channels)
        )
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(base_channels)
        )
        
        # Multi-scale receptive field module
        self.multiscale = MultiScaleReceptiveField(base_channels, base_channels * 2)
        
        # Self-attention module for global context with edge and flat region enhancement
        self.self_attention = SelfAttention1D(base_channels * 2, edge_enhance=True, flat_region_enhance=True)
        
        # NEW: Flat region detector
        self.flat_detector = FlatRegionDetector(1, base_channels // 2)
        
        # Sequence of gated residual blocks with skip connections
        self.residual_blocks = nn.Sequential(
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=1),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=2),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=4),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=8)
        )
        
        # Enhanced edge protection module
        self.edge_protection = nn.Sequential(
            nn.Conv1d(1, base_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels//2, base_channels//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels//4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # NEW: Flat region enhancement module
        self.flat_enhancement = nn.Sequential(
            nn.Conv1d(1, base_channels//2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels//2, base_channels//4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels//4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # Enhanced adaptive smoothing module with flat region awareness
        self.adaptive_smoothing = nn.Sequential(
            nn.Conv1d(base_channels * 2 + base_channels + 1, base_channels * 2, kernel_size=3, padding=1),  # +1 for flat features
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output smoothing weights
        )
        
        # Enhanced output layer with more capacity
        self.output_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        )
        
    def forward(self, x, smooth_factor):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Normalize smooth factor to [0, 1] range (assuming range 1-256)
        smooth_factor_normalized = smooth_factor #/ 256.0
        
        # Process smooth factor with enhanced network
        sf_encoded = self.smooth_factor_fc(smooth_factor_normalized.unsqueeze(-1))  # (batch_size, base_channels)
        sf_encoded = sf_encoded.unsqueeze(-1)  # (batch_size, base_channels, 1)
        
        # Add channel dimension: (batch_size, 1, seq_len)
        x_input = x.unsqueeze(1)
        
        # NEW: Detect flat regions
        flat_region_mask = self.flat_detector(x_input)  # (batch_size, 1, seq_len)
        
        # Initial feature extraction
        features = self.initial_conv(x_input)
        
        # Multi-scale receptive field
        features = self.multiscale(features)
        
        # Self-attention for global context with flat region awareness
        features, attention_weights, peak_mask, flat_mask = self.self_attention(features)
        
        # Gated residual processing
        features = self.residual_blocks(features)
        
        # Expand smooth factor encoding to match sequence length
        sf_expanded = sf_encoded.repeat(1, 1, seq_len)  # (batch_size, base_channels, seq_len)
        
        # NEW: Expand flat region mask to match feature dimensions
        flat_features = flat_region_mask  # Already (batch_size, 1, seq_len)
        
        # Concatenate features with smooth factor encoding and flat features
        features_with_sf = torch.cat([features, sf_expanded, flat_features], dim=1)
        
        # Adaptive smoothing weights with smooth factor and flat region information
        smooth_weights = self.adaptive_smoothing(features_with_sf)
        
        # Use peak mask to protect scattering peaks from excessive smoothing
        peak_protection = 1 - peak_mask.unsqueeze(1)  # Invert mask: 1 for non-peaks, 0 for peaks
        
        # NEW: Use flat mask to enhance smoothing in flat regions
        flat_enhancement = flat_region_mask * 0.4  # Add extra smoothing in flat regions
        
        # Apply peak protection and flat enhancement to smoothing weights
        smooth_weights = smooth_weights * peak_protection + 0.05 * (1 - peak_protection)
        smooth_weights = smooth_weights + flat_enhancement
        smooth_weights = torch.clamp(smooth_weights, 0, 1)  # Ensure weights stay in [0,1]
        
        # Calculate edge protection mask
        edge_mask = self.edge_protection(x_input)
        
        # NEW: Calculate flat region enhancement mask
        flat_enhance_mask = self.flat_enhancement(x_input)
        
        # Apply edge protection to smoothing weights - edges get moderate smoothing
        smooth_weights = smooth_weights * (1 - edge_mask) + 0.3 * edge_mask
        
        # NEW: Apply flat region enhancement - flat regions get more smoothing
        smooth_weights = smooth_weights * (1 + 0.5 * flat_enhance_mask)
        smooth_weights = torch.clamp(smooth_weights, 0, 1)  # Ensure weights stay in [0,1]
        
        # Apply smoothing weights
        # Use sliding window average as base smoothing
        base_smooth = F.avg_pool1d(x_input, kernel_size=5, stride=1, padding=2)
        
        # Combine original input and base smoothing with adaptive weights
        smoothed = smooth_weights * base_smooth + (1 - smooth_weights) * x_input
        
        # Final output
        output = self.output_conv(smoothed).squeeze(1)
        
        return output, smooth_weights.squeeze(1), attention_weights, peak_mask, flat_mask

# Function to create dataset splits
def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into training, validation, and test sets

    The split is drawn on the samples, not on the curves: a single sample yields
    many curve pairs, so splitting the curves at random would put curves of the
    same sample on both sides of the split and make the held-out loss an
    over-optimistic estimate. Assigning whole samples keeps every curve of a
    sample inside one subset.

    Parameters:
        dataset: The complete dataset
        train_ratio: Proportion of samples for training
        val_ratio: Proportion of samples for validation
        test_ratio: Proportion of samples for testing
        random_seed: Random seed for reproducibility

    Returns:
        train_set, val_set, test_set: Split datasets
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Partition the samples, not the curves
    sample_ids = sorted(set(dataset.samples))
    total_samples = len(sample_ids)

    if total_samples < 3:
        raise ValueError(
            f"Only {total_samples} sample(s) were detected, but a training/validation/test split "
            f"needs at least 3. The sample of a curve is taken from the folder named in the "
            f"'# Data from ...' header of the paired file, so check that the header is present and "
            f"that its path ends with the sample folder rather than a shared sub-folder."
        )

    random.Random(random_seed).shuffle(sample_ids)

    n_train = int(train_ratio * total_samples)
    n_val = int(val_ratio * total_samples)

    train_samples = set(sample_ids[:n_train])
    val_samples = set(sample_ids[n_train:n_train + n_val])
    test_samples = set(sample_ids[n_train + n_val:])

    # Collect the curve indices belonging to each partition
    train_idx = [i for i, s in enumerate(dataset.samples) if s in train_samples]
    val_idx = [i for i, s in enumerate(dataset.samples) if s in val_samples]
    test_idx = [i for i, s in enumerate(dataset.samples) if s in test_samples]

    print(f"Sample split: {len(train_samples)} training, {len(val_samples)} validation, "
          f"{len(test_samples)} test")
    print(f"Curve split:  {len(train_idx)} training, {len(val_idx)} validation, "
          f"{len(test_idx)} test")

    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))

# Enhanced training function with flat region loss
def train_model_v2(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=10):
    model = model.to(device)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with more aggressive reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Record training and validation losses
    train_mse_losses = []
    train_mae_losses = []
    val_mse_losses = []
    val_mae_losses = []
    
    print("Starting model training...")
    best_val_mae = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_mse = 0
        epoch_train_mae = 0
        num_train_batches = 0
        
        for batch_idx, (X_batch, y_batch, sf_batch) in enumerate(train_loader):
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            sf_batch = sf_batch.to(device)  # Move smooth factor to device
            
            # Forward pass
            output, smooth_weights, attention_weights, peak_mask, flat_mask = model(X_batch, sf_batch)
            
            # Calculate loss
            mse_loss = mse_criterion(output, y_batch)
            mae_loss = mae_criterion(output, y_batch)
            
            # Add smoothing regularization - penalize second-order differences
            smoothness_loss = torch.mean(torch.abs(output[:, 2:] - 2 * output[:, 1:-1] + output[:, :-2]))
            
            # Add attention sparsity regularization
            attention_weights_mean = torch.mean(attention_weights, dim=1)  # Average head dimension
            attention_weights_mean = torch.mean(attention_weights_mean, dim=1)  # Average sequence dimension
            attention_sparsity = torch.mean(attention_weights_mean * (1 - attention_weights_mean))

            # Calculate gradient of input for peak protection
            input_grad = torch.abs(X_batch[:, 1:] - X_batch[:, :-1])
            input_grad = F.pad(input_grad, (0, 1), mode='replicate')  # Match length

            # Set gradient threshold, e.g., using 95th percentile of gradient values
            threshold = torch.quantile(input_grad, 0.9)
            # Only calculate loss for high gradient regions
            input_peak_mask = (input_grad > threshold).float()
            
            # Use attention_weights_mean instead of attention_weights
            peak_loss = torch.mean(attention_weights_mean * input_grad * input_peak_mask)
            
            # Calculate edge-specific loss with higher weight
            edge_ratio = 0.1  # Edge region ratio
            edge_size = int(X_batch.size(1) * edge_ratio)
            
            # Left edge loss
            left_edge_loss = F.mse_loss(
                output[:, :edge_size], 
                y_batch[:, :edge_size]
            )
            
            # Right edge loss
            right_edge_loss = F.mse_loss(
                output[:, -edge_size:], 
                y_batch[:, -edge_size:]
            )
            
            # Total edge loss with higher weight
            edge_loss = left_edge_loss + right_edge_loss  # Increased edge weight
            
            # Enhanced noise region detection and loss
            # Identify noise regions (high frequency content)
            noise_mask = 1 - peak_mask.unsqueeze(1)  # Non-peak regions are more likely to be noise
            noise_loss = F.mse_loss(output * noise_mask, y_batch * noise_mask) * 2.0
            
            # NEW: Flat region smoothing enhancement loss
            # Encourage higher smoothing weights in flat regions
            flat_region_target = flat_mask.detach()  # Use detected flat regions as target
            flat_smoothing_loss = F.mse_loss(smooth_weights, flat_region_target) * 0.5
            
            # Flat region performance loss - ensure good smoothing in flat areas
            flat_region_mse = F.mse_loss(output * flat_region_target, y_batch * flat_region_target) * 2.0
            
            # Total loss - better balanced regularization terms with flat region enhancement
            total_loss = (mse_loss + mae_loss + 0.2 * smoothness_loss + 
                         0.05 * attention_sparsity + 0.5 * peak_loss + 
                         0.3 * edge_loss + 0.3 * noise_loss +
                         0.2 * flat_smoothing_loss + 0.3 * flat_region_mse)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_mse += mse_loss.item()
            epoch_train_mae += mae_loss.item()
            num_train_batches += 1
        
        avg_train_mse = epoch_train_mse / num_train_batches
        avg_train_mae = epoch_train_mae / num_train_batches
        train_mse_losses.append(avg_train_mse)
        train_mae_losses.append(avg_train_mae)
        
        # Validation phase - use the smooth factors from the validation dataset
        val_mse, val_mae = evaluate_model_v2(model, val_loader)
        val_mse_losses.append(val_mse)
        val_mae_losses.append(val_mae)
        
        # Update learning rate
        scheduler.step(val_mae)
        
        # Check for improvement
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_smoothing_model_NN11_v2.pth')
            print(f"New best model saved, Validation MAE: {best_val_mae:.6f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.2e}, '
                  f'Train MSE: {avg_train_mse:.6f}, Train MAE: {avg_train_mae:.6f}, '
                  f'Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("Training completed!")
    return model, train_mse_losses, train_mae_losses, val_mse_losses, val_mae_losses

# Updated evaluation function for V2 model
def evaluate_model_v2(model, dataloader):
    """
    Evaluate model performance on a dataset
    
    Parameters:
        model: The trained model
        dataloader: DataLoader for evaluation
    
    Returns:
        mse_loss: Mean Squared Error
        mae_loss: Mean Absolute Error
    """
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    total_mse = 0
    total_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch, sf_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            sf_batch = sf_batch.to(device)  # Move smooth factor to device
            
            output, smooth_weights, attention_weights, peak_mask, flat_mask = model(X_batch, sf_batch)
            
            mse_loss = mse_criterion(output, y_batch)
            mae_loss = mae_criterion(output, y_batch)
            
            total_mse += mse_loss.item()
            total_mae += mae_loss.item()
            num_batches += 1
    
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_mse, avg_mae

# Enhanced prediction function for V2 model with log coordinate support
def predict_v2(model, x_values, smooth_factor=16.0, use_log_coordinates=False):
    """
    Use trained model for prediction
    
    Parameters:
        model: Trained model
        x_values: Input values
        smooth_factor: Smoothing factor to control smoothing degree
        use_log_coordinates: Whether the model was trained on log coordinates
    
    Returns:
        smoothed_curve: Smoothed curve
        smooth_weights: Smoothing weights
        attention_weights: Attention weights
        peak_mask: Peak mask
        flat_mask: Flat region mask
    """
    model.eval()  # Set to evaluation mode
    
    # Remove negative values by filtering them out
    valid_mask = x_values > 0
    x_values_filtered = x_values[valid_mask]
    
    # If all values are negative, return zeros
    if len(x_values_filtered) == 0:
        print("Warning: All input values are negative or zero. Returning zeros.")
        zero_array = np.zeros_like(x_values)
        return zero_array, zero_array, np.zeros((1, 1, len(x_values))), zero_array, zero_array
    
    # Apply log transformation if needed
    if use_log_coordinates:
        x_values_filtered = np.log1p(x_values_filtered)
    
    # Convert to tensor and move to device
    x_tensor = torch.tensor(x_values_filtered, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor([smooth_factor], dtype=torch.float32).to(device)
    
    # Prediction
    with torch.no_grad():
        smoothed_curve, smooth_weights, attention_weights, peak_mask, flat_mask = model(x_tensor.unsqueeze(0), sf_tensor)
        smoothed_curve = smoothed_curve.cpu().numpy()
        smooth_weights = smooth_weights.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()
        peak_mask = peak_mask.cpu().numpy()
        flat_mask = flat_mask.cpu().numpy()
    
    # Convert back from log coordinates if needed
    if use_log_coordinates:
        smoothed_curve = np.expm1(smoothed_curve)
    
    # Create full-length arrays with zeros for negative positions
    full_smoothed = np.zeros_like(x_values)
    full_weights = np.zeros_like(x_values)
    full_peak_mask = np.zeros_like(x_values)
    full_flat_mask = np.zeros_like(x_values)
    
    # Fill in the valid positions
    full_smoothed[valid_mask] = smoothed_curve.flatten()
    full_weights[valid_mask] = smooth_weights.flatten()
    full_peak_mask[valid_mask] = peak_mask.flatten()
    full_flat_mask[valid_mask] = flat_mask.flatten()
    
    return full_smoothed, full_weights, attention_weights, full_peak_mask, full_flat_mask

# Enhanced visualization with flat region highlighting and log coordinate support
def visualize_results_v2(original_curve, smoothed_curve, smooth_weights, peak_mask, flat_mask, smooth_factor, use_log_coordinates=False, save_plot=True):
    """
    Visualize original curve, smoothed curve, and smoothing weights with flat region highlighting
    
    Parameters:
        original_curve: Original curve
        smoothed_curve: Smoothed curve
        smooth_weights: Smoothing weights
        peak_mask: Peak detection mask
        flat_mask: Flat region mask
        smooth_factor: Smoothing factor used
        use_log_coordinates: Whether the data is in log coordinates
        save_plot: Whether to save the plot
    """
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original and smoothed curves
    x_range = np.arange(len(original_curve))
    ax1.plot(x_range, original_curve, 'b-', alpha=0.7, label='Original curve')
    ax1.plot(x_range, smoothed_curve, 'r-', linewidth=1.5, label='Smoothed curve')
    
    # Highlight flat regions
    flat_threshold = 0.5
    flat_regions = flat_mask > flat_threshold
    if np.any(flat_regions):
        # Find contiguous flat regions
        flat_indices = np.where(flat_regions)[0]
        if len(flat_indices) > 0:
            # Group consecutive indices
            flat_groups = np.split(flat_indices, np.where(np.diff(flat_indices) != 1)[0] + 1)
            for group in flat_groups:
                if len(group) > 5:  # Only highlight significant flat regions
                    ax1.axvspan(group[0], group[-1], alpha=0.2, color='green', label='Flat regions' if group is flat_groups[0] else "")
    
    # Highlight edge regions
    edge_ratio = 0.1
    edge_size = int(len(original_curve) * edge_ratio)
    ax1.axvspan(0, edge_size, alpha=0.2, color='yellow', label='Edge regions')
    ax1.axvspan(len(original_curve)-edge_size, len(original_curve), alpha=0.2, color='yellow')
    
    # Set y-axis scale based on coordinate system
    if use_log_coordinates:
        ax1.set_yscale('log')
        ax1.set_ylabel('Log Intensity')
    else:
        ax1.set_ylabel('Intensity')
    
    ax1.set_title(f'Curve Smoothing Effect Comparison (Smooth Factor: {smooth_factor})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot smooth weights and flat mask
    ax2.plot(x_range, smooth_weights, 'g-', alpha=0.8, label='Smooth weights')
    ax2.plot(x_range, flat_mask, 'm-', alpha=0.6, label='Flat region mask')
    ax2.axvspan(0, edge_size, alpha=0.2, color='yellow')
    ax2.axvspan(len(original_curve)-edge_size, len(original_curve), alpha=0.2, color='yellow')
    ax2.set_ylabel('Weight / Mask Value')
    ax2.set_title('Smooth Weights and Flat Region Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot peak mask and flat mask comparison
    ax3.plot(x_range, peak_mask, 'r-', alpha=0.7, label='Peak mask')
    ax3.plot(x_range, flat_mask, 'm-', alpha=0.7, label='Flat mask')
    ax3.axvspan(0, edge_size, alpha=0.2, color='yellow')
    ax3.axvspan(len(original_curve)-edge_size, len(original_curve), alpha=0.2, color='yellow')
    ax3.set_xlabel('Data point index')
    ax3.set_ylabel('Mask Value')
    ax3.set_title('Peak vs Flat Region Detection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('smoothing_visualization_NN11_v2.png', dpi=300, bbox_inches='tight')
        print("Visualization results saved as 'smoothing_visualization_NN11_v2.png'")
    
    plt.show()

# Enhanced main function with V2 model and log coordinate support
if __name__ == "__main__":
    # Set folder path
    folder_path = "./curve_pairs"  # Replace with your folder path
    
    # Set whether to use log coordinates
    use_log_coordinates = True  # Set to True to use log coordinates
    
    # Create dataset instance
    print("Loading dataset...")
    dataset = CurveDataset(folder_path, use_log_coordinates=use_log_coordinates)
    
    # Create data splits
    train_set, val_set, test_set = create_data_splits(
        dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    
    # Create data loaders
    batch_size = 32  # Reduced batch size for better generalization
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize enhanced model
    model = AdaptiveSmoothingNetV2(base_channels=64)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train enhanced model
    model, train_mse_losses, train_mae_losses, val_mse_losses, val_mae_losses = train_model_v2(
        model, train_loader, val_loader, num_epochs=500, learning_rate=0.005
    )
    
    # Evaluate on test set
    test_mse, test_mae = evaluate_model_v2(model, test_loader)
    print(f"\nTest set performance - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'final_smoothing_model_NN11_v2.pth')
    print("Final model saved as 'final_smoothing_model_NN11_v2.pth'")
    
    # Example: Use model for prediction on a test sample
    test_sample_idx = np.random.randint(0, len(test_set))
    test_sample_input, test_sample_target, test_sample_sf = dataset[test_set.indices[test_sample_idx]]
    original_length = dataset.curve_lengths[test_set.indices[test_sample_idx]]
    
    # Test with smooth factor 256.0 only (removed the loop)
    test_sf = 256.0
    smoothed_curve, smooth_weights, attention_weights, peak_mask, flat_mask = predict_v2(
        model, test_sample_input.cpu().numpy(), test_sf, use_log_coordinates=use_log_coordinates
    )
    
    # Enhanced visualization with flat region information
    print(f"Generating enhanced visualization charts for smooth factor {test_sf}...")
    visualize_results_v2(
        test_sample_input[:original_length].cpu().numpy(),
        smoothed_curve[:original_length],
        smooth_weights[:original_length],
        peak_mask[:original_length],
        flat_mask[:original_length],
        test_sf,
        use_log_coordinates=use_log_coordinates,
        save_plot=True
    )
    
    # Calculate flat region smoothing statistics
    flat_threshold = 0.5
    flat_regions = flat_mask[:original_length] > flat_threshold
    if np.any(flat_regions):
        flat_smooth_mean = np.mean(smooth_weights[:original_length][flat_regions])
        print(f"Smooth factor {test_sf}: Mean smooth weight in flat regions = {flat_smooth_mean:.4f}")
    print(f"Smooth factor {test_sf}: Overall mean smooth weight = {np.mean(smooth_weights[:original_length]):.4f}")