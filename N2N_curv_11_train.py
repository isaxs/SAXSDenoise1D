# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import copy
import random
import time
import warnings
import atexit
import hashlib
import json
import shutil
import signal
import subprocess
import chardet
warnings.filterwarnings('ignore')

# ==================== Global Configuration ====================
class Config:
    # Data related parameters
    # The corpus is partitioned once, by sample, into three physical folders
    # (see _split_manifest.json). This script only reads them - it never splits
    # the data at run time.
    data_root = "./Data/"            # container holding the three splits
    train_root = "./Data/train/"
    val_root = "./Data/val/"
    test_root = "./Data/test/"
    data_suffix = ".dat"   # Data file suffix
    file_encoding = None   # File encoding, if None auto-detect
    
    # Training related parameters
    # Pairs are re-drawn on every access, so this is the length of one epoch and
    # not the size of a fixed pool. None -> one pair per training frame, i.e.
    # every frame is used about once as an input per epoch.
    pairs_per_epoch = None
    batch_size = 1         # Batch size must be 1 for variable length
    # Hard ceiling for a run. It is raised freely to extend a run that stopped
    # early, which is why the schedule below is configured separately: the two
    # answer different questions.
    num_epochs = 500
    learning_rate = 1e-3   # Initial learning rate
    # Designed split ratios, train : validation : test = 70 : 15 : 15, the same
    # ratio used for the Noise2Clean benchmark. These are recorded for reference
    # and reporting only - the actual partition is the one baked into Data/.
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    
    # ---- Model architecture ----
    # 'dilated': full-resolution residual CNN (see DilatedResidualDenoiser1D).
    # 'unet'   : the resampling U-Net used by v10 and the first v11 cut.
    arch = "dilated"
    # Mild decoupled weight decay. The evidence that the old model was
    # memorising rather than learning is that its edge over a fixed Gaussian
    # transferred only partly (6.5% better on the training pool, 0.9% on
    # validation). Most of the fix is the smaller network; this is the cheap
    # second knob, and setting it to 0.0 restores plain Adam.
    weight_decay = 1e-5

    # Model saving
    # Versioned names on purpose: v10's weights are the only baseline this
    # architecture change can be judged against, so nothing here overwrites
    # them. (The eight downstream *_test.py scripts are unaffected either way -
    # they all load their own ./models/denoise_1d.pth and carry their own copy
    # of the older network.)
    model_save_path = "./models/denoiseN2N_1d_v11.pth"  # Best model save path
    final_model_save_path = "./models/denoiseN2N_1d_v11_final.pth"  # Final model save path
    
    # Learning rate scheduler parameters
    lr_scheduler_type = "cosine"  # LR scheduler type
    # The epochs the cosine actually anneals over. Deliberately separate from
    # num_epochs: num_epochs is the run's ceiling and moves whenever a run is
    # extended, while this decides the shape of the trajectory and must not
    # move with it. Tied to a 500-epoch ceiling and stopped at epoch 113, the
    # LR was still 8.8e-4 when the run ended - it never annealed at all.
    lr_cosine_t_max = 150
    lr_min = 1e-5  # Minimum learning rate
    lr_patience = 10  # Patience for LR reduction
    lr_factor = 0.5  # LR reduction factor

    # Early Stop parameters
    # The patience used to be 100 to survive a metric that was re-drawn from a
    # noisy pool every epoch. With a frozen validation subset the run-to-run
    # noise is far smaller, so a shorter window is enough.
    early_stop_patience = 50  # Early stopping patience
    early_stop_min_delta = 1e-5  # Minimum improvement delta

    # ---- Validation protocol ----
    # Validation pairs used to be re-drawn every epoch from a 75k-pair pool,
    # which made the epoch-to-epoch number move by more than any real
    # improvement: picking the best epoch was picking the luckiest draw. A
    # frozen subset makes consecutive epochs comparable. It is drawn uniformly
    # over the ordered-pair pool, not one draw per sample, so that it shares a
    # weighting with the exhaustive test pass.
    val_pairs_fixed = 3000  # None to go back to re-drawing every epoch
    val_pairs_seed = 12345

    # ---- Checkpoint / resume ----
    # The full training state (weights, optimizer, scheduler, early-stopping
    # counter, RNG) is written to resume_checkpoint_path at the end of every
    # epoch, so a run can be stopped at any moment and picked up where it left
    # off. Nothing is split across files: one checkpoint is one consistent
    # epoch boundary.
    resume = True  # Resume automatically when a checkpoint exists
    resume_checkpoint_path = "./models/checkpoints/denoiseN2N_1d_v11_resume.pth"
    save_resume_every_n_epochs = 1  # ~10 MB per write, ~1-2% of an epoch
    # The patience counter is restored faithfully by default. Set this to True
    # to hand the model a fresh patience budget instead - best_loss and the
    # saved best model are kept either way, so a worse model can never
    # overwrite the best one.
    reset_early_stop_on_resume = False
    # Raised num_epochs past the checkpoint's budget is the normal way to
    # extend a finished run, so it just continues. Only a checkpoint that
    # already covers the whole budget is "nothing left to do".
    resume_restart_if_completed = True  # False -> print guidance and exit instead
    # Warm start: a run that predates this feature has no resume state on
    # disk (only weights), so without this the first resume would train from
    # scratch. With it on, the weights and best_val_loss of an older .pth are
    # adopted while optimizer/scheduler/counter/RNG start fresh. That is a
    # warm restart, not a true resume - turn it off for a clean comparison.
    #
    # It only applies when there is no resume checkpoint, so once a run is
    # under way this setting stops mattering.
    #
    # Off for v11: the older files were written by the BatchNorm network, so
    # they cannot load into this one, and their val_loss was measured against
    # the old re-drawn validation protocol - carrying it over would seed
    # best_val_loss with a number on a different scale.
    bootstrap_from_legacy_best = False
    bootstrap_source_path = "./models/denoiseN2N_1d_final.pth"

    # ---- Concurrency ----
    # Guards against two instances of this script writing the same files.
    use_lock_file = True
    lock_file_path = "./models/.n2n_10_train.lock"

    # Samples whose frames drift over the measurement series. Detected by
    # splitting each sample's frames into two halves and comparing the two
    # half-averaged curves against the noise floor: ratio = observed difference
    # / difference expected from pure noise. 1.0 means the frames really are
    # independent observations of one static signal. Measured ratios: PP-S 3.52,
    # iPB1-S 2.27, Pebax-6333-S 1.52, POM-RT18-S 1.49, FG1901-GT-S 1.34,
    # P320 1.27 (the nearest clean sample is HMDI-soft at 1.17). For these a
    # pair drawn across the whole series is not "same signal + independent
    # noise", so pairing is restricted to temporally nearby frames; every other
    # sample keeps the full pair space. Re-derive with the audit script if the
    # data changes.
    drift_samples = {"PP-S", "iPB1-S", "Pebax-6333-S", "POM-RT18-S",
                     "FG1901-GT-S", "P320"}
    drift_max_frame_gap = 5  # |i - j| allowed when pairing a drifting sample

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

# The corpus used to be partitioned here, at run time, by splitting the sample
# folders of Data/ on every launch. That partition is now materialised on disk as
# Data/train, Data/val and Data/test (see _split_manifest.json for the seed,
# the ratios and the exact folder assignment), so nothing is split at run time.


# ==================== Dataset Class ====================
class FullCurveDataset(Dataset):
    """Dataset over whole curves, drawing noise-noise pairs within one sample.

    In the default (sampling) mode nothing is pre-computed: every __getitem__
    picks a sample folder, draws two of its frames and returns them in a random
    order. A pair and its reverse are equally valid Noise2Noise examples - both
    directions estimate the same conditional mean of the clean curve - so
    randomising the direction uses the whole pair space instead of one arbitrary
    half of it. Because the draw happens on every access, successive epochs see
    different pairs and the whole corpus is eventually covered, which a frozen
    pool of pairs would never do.

    In exhaustive mode the full list of ordered pairs is materialised instead.
    That is what the held-out evaluation uses: one forward pass per pair, and no
    sampling variance in the reported number.

    In fixed mode a subset of that pool is frozen at construction and reused
    every time. Re-drawing the validation pairs each epoch used to move the
    reported loss by more than any real improvement, which made picking the
    best epoch a matter of luck and let noise trigger early stopping. Freezing
    turns the subset's own sampling error into a constant offset that cancels
    out of any epoch-to-epoch comparison. The subset is drawn uniformly over
    the ordered-pair pool, not one draw per sample: the two weightings differ
    by 17% on val and 24% on test, and pool-uniform is what the exhaustive
    evaluation reports, so only that weighting makes the two comparable.
    """

    def __init__(self, data_root, num_pairs=None, folders=None, exhaustive=False,
                 fixed_pairs=None, fixed_pairs_seed=None):
        self.data_root = data_root
        self.exhaustive = exhaustive
        if exhaustive and fixed_pairs:
            raise ValueError("exhaustive=True enumerates every pair and "
                             "fixed_pairs=N freezes a subset of them; pass one.")

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

        # Only a sample with at least two frames can yield a noise-noise pair
        self.valid_folders = [f for f in self.folders if f in self.folder_files]

        if not self.valid_folders:
            print("Warning: no sample holds at least two readable frames")
            self.pairs = []
            self.num_pairs = 0
            self.n_frames = 0
            self.total_pairs = 0
            self.mode = 'sampled'
            self.pairs_digest = None
            return

        self.n_frames = sum(len(self.folder_files[f]) for f in self.valid_folders)
        # Every ordered pair this split can ever yield. For train/validation it
        # is the pool the per-epoch draws come from, which is what tells you how
        # much of the corpus a run has actually seen; for the held-out split it
        # is the evaluation set itself.
        self.total_pairs = sum(len(self._allowed_index_pairs(f))
                               for f in self.valid_folders)

        # The full ordered pool, materialised only when a concrete list is
        # needed (exhaustive evaluation or a frozen subset).
        pool = None
        if exhaustive or fixed_pairs:
            pool = [(files[i], files[j])
                    for folder in self.valid_folders
                    for files in [self.folder_files[folder]]
                    for i, j in self._allowed_index_pairs(folder)]

        if exhaustive:
            self.mode = 'exhaustive'
            self.pairs = pool
        elif fixed_pairs:
            if fixed_pairs > len(pool):
                raise ValueError(f"fixed_pairs={fixed_pairs} exceeds the "
                                 f"{len(pool)} ordered pairs this split holds")
            # A private Random, never the module-level one: that stream is what
            # the training draws consume, and seeding it here would shift every
            # pair the run sees afterwards - silently breaking the guarantee
            # that a resumed run draws the same pairs an uninterrupted one would.
            self.mode = 'fixed'
            self.pairs = random.Random(fixed_pairs_seed).sample(pool, fixed_pairs)
        else:
            self.mode = 'sampled'
            self.pairs = None  # drawn on demand, there is no pool

        self.num_pairs = (len(self.pairs) if self.pairs is not None
                          else (self.n_frames if num_pairs is None else num_pairs))

        lengths = [len(self.file_data[f])
                   for folder in self.valid_folders
                   for f in self.folder_files[folder]]
        print(f"Dataset ready: {len(self.valid_folders)} samples, {len(lengths)} frames, "
              f"{self.num_pairs} pairs per epoch"
              + {'exhaustive': " (exhaustive)",
                 'fixed': f" (frozen subset of the {self.total_pairs:,}-pair pool)",
                 'sampled': ""}[self.mode])
        print(f"Curve length range: {min(lengths)} to {max(lengths)} points")
        print(f"Average curve length: {np.mean(lengths):.0f} points")

        drifting = sorted(set(self.valid_folders) & config.drift_samples)
        if drifting:
            print(f"Drifting samples in this split (pairing limited to "
                  f"|i-j| <= {config.drift_max_frame_gap}): {', '.join(drifting)}")

        if self.mode == 'fixed':
            # A digest so two runs can prove they froze the same pairs, and so a
            # silent reordering of the corpus shows up instead of quietly
            # changing what the metric means.
            self.pairs_digest = hashlib.sha1(
                '\n'.join(f"{a}|{b}" for a, b in self.pairs).encode()
            ).hexdigest()[:12]
            print(f"Frozen subset: {self.num_pairs:,} of {self.total_pairs:,} "
                  f"ordered pairs, seed {fixed_pairs_seed}, digest {self.pairs_digest}")

    def _allowed_index_pairs(self, folder):
        """(i, j) frame index pairs that may form a noise-noise pair.

        Normally any two distinct frames of the sample. For a drifting sample the
        frames are not independent observations of one static signal, so only
        temporally nearby frames are allowed.
        """
        n = len(self.folder_files[folder])
        gap = config.drift_max_frame_gap
        drifting = folder in config.drift_samples
        return [(i, j) for i in range(n) for j in range(n)
                if i != j and (not drifting or abs(i - j) <= gap)]

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        if self.pairs is not None:
            # a materialised list: the exhaustive pool or a frozen subset
            file1, file2 = self.pairs[idx]
        else:
            # Fresh pair on every access, direction included
            folder = random.choice(self.valid_folders)
            files = self.folder_files[folder]
            n = len(files)
            if folder in config.drift_samples:
                # draw the second frame from a window around the first
                i = random.randrange(n)
                gap = config.drift_max_frame_gap
                j = random.choice([k for k in range(max(0, i - gap),
                                                     min(n, i + gap + 1))
                                   if k != i])
            else:
                i, j = random.sample(range(n), 2)
            file1, file2 = files[i], files[j]
            if random.random() < 0.5:
                file1, file2 = file2, file1

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
def curve_norm(channels):
    """Per-curve, per-channel normalisation that behaves the same in train and eval.

    GroupNorm with one group per channel is InstanceNorm1d(C, affine=True) by
    another name, and it is bit-for-bit the statistic BatchNorm1d computes here
    at batch_size=1 in *training* mode: mean and biased variance over the one
    curve in the batch. That is the mode the diagnostic measured as the better
    of the two (0.008067 against 0.008671 on 600 validation pairs), so this
    keeps the function the weights were optimised against and only stops
    eval() from swapping it for population statistics.

    It carries no running buffers, so there is nothing to fall out of step with
    the data, and nothing that train() and eval() can disagree about.

    num_groups=channels is deliberate. GroupNorm(8, C) is also per-curve, but it
    pools the statistic over 8 channels and so removes only a shared group level
    rather than each channel's own mean and variance - a different function from
    the one this pipeline has been training with. If training ever turns spiky,
    8 is the one knob to try.
    """
    return nn.GroupNorm(num_groups=channels, num_channels=channels)


class AdaptiveUNet1D_Denoising(nn.Module):
    """Adaptive 1D U-Net that adjusts to input curve dimensions.

    Four levels, 1 -> 32 -> 64 -> 128 -> 256 as before, but every decoder stage
    is now up-conv -> exact-length align -> concat(skip) -> two convs. Two
    things changed in the decoder:

    - The full-resolution encoder features reach the output. Previously enc1's
      e1 was never used again and the finest input to the decoder was at L/2,
      with the last hop being a bare F.interpolate of that half-resolution map.
      For a task whose whole point is preserving fine structure while removing
      ~3% noise, discarding the finest features was a handicap.
    - A concat is always followed by convolutions before anything resamples it
      again. Previously the concatenated tensor went straight into the next
      transposed conv, so no layer ever saw the combined feature map.
    """

    # Bump this whenever the forward path or any layer definition changes. The
    # resume path compares weight names and shapes, which cannot see a change
    # that preserves both (GroupNorm(8, C) -> GroupNorm(C, C), a removed skip,
    # a reordered concat); this tag can.
    ARCH_TAG = 'unet1d-v11-gn-percurve-fullres-skip'

    def __init__(self, in_channels=1):
        super().__init__()

        # ---- Encoder (unchanged except for the normalisation) ----
        self.enc1 = nn.Sequential(                               # [B,  32, L  ]
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            curve_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            curve_norm(32),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(                               # [B,  64, L/2]
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2),
            curve_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            curve_norm(64),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(                               # [B, 128, L/4]
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            curve_norm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            curve_norm(128),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(                         # [B, 256, L/8]
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            curve_norm(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            curve_norm(256),
            nn.ReLU(inplace=True)
        )

        # ---- Decoder: up-conv -> align -> concat(skip) -> two convs ----
        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.dec3 = nn.Sequential(                               # [B, 128, L/4]
            nn.Conv1d(256, 128, kernel_size=3, padding=1),    # 128 up + 128 skip
            curve_norm(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            curve_norm(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.dec2 = nn.Sequential(                               # [B,  64, L/2]
            nn.Conv1d(128, 64, kernel_size=3, padding=1),     #  64 up +  64 skip
            curve_norm(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            curve_norm(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        self.dec1 = nn.Sequential(                               # [B,  32, L  ]
            nn.Conv1d(64, 32, kernel_size=3, padding=1),      #  32 up +  32 skip
            curve_norm(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            curve_norm(32),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv1d(32, in_channels, kernel_size=1)

        # Start as the identity. With the default init the untrained network
        # moves a curve by 0.23-0.40 in log1p units - 20-40x the noise scale it
        # is supposed to remove - so the first epochs are spent re-learning
        # "output the input" while an always-active gradient clip bites. Zeroing
        # the head costs nothing (its input still carries gradient, so no
        # symmetry is lost) and puts epoch 1 at the identity baseline.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _align(x, ref):
        """Match x to ref's length when the transposed conv is off by one.

        The encoder's stride-2 convs round an odd length up (565 -> 283 -> 142
        -> 71) and a transposed conv doubles whatever it is given, which is not
        always that same number back: 580 -> 290 -> 145 -> 73 up-convolves to
        146, one more than e3 has. About half the corpus has an odd length, so
        this fires often. Equal lengths take the identity path.
        """
        if x.shape[2] == ref.shape[2]:
            return x
        return F.interpolate(x, size=ref.shape[2], mode='linear', align_corners=False)

    def forward(self, x):
        original_size = x.shape[2]

        # ---- Encoder ----
        e1 = self.enc1(x)              # [B,  32, L  ]
        e2 = self.enc2(e1)             # [B,  64, L/2]
        e3 = self.enc3(e2)             # [B, 128, L/4]
        b = self.bottleneck(e3)        # [B, 256, L/8]

        # ---- Decoder: every level gets a skip, and every concat is fused by
        #      convolutions before it is resampled again. ----
        d3 = self._align(self.up3(b), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))     # [B, 128, L/4]

        d2 = self._align(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))     # [B,  64, L/2]

        d1 = self._align(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))     # [B,  32, L  ]

        out = self.head(d1)                            # [B,   1, L  ]
        if out.shape[2] != original_size:
            # e1 is never resampled, so this is a no-op today; kept so the
            # output length is guaranteed regardless of how the encoder rounds.
            out = F.interpolate(out, size=original_size, mode='linear',
                                align_corners=False)

        # Global residual: the network predicts a correction to the input.
        return x + out


def group_norm(channels, num_groups=8):
    """GroupNorm that works at any batch size and is identical in train and eval.

    Fewer groups than channels on purpose here. GroupNorm(C, C) - what the U-Net
    above uses - normalises every channel of every curve on its own, which is the
    most aggressive option available and discards each channel's own level.
    The denoising problem needs that level: the optimal amount of smoothing
    varies four-fold across the validation samples (sigma 1.5 for DEXCO-SIS-4211
    against 6 for PET-Xiao-S), so a network that cannot see how loud a curve is
    cannot pick how hard to smooth it. Pooling the statistic over a group keeps
    each channel's deviation from its group's level.
    """
    return nn.GroupNorm(min(num_groups, channels), channels)


class ResidualDilatedBlock1D(nn.Module):
    """Two dilated convs plus a residual add. Starts as the identity."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.norm = group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.act = nn.ReLU(inplace=True)
        # Zero the second conv so the block contributes nothing at
        # initialisation. The residual stream then carries the curve through
        # untouched and each block adds its correction as it learns, which is
        # what keeps a twelve-conv stack trainable without a normaliser on the
        # residual path.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.act(self.norm(self.conv1(x)))
        # No activation on the residual path. A ReLU there would clip every
        # negative value before the next block, and the residual stream is what
        # carries the curve to the output - a denoiser cannot afford to lose
        # half of its range on the way. The nonlinearity lives inside the block,
        # which is the ConvNeXt/transformer arrangement.
        return x + self.conv2(h)


class DilatedResidualDenoiser1D(nn.Module):
    """Full-resolution residual denoiser with a dilated receptive field.

    Replaces the U-Net above, for three measured reasons.

    Every level of a U-Net halves the resolution and doubles it back, which is a
    low-pass filter by construction. That matches what the U-Net was actually
    doing: a 717k-parameter U-Net beat a fixed Gaussian blur by only 0.9% on the
    validation pool (0.009213 against 0.009297) while beating it by 6.5% on the
    training pool (0.005983 against 0.006396). It learned the right amount of
    smoothing for the 46 training samples and could not transfer it, because a
    network built around resampling has little else to offer.

    Downsampling to L/8 also throws away the resolution this task is about: the
    curves are 536-580 points and the whole job is to keep their shape while
    removing ~3% noise.

    So: no resampling anywhere. Dilations 1..32 give a receptive field of about
    257 points - enough to see the local level that sets the smoothing strength -
    while staying at full resolution, and do it with roughly a eighth of the
    parameters (about 148k against 717k), which is the cleaner fix for a model
    that was memorising 46 samples.
    """

    ARCH_TAG = 'dilatedres1d-v11-fullres-d1to32-w64'

    def __init__(self, in_channels=1, width=64, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=5, padding=2),
            group_norm(width),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualDilatedBlock1D(width, d)
                                    for d in dilations])
        self.head = nn.Conv1d(width, in_channels, kernel_size=1)
        # Start at the identity: untrained, this network should change nothing.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.body(self.stem(x))
        # Global residual. Every conv uses padding=dilation, so the length is
        # preserved exactly and no interpolation is needed anywhere.
        return x + self.head(h)


# Which of the two denoisers a run trains. Switching this changes ARCH_TAG, and
# the resume path refuses to carry weights across architectures, so a new run
# starts clean rather than half-loading.
MODEL_CLASSES = {
    'dilated': DilatedResidualDenoiser1D,
    'unet': AdaptiveUNet1D_Denoising,
}


def build_model(in_channels=1):
    name = getattr(config, 'arch', 'dilated')
    if name not in MODEL_CLASSES:
        raise SystemExit(f"ERROR: config.arch={name!r} is not one of "
                         f"{sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[name](in_channels=in_channels)


# ==================== Resume / Checkpoint Utilities ====================
RESUME_CHECKPOINT_VERSION = 2  # 2: adds arch_tag

# A change in these makes the historical losses and the Adam moments
# meaningless, so resuming across them is refused rather than silently mixed.
_RESUME_ABORT_KEYS = ('clip_negative', 'use_log1p', 'data_root', 'train_root',
                      'val_root', 'test_root', 'data_suffix', 'file_encoding',
                      # Changing the frozen subset changes what the historical
                      # val losses mean, and best_val_loss and the patience
                      # counter are compared by value against the new metric.
                      # Better to stop than to carry over a meaningless best.
                      'val_pairs_fixed', 'val_pairs_seed')
# These are legitimate things to change between runs; warn, do not block.
_RESUME_WARN_KEYS = ('num_epochs', 'lr_cosine_t_max', 'learning_rate', 'batch_size',
                     'pairs_per_epoch', 'lr_scheduler_type', 'lr_min', 'lr_patience',
                     'lr_factor', 'early_stop_patience', 'early_stop_min_delta',
                     'drift_samples', 'drift_max_frame_gap', 'random_seed')
_RESUME_REQUIRED_KEYS = ('arch_tag', 'epochs_completed', 'model_state_dict',
                         'best_model_state_dict', 'best_val_loss',
                         'optimizer_state_dict', 'scheduler_state_dict', 'early_stop',
                         'train_losses', 'val_losses', 'rng', 'config')


def snapshot_config():
    """The configuration as a plain dict.

    Not config.__dict__ - every field of Config is a class attribute and there
    is no __init__, so an instance's __dict__ is empty. Every checkpoint
    written before this function existed recorded {} for its config.
    """
    return {k: v for k, v in vars(Config).items() if not k.startswith('__')}


def atomic_torch_save(obj, path, retries=5, delay=0.5):
    """Write through a .tmp file and rename, so `path` is never seen half-written.

    A plain torch.save truncates the old file first; a crash inside it leaves a
    broken checkpoint that every downstream test script would fail to load.
    os.replace onto the same volume is atomic on Windows and POSIX alike.

    Never raises: losing one epoch of progress is survivable, aborting a
    500-epoch run because a file was briefly locked is not.
    """
    tmp = path + '.tmp'
    parent = os.path.dirname(path)
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())  # get the bytes down before the rename
    except Exception as e:
        print(f"WARNING: could not write {tmp} ({e}); the previous file is untouched")
        return False

    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return True
        except OSError as e:
            if attempt == retries - 1:
                # The .tmp holds a complete file, only the rename failed, so
                # leave it in place and say where it is.
                print(f"WARNING: could not replace {path} ({e}); "
                      f"the new state is complete in {tmp}")
                return False
            time.sleep(delay)


def capture_rng_state():
    """The four random streams the data pipeline draws from.

    Pairs are drawn inside __getitem__ (python's random) and the loader
    shuffles with the torch generator, so restoring these makes the epoch
    after a resume see the same pairs the interrupted run would have.
    """
    state = {'python': random.getstate(),
             'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    try:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda'])
        return True
    except Exception as e:
        print(f"WARNING: could not restore the RNG state ({e}); "
              f"training continues on a fresh random stream")
        return False


def save_resume_checkpoint(path, epochs_completed, completed, exit_reason, model,
                           optimizer, scheduler, early_stopping, best_val_loss,
                           best_epoch, best_model_state, train_losses, val_losses):
    """Persist everything needed to continue, at an epoch boundary."""
    payload = {
        'checkpoint_version': RESUME_CHECKPOINT_VERSION,
        'script': os.path.basename(__file__),
        'saved_at': time.time(),
        # Weight names and shapes cannot see every architecture change, so the
        # model carries its own tag and the resume path checks it first.
        'arch_tag': getattr(model, 'ARCH_TAG', None),
        # 1-indexed count of finished epochs, which is also the index of the
        # next one. Deliberately not named 'epoch': the older model files use
        # that name for the 0-indexed epoch that produced the best weights.
        'epochs_completed': epochs_completed,
        'completed': completed,
        'exit_reason': exit_reason,
        'model_state_dict': model.state_dict(),
        # Without this the best weights would be lost on a restart and the
        # close-out would evaluate and re-save whatever the last epoch held.
        'best_model_state_dict': best_model_state,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        # Separate from best_val_loss: this pair drives the patience counter,
        # that one drives whether the best file is overwritten.
        'early_stop': {'counter': early_stopping.counter,
                       'best_loss': early_stopping.best_loss},
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
        'rng': capture_rng_state(),
        'config': snapshot_config(),
    }
    return atomic_torch_save(payload, path)


def load_resume_checkpoint(path):
    """Load a resume checkpoint, or None when there is nothing to resume from."""
    if not path or not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise SystemExit(f"ERROR: cannot read the resume checkpoint {path}: {e}\n"
                         f"       Delete that file to start from scratch.")
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise SystemExit(f"ERROR: {path} is not a resume checkpoint.\n"
                         f"       Delete that file to start from scratch.")
    version = ckpt.get('checkpoint_version')
    if version != RESUME_CHECKPOINT_VERSION:
        raise SystemExit(f"ERROR: {path} has checkpoint_version={version!r}, "
                         f"this script writes version {RESUME_CHECKPOINT_VERSION}.\n"
                         f"       Delete that file to start from scratch.")
    missing = [k for k in _RESUME_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise SystemExit(f"ERROR: resume checkpoint {path} is missing {missing}.\n"
                         f"       Delete that file to start from scratch.")
    return ckpt


def validate_resume_config(ckpt_config):
    """Compare the checkpoint's config against the current one."""
    current = snapshot_config()
    if not ckpt_config:
        print("NOTE: this checkpoint carries no config snapshot (it predates "
              "snapshot_config), so the consistency check is skipped.")
        return

    aborts = [(k, ckpt_config[k], current[k]) for k in _RESUME_ABORT_KEYS
              if k in ckpt_config and k in current and ckpt_config[k] != current[k]]
    if aborts:
        print("ERROR: the data or its preprocessing changed since this checkpoint "
              "was written:")
        for key, old, new in aborts:
            print(f"  {key}: {old!r} -> {new!r}")
        print("  Historical losses and the Adam moments are not comparable across "
              "that change.")
        print(f"  Delete {config.resume_checkpoint_path} to start from scratch.")
        raise SystemExit(1)

    warnings = []
    for key in _RESUME_WARN_KEYS:
        if key in ckpt_config and key in current and ckpt_config[key] != current[key]:
            warnings.append((key, ckpt_config[key], current[key]))
    if warnings:
        print("NOTE: configuration differs from the checkpoint's snapshot:")
        for key, old, new in warnings:
            print(f"  {key}: {old!r} -> {new!r}")
        if any(k == 'learning_rate' for k, _, _ in warnings):
            print("  learning_rate has no effect on a resumed run: the restored "
                  "optimizer state carries its own lr.")
        if any(k == 'batch_size' or k == 'pairs_per_epoch' for k, _, _ in warnings):
            print("  The number of steps per epoch changed, so the schedule's "
                  "per-epoch meaning changed with it.")


def backup_legacy_models():
    """Copy the model files aside before a fresh run overwrites them."""
    for path in (config.model_save_path, config.final_model_save_path):
        if os.path.exists(path):
            try:
                shutil.copy2(path, path + '.bak')
                print(f"  backed up {path} -> {path}.bak")
            except Exception as e:
                print(f"  WARNING: could not back up {path} ({e})")


def bootstrap_from_legacy(path, model):
    """Warm-start from a model file written before resume support existed.

    Such a file holds weights only, so the optimizer moments, the LR schedule,
    the patience counter and the RNG are all new. That makes this a warm
    restart rather than a continuation, and the caller is told so.
    """
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
    except Exception as e:
        print(f"WARNING: warm start from {path} failed ({e}); training from scratch")
        return None
    if 'val_loss' not in ck:
        print(f"WARNING: {path} has no val_loss to seed best_val_loss; "
              f"training from scratch")
        return None

    val_loss = float(ck['val_loss'])
    # The legacy 'epoch' field is NOT a count of trained epochs and must not be
    # used as a resume point: the best-model file stores the 0-indexed epoch
    # that produced the best weights, while the final-model file stores
    # num_epochs outright. Deriving a start from it can land at or past
    # num_epochs, leaving the loop with nothing to run while the run still
    # reports itself as finished.
    #
    # Since the optimizer and the LR schedule both start fresh here, this is a
    # new schedule over old weights: it begins at epoch 1 and the source's own
    # epoch is carried along as provenance only.
    return {'warm_start': True, 'source': path, 'best_val_loss': val_loss,
            'best_model_state': copy.deepcopy(ck['model_state_dict']),
            'best_epoch': 0, 'epochs_completed': 0,
            'source_epoch': ck.get('epoch')}


class ProcessLock:
    """Single-instance lock, so two copies of this script cannot interleave writes.

    A stale lock is taken over rather than blocking forever: the recorded PID
    is probed, and only a live process keeps the lock.
    """

    def __init__(self, path):
        self.path = path
        self.pid = os.getpid()
        self.acquired = False
        self.holder = None

    @staticmethod
    def _pid_alive(pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        if os.name == 'nt':
            # On Windows os.kill(pid, 0) is NOT a liveness probe - CPython
            # routes any signal to TerminateProcess, which would kill the very
            # process being asked about. Ask tasklist instead.
            try:
                out = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/NH', '/FO', 'CSV'],
                    capture_output=True, timeout=15)
                text = out.stdout.decode('utf-8', 'replace')
            except Exception:
                return True  # cannot tell -> assume alive, refuse to start
            for line in text.splitlines():
                # Compare the PID column, not a substring of the whole line:
                # a process named "12345" would otherwise match PID 1234.
                cols = [c.strip().strip('"') for c in line.split(',')]
                if len(cols) > 1 and cols[1] == str(pid):
                    return True
            return False
        try:
            os.kill(pid, 0)  # POSIX: signal 0 only checks that it exists
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def _read_holder(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def acquire(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        for _ in range(4):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = self._read_holder()
                if holder is None:
                    # Either a process is between creating the file and writing
                    # it, or the write was interrupted. Only a recent one can
                    # still be inside that window.
                    try:
                        age = time.time() - os.path.getmtime(self.path)
                    except OSError:
                        age = 1e9
                    if age < 60:
                        self.holder = {'pid': '?',
                                       'note': f'the lock file is {age:.0f}s old'}
                        return False
                    print(f"Removing an unreadable lock file at {self.path}")
                elif self._pid_alive(holder.get('pid')):
                    self.holder = holder
                    return False
                else:
                    print(f"Taking over a stale lock left by PID {holder.get('pid')}")
                try:
                    os.remove(self.path)
                except OSError:
                    pass
                continue  # retry the exclusive create; only one process wins

            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'pid': self.pid, 'started': time.time(),
                           'script': os.path.basename(__file__)}, f)
            self.acquired = True
            return True
        return False

    def release(self):
        if not self.acquired:
            return
        try:
            # Re-read first: if a takeover happened, this file belongs to
            # someone else now.
            holder = self._read_holder()
            if holder and int(holder.get('pid', -1)) == self.pid:
                os.remove(self.path)
        except Exception:
            pass
        self.acquired = False


def cosine_is_spent(scheduler):
    """True once a cosine schedule has reached eta_min and would start climbing.

    CosineAnnealingLR is a pure function of last_epoch: past T_max the cosine
    turns around and walks the learning rate back up towards its initial value,
    so a run extended beyond its annealing budget would spend the extra epochs
    at a *rising* lr - a warm restart by accident, and one that contradicts
    "raise num_epochs to keep training". Parking the schedule at eta_min is the
    intended reading of a spent budget.
    """
    return (isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)
            and scheduler.last_epoch >= scheduler.T_max)


def check_recent_writers(window=300):
    """Advisory-only concurrency check, for when the lock file is turned off."""
    now = time.time()
    for path in (config.model_save_path, config.final_model_save_path,
                 config.resume_checkpoint_path):
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        if age < window:
            print(f"WARNING: {path} was written {age:.0f}s ago - another run of "
                  f"this script may be active against the same files.")


# ==================== Training Functions ====================
def train_noise2noise():
    """Train Noise2Noise denoising model"""
    print("="*60)
    print("Training Noise2Noise Denoising Model")
    print("="*60)
    
    # The three subsets are physical folders, partitioned once by sample
    roots = {'train': config.train_root,
             'val': config.val_root,
             'test': config.test_root}
    for name, root in roots.items():
        if not os.path.isdir(root):
            print(f"Error: {name} directory {root} does not exist!")
            return None, None, None, None, None

    n_folders = {name: len([d for d in os.listdir(root)
                            if os.path.isdir(os.path.join(root, d))])
                 for name, root in roots.items()}
    if not all(n_folders.values()):
        print(f"Error: empty split in {n_folders}!")
        return None, None, None, None, None

    n_samples = sum(n_folders.values())
    print(f"Samples (folders): {n_folders['train']} training, "
          f"{n_folders['val']} validation, {n_folders['test']} test "
          f"-> {n_folders['train']/n_samples:.1%} : {n_folders['val']/n_samples:.1%} : "
          f"{n_folders['test']/n_samples:.1%}")

    # One dataset per subset, each drawing pairs only from its own samples.
    # Train and validation draw a fresh pair on every access, so `num_pairs` is
    # just the length of an epoch; the held-out set is enumerated exhaustively
    # instead, so the reported test loss carries no sampling variance.
    try:
        train_dataset = FullCurveDataset(config.train_root,
                                         num_pairs=config.pairs_per_epoch)
        # The training set keeps re-drawing every epoch on purpose. Validation
        # freezes its subset so consecutive epochs are comparable at all.
        val_dataset = FullCurveDataset(config.val_root,
                                       num_pairs=config.pairs_per_epoch,
                                       fixed_pairs=config.val_pairs_fixed,
                                       fixed_pairs_seed=config.val_pairs_seed)
        test_dataset = FullCurveDataset(config.test_root, exhaustive=True)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None, None, None, None

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: one of the three splits contains no pairs!")
        return None, None, None, None, None

    # The noise-pair counts in one place. The per-dataset lines above say how
    # many frames each split loaded but not which split they belong to, and they
    # never say how large the pool behind the per-epoch draws is.
    print()
    print("="*62)
    print("Noise-pair counts")
    print("-"*62)
    print(f"{'split':<8}{'samples':>9}{'frames':>9}{'pairs/epoch':>14}"
          f"{'total pairs':>15}")
    for name, ds in (('train', train_dataset),
                     ('val', val_dataset),
                     ('test', test_dataset)):
        per_epoch = 'n/a' if ds.exhaustive else f"{len(ds):,}"
        print(f"{name:<8}{len(ds.valid_folders):>9,}{ds.n_frames:>9,}"
              f"{per_epoch:>14}{ds.total_pairs:>15,}")
    print("-"*62)
    print("train/val draw a fresh pair on every access (both directions), so")
    print("'pairs/epoch' is one epoch's length and 'total pairs' is the whole")
    print("pool those draws come from. test is enumerated exhaustively in a")
    print("single pass, so there 'total pairs' is exactly the number of")
    print("forward passes behind the reported test loss.")
    print("="*62)
    print()
    
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
    model = build_model(in_channels=1)
    model = model.to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture Summary:")
    print("-"*40)
    print(f"Architecture: {getattr(model, 'ARCH_TAG', '?')}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Parameter breakdown: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    
    # L1 Loss only (as requested)
    criterion = nn.L1Loss()

    # The identity baseline is a property of the frozen pair set, not of the
    # model, so it is measured once. Every epoch's validation loss can then be
    # read as "x% of the way from doing nothing to the best a denoiser can do",
    # which is the only comparison that survives a change of split or of pair
    # weighting. The raw numbers do not survive it: the validation pool's
    # identity loss is 0.0122 while the test pool's is 0.0068, so a val loss of
    # 0.0082 and a test loss of 0.0057 are not on one scale at all.
    val_identity = None
    if val_dataset.mode == 'fixed':
        with torch.no_grad():
            _ident = 0.0
            for noisy1, noisy2 in val_loader:
                _ident += criterion(noisy1, noisy2).item()
        val_identity = _ident / max(len(val_loader), 1)
        print(f"Identity baseline on the frozen validation subset: "
              f"{val_identity:.6f} (L1(x1, x2), no model at all)")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    if config.lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_cosine_t_max,
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
    os.makedirs(os.path.dirname(config.resume_checkpoint_path), exist_ok=True)

    # ==================== Resume ====================
    start_epoch = 0       # index of the first epoch to run
    last_epoch = -1       # index of the last epoch actually run
    best_epoch = 0        # 1-indexed epoch that produced the best weights
    rng_to_restore = None
    stopped_early = False

    ckpt = load_resume_checkpoint(config.resume_checkpoint_path) if config.resume else None

    # Tested against the budget, not against the 'completed' flag: a run that
    # stopped early at epoch 113 of 500 still has 387 epochs to do, and
    # raising num_epochs past a finished budget is how a run gets extended.
    if ckpt is not None and int(ckpt['epochs_completed']) >= config.num_epochs:
        print("="*60)
        print(f"WARNING: {config.resume_checkpoint_path}")
        print(f"         already covers {ckpt['epochs_completed']} epochs "
              f"(last exit_reason={ckpt['exit_reason']!r}) and num_epochs is "
              f"{config.num_epochs} - nothing is left to train.")
        if not config.resume_restart_if_completed:
            print("  Pick one: (a) raise num_epochs to extend the run, "
                  "(b) delete that checkpoint, or (c) set resume=False.")
            raise SystemExit(0)
        print("  resume_restart_if_completed=True -> backing up the current "
              "models and starting over:")
        backup_legacy_models()
        ckpt = None

    if ckpt is not None:
        validate_resume_config(ckpt.get('config') or {})

        # Checked before the weight comparison because that comparison cannot
        # see every architecture change: GroupNorm(8, 256) and GroupNorm(256, 256)
        # produce identical state dicts, and a removed skip or a reordered concat
        # changes no weight at all.
        current_tag = getattr(model, 'ARCH_TAG', None)
        if ckpt.get('arch_tag') != current_tag:
            raise SystemExit(
                f"ERROR: this checkpoint was written for model "
                f"{ckpt.get('arch_tag')!r}, this script defines {current_tag!r}.\n"
                f"       Move {config.resume_checkpoint_path} aside to start the "
                f"new architecture from scratch.")

        # load_state_dict's own error message is opaque; check up front so the
        # advice (delete the file) comes with the failure.
        current_sd = model.state_dict()
        saved_sd = ckpt['model_state_dict']
        mismatched = [k for k in current_sd
                      if k not in saved_sd or current_sd[k].shape != saved_sd[k].shape]
        if set(saved_sd) - set(current_sd) or mismatched:
            raise SystemExit(
                "ERROR: the model architecture does not match this checkpoint, "
                "so it cannot be resumed.\n"
                f"       Delete {config.resume_checkpoint_path} to start from scratch.")

        start_epoch = int(ckpt['epochs_completed'])  # 1-indexed == next index
        model.load_state_dict(saved_sd)
        # The best weights exist only here. Without them a restart would
        # evaluate and re-save whatever the last epoch happened to hold.
        best_model_state = ckpt['best_model_state_dict']
        best_val_loss = float(ckpt['best_val_loss'])
        best_epoch = int(ckpt.get('best_epoch', 0))
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception as e:
            print(f"WARNING: could not restore the LR scheduler state ({e}); "
                  f"using a freshly built scheduler")

        # CosineAnnealingLR keeps T_max inside its state dict and
        # load_state_dict copies it wholesale, so the schedule would stay aimed
        # at whatever budget it was written with. Re-aim it at the current one.
        if config.lr_scheduler_type == 'cosine':
            old_tmax = getattr(scheduler, 'T_max', None)
            new_tmax = config.lr_cosine_t_max
            if old_tmax is not None and old_tmax != new_tmax:
                print(f"WARNING: cosine T_max {old_tmax} -> {new_tmax} "
                      f"(re-aimed at the new budget)")
            scheduler.T_max = new_tmax
            scheduler.eta_min = config.lr_min
            if cosine_is_spent(scheduler):
                print(f"NOTE: the schedule has already reached its minimum "
                      f"(last_epoch={scheduler.last_epoch} >= T_max={new_tmax}); "
                      f"the remaining epochs run at lr={config.lr_min:.1e}.")

        early_stopping.best_loss = ckpt['early_stop']['best_loss']
        early_stopping.counter = ckpt['early_stop']['counter']
        early_stopping.early_stop = False   # never restore "already stopped"
        early_stopping.patience = config.early_stop_patience
        early_stopping.min_delta = config.early_stop_min_delta
        if config.reset_early_stop_on_resume:
            print(f"reset_early_stop_on_resume: patience counter "
                  f"{early_stopping.counter} -> 0 "
                  f"(best_loss stays at {early_stopping.best_loss:.6f})")
            early_stopping.counter = 0

        train_losses = list(ckpt['train_losses'])
        val_losses = list(ckpt['val_losses'])
        rng_to_restore = ckpt['rng']

        print("="*60)
        print(f"Resuming from {config.resume_checkpoint_path}")
        print(f"  epoch         : {start_epoch + 1}/{config.num_epochs}")
        print(f"  best_val_loss : {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"  early stop    : counter {early_stopping.counter}"
              f"/{early_stopping.patience}")
        print(f"  LR            : {optimizer.param_groups[0]['lr']:.3e}")
        print(f"  last exit     : {ckpt['exit_reason']!r}, "
              f"{len(train_losses)} epochs of history restored")
        if (not config.reset_early_stop_on_resume
                and early_stopping.counter >= early_stopping.patience):
            print("  WARNING: the restored patience counter is already exhausted, "
                  "so this run stops at the end of its first epoch unless the "
                  "validation loss improves. Set reset_early_stop_on_resume=True "
                  "or raise early_stop_patience to keep training.")
    elif config.resume and config.bootstrap_from_legacy_best:
        warm = bootstrap_from_legacy(config.bootstrap_source_path, model)
        if warm is not None:
            best_val_loss = warm['best_val_loss']
            best_model_state = warm['best_model_state']
            best_epoch = warm['best_epoch']
            start_epoch = warm['epochs_completed']
            early_stopping.best_loss = best_val_loss
            early_stopping.counter = 0
            print("="*60)
            print(f"WARM START from {warm['source']}")
            print(f"  weights and best_val_loss ({best_val_loss:.6f}) adopted "
                  f"(source recorded epoch={warm.get('source_epoch')!r}, "
                  f"provenance only).")
            print("  Adam moments, the LR schedule, the patience counter and the "
                  "RNG are all new, so this is a warm restart rather than a "
                  "continuation.")
            print(f"  Training {config.num_epochs} epochs from epoch 1, LR "
                  f"starting at {config.learning_rate:.3e}.")

    # A start at or past the budget would run an empty loop, and the run would
    # then report itself as finished without having trained anything. Both
    # branches above are meant to prevent that; this makes it impossible to
    # miss, since a silent no-op wastes a whole scheduling cycle.
    if start_epoch >= config.num_epochs:
        raise SystemExit(f"ERROR: nothing to train - start_epoch={start_epoch} "
                         f">= num_epochs={config.num_epochs}.")

    print("\nStarting training...")
    print("-"*60)

    # Last thing before the loop, so that nothing consumes the torch RNG
    # between the restore and the first shuffle.
    if rng_to_restore is not None:
        restore_rng_state(rng_to_restore)
    
    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        last_epoch = epoch
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
        
        # Update learning rate scheduler. A spent cosine is left where it is
        # rather than stepped past T_max, where it would climb again.
        if config.lr_scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        elif not cosine_is_spent(scheduler):
            scheduler.step()
        
        # Record loss
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Clear line and print progress in place
        gain = (f' | Gain vs ident: '
                f'{(val_identity - avg_val_loss) / val_identity * 100:5.2f}%'
                if val_identity else '')
        print(f'\rEpoch {epoch+1:3d}/{config.num_epochs} | '
              f'Time: {epoch_time:5.1f}s | '
              f'Train Loss: {avg_train_loss:8.6f} | '
              f'Val Loss: {avg_val_loss:8.6f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}{gain}', end='')
        
        # Save best model
        if avg_val_loss < best_val_loss and val_batches > 0:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # deepcopy, not .copy(): state_dict() hands out detached views that
            # share storage with the live parameters, so a shallow copy would be
            # mutated in place by every later optimizer.step() and would end up
            # holding the LAST epoch instead of the best one.
            best_model_state = copy.deepcopy(model.state_dict())

            atomic_torch_save({
                'epoch': epoch,
                'arch_tag': getattr(model, 'ARCH_TAG', None),
                'model_state_dict': best_model_state,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': snapshot_config()
            }, config.model_save_path)

            print(f"\n✅ Saved best model, validation loss: {avg_val_loss:.6f}")

        # Early Stopping check. Guarded on val_batches for the same reason the
        # best-model branch above is: avg_val_loss is 0 when there is nothing
        # to validate, and a zero reads as a huge improvement.
        if val_batches > 0 and early_stopping(avg_val_loss):
            stopped_early = True
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break

        # Everything above is one consistent epoch boundary, which is why the
        # checkpoint is written here and nowhere else.
        if config.resume and (epoch + 1) % config.save_resume_every_n_epochs == 0:
            save_resume_checkpoint(config.resume_checkpoint_path, epoch + 1, False, None,
                                   model, optimizer, scheduler, early_stopping,
                                   best_val_loss, best_epoch, best_model_state,
                                   train_losses, val_losses)
    
    print()  # New line after training completes
    print("-"*60)
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Mark the run finished before the test pass, which runs one forward pass
    # per held-out pair and takes far longer than a single epoch. Interrupting
    # that must not lose the training state.
    if config.resume:
        save_resume_checkpoint(config.resume_checkpoint_path, last_epoch + 1, True,
                               'early_stop' if stopped_early else 'budget',
                               model, optimizer, scheduler, early_stopping,
                               best_val_loss, best_epoch, best_model_state,
                               train_losses, val_losses)

    # Restore the weights selected on the validation set, so that what is returned
    # and evaluated below is the selected model and not the last epoch
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save final model
    if best_model_state is not None:
        atomic_torch_save({
            'epoch': config.num_epochs,
            'arch_tag': getattr(model, 'ARCH_TAG', None),
            'model_state_dict': best_model_state,
            'train_loss': train_losses[-1] if train_losses else 0,
            'val_loss': best_val_loss,
            'config': snapshot_config()
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
    print(f"Pairs per epoch: "
          f"{config.pairs_per_epoch if config.pairs_per_epoch is not None else 'auto (one per training frame)'}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size} (for variable length)")
    print(f"Epochs: {config.num_epochs}")
    if config.lr_scheduler_type == "cosine":
        print(f"LR schedule: cosine over {config.lr_cosine_t_max} epochs "
              f"to {config.lr_min:.1e}")
    print(f"Validation pairs: "
          + (f"{config.val_pairs_fixed} frozen (seed {config.val_pairs_seed})"
             if config.val_pairs_fixed else "re-drawn every epoch"))
    print(f"Model architecture: {getattr(MODEL_CLASSES[config.arch], 'ARCH_TAG', '?')}")
    if config.device == 'cuda':
        print(f"Compute device: GPU - cuda:{torch.cuda.current_device()} "
              f"({torch.cuda.get_device_name(0)})")
    else:
        print("Compute device: CPU (no CUDA device available)")
    print(f"Designed split (train:val:test): {config.train_ratio:.2f} : {config.val_ratio:.2f} : {config.test_ratio:.2f}")
    print(f"Clip negative values: {config.clip_negative}")
    print(f"Use log1p transform: {config.use_log1p}")
    print(f"Loss function: L1 Loss only")
    print(f"Resume: {'on' if config.resume else 'off'}"
          + (f" ({config.resume_checkpoint_path})" if config.resume else ""))
    print("-"*30)

    # ---- Single-instance guard ----
    # Two copies of this script write the same model files, and the last one to
    # finish silently overwrites the other's results.
    if config.use_lock_file:
        lock = ProcessLock(config.lock_file_path)
        if not lock.acquire():
            holder = lock.holder or {}
            note = f", {holder['note']}" if holder.get('note') else ""
            print(f"Error: another run of this script holds {config.lock_file_path} "
                  f"(PID {holder.get('pid')}{note}).")
            print("       Wait for it to finish. If it is already gone, delete that "
                  "file and start again.")
            return
        # atexit rather than try/finally, so the training function stays flat.
        atexit.register(lock.release)
        print(f"Lock acquired: {config.lock_file_path} (PID {lock.pid})")
    else:
        check_recent_writers()

    # Ctrl+C costs at most the epoch in flight, since every completed epoch is
    # already on disk. This only replaces the traceback with a clear line.
    def _on_sigint(signum, frame):
        print(f"\nInterrupted. The completed epochs are in "
              f"{config.resume_checkpoint_path}; run this script again to "
              f"continue from there.")
        raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except ValueError:
        pass  # not the main thread; the default handler still applies

    # Check the three dataset folders
    for name, root in (("train", config.train_root),
                       ("val", config.val_root),
                       ("test", config.test_root)):
        if not os.path.isdir(root):
            print(f"Error: {name} directory {root} does not exist!")
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
    print(f"Resume state saved to: {config.resume_checkpoint_path}")
    print(f"Best validation loss: {result[3]:.6f}")
    print(f"Test loss: {result[4]:.6f}")
    print("="*60)
    print("To train further, raise config.num_epochs and run again - the run "
          "continues from the resume state.")

# ==================== Run Main Program ====================
if __name__ == "__main__":
    main()