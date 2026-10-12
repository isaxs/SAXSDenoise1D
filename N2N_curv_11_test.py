# test.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 添加这一行

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import chardet
warnings.filterwarnings('ignore')

# Set matplotlib to use better style with larger fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
# 移除seaborn样式，使用简单白色背景
plt.style.use('default')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# ==================== Global Configuration ====================
class Config:
    # Data related parameters
    data_root = "./Data/"  # Data root目录
    data_suffix = ".dat"   # Data file suffix
    file_encoding = None   # File encoding, if None auto-detect
    
    # Model related parameters
    # What N2N_curv_11_train.py writes. The older ./models/denoise_1d.pth was
    # produced by the v08/v09 line, on the random split those versions used at
    # run time rather than the Data/train|val|test partition in use now, and it
    # is the weakest of the models on disk: on the current exhaustive test set
    # it reaches a 14.65% gain over doing nothing, against 16.07% for the v10
    # U-Net and 25.6% for the v11 network.
    model_path = "./models/denoiseN2N_1d_v11.pth"  # Trained model path

    # Which network the trainer wrote is recorded in the checkpoint's arch_tag;
    # this file asks the trainer for the class rather than assuming one.
    trainer_script = "N2N_curv_11_train.py"
    
    # Data preprocessing parameters
    clip_negative = True     # Clip negative values to 0
    use_log1p = True         # Apply log1p transformation
    
    # System parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    random_seed = 42       # Random seed
    
# Initialize configuration
config = Config()
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

# ==================== Data Preprocessing Functions ====================
def preprocess_data(data):
    """Preprocess data: clip negative values and apply log1p"""
    data = data.astype(np.float32)
    
    # 1. Clip negative values to 0
    if config.clip_negative:
        data = np.clip(data, 0, None)  # Use np.clip for consistency
    
    # 2. Apply log1p transformation if enabled
    if config.use_log1p:
        data = np.log1p(data + 1e-8)
    
    return data

def inverse_preprocess(data):
    """Inverse preprocessing: inverse log1p and ensure non-negative"""
    if config.use_log1p:
        # Inverse log1p: exp(x) - 1
        data = np.expm1(data)
    
    # Ensure non-negative
    if config.clip_negative:
        data = np.maximum(data, 0)
    
    return data

# ==================== Network (imported from the trainer) ====================
# Imported, not copied. Every *_test.py in this repo used to carry its own copy of
# the network, and this file is what that costs: by the time v11 moved the
# trainer to a full-resolution dilated network, this file still held the
# BatchNorm U-Net from three versions earlier, so the two could only drift
# further apart. One definition, one place to change.
def _load_trainer():
    """Import the training script for its network classes and its config.

    Safe to execute: the trainer's entry point sits behind an
    `if __name__ == "__main__"` guard, so importing it only defines things.
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, config.trainer_script)
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: {config.trainer_script} was not found next to this file.\n"
            f"       The network definition lives there; this script imports it "
            f"instead of keeping a second copy that can drift out of step.")
    spec = importlib.util.spec_from_file_location("n2n_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()
_MODELS_BY_TAG = {cls.ARCH_TAG: cls for cls in trainer.MODEL_CLASSES.values()}


def build_model_from_checkpoint(checkpoint, in_channels=1):
    """Instantiate whichever network the checkpoint was written for.

    The trainer stamps every checkpoint with its architecture, so the file
    itself says how to load it. The weights alone would not be enough to tell:
    GroupNorm(8, C) and GroupNorm(C, C) produce identical state dicts, and a
    changed forward path (a removed skip, a reordered concat) changes no weight
    at all.
    """
    tag = checkpoint.get('arch_tag')
    if tag in _MODELS_BY_TAG:
        return _MODELS_BY_TAG[tag](in_channels=in_channels)

    # Checkpoints written before the tag existed record nothing about their
    # network, so fall back to matching on weight names.
    saved = set(checkpoint.get('model_state_dict') or {})
    if saved:
        for cls in _MODELS_BY_TAG.values():
            if set(cls(in_channels=in_channels).state_dict()) == saved:
                print(f"Note: {config.model_path} carries no arch_tag; matched it "
                      f"to {cls.ARCH_TAG} by its weight names.")
                return cls(in_channels=in_channels)

    raise SystemExit(
        f"ERROR: {config.model_path} has arch_tag={tag!r}, which matches none of "
        f"{sorted(_MODELS_BY_TAG)}, and its weight names match no network the "
        f"trainer defines either.\n"
        f"       Point config.model_path at a checkpoint N2N_curv_11_train.py "
        f"wrote. Weights from the older generations (./models/denoise_1d.pth and "
        f"its *_final sibling) belong to a different, hard-coded network and "
        f"cannot be loaded here.")
# ==================== File Reading Functions ====================
def detect_encoding(filepath):
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

def read_curve_file(filepath):
    """Read curve file, extract x (1st column) and y (2nd column) data"""
    # Detect encoding
    detected_encoding = detect_encoding(filepath)
    encodings_to_try = [detected_encoding, 'utf-8', 'gbk']
    
    for encoding in encodings_to_try:
        try:
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                lines = f.readlines()
            
            # Find where data starts (skip header)
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
            
            # Extract x (1st column) and y (2nd column) data
            x_data = []
            y_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith(('#', '//', '%')):
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        x_data.append(x_val)
                        y_data.append(y_val)
                except:
                    continue
            
            if len(x_data) == 0 or len(y_data) == 0:
                # Try alternative separators
                x_data, y_data = parse_alternative_format(lines)
            
            if len(x_data) == 0 or len(y_data) == 0:
                print(f"Warning: No valid data found in {filepath}")
                # Create dummy data
                x_data = np.arange(100).astype(np.float32)
                y_data = np.random.randn(100).astype(np.float32)
            
            # Apply preprocessing to y_data
            y_data = np.array(y_data, dtype=np.float32)
            y_data = preprocess_data(y_data)
            
            return np.array(x_data, dtype=np.float32), y_data
            
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, return dummy data
    print(f"Error: Cannot read file {filepath}")
    x_data = np.arange(100).astype(np.float32)
    y_data = preprocess_data(np.random.randn(100).astype(np.float32))
    return x_data, y_data

def parse_alternative_format(lines):
    """Alternative parsing method for different file formats"""
    x_data = []
    y_data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Try multiple separators
        for sep in ['\t', ' ', ',', ';', '|']:
            parts = line.split(sep)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 2:
                try:
                    # Assume first two columns are x and y
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    x_data.append(x_val)
                    y_data.append(y_val)
                    break
                except ValueError:
                    continue
    
    return x_data, y_data

# ==================== Denoising Functions ====================
def denoise_full_curve(model, y_data, device='cpu'):
    """
    Denoise full curve (no sliding window needed)
    Args:
        model: Trained denoising model
        y_data: Full y-coordinate data to denoise (already preprocessed)
        device: Device to run model on
    Returns:
        denoised_y: Denoised full y-coordinate data (preprocessed space)
    """
    n_samples = len(y_data)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(y_data).unsqueeze(0).unsqueeze(0).to(device)
    
    # Denoise
    with torch.no_grad():
        denoised_tensor = model(input_tensor)
        denoised_y = denoised_tensor.cpu().numpy().squeeze()
    
    print(f"Denoised {n_samples} samples")
    return denoised_y

# ==================== Evaluation Metrics Functions ====================
def calculate_metrics(original, denoised):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # 一阶差分范数计算
    original_first_diff = np.diff(original, 1)
    denoised_first_diff = np.diff(denoised, 1)
    
    # 计算一阶差分L2范数（欧几里得范数）
    metrics['first_diff_norm_original'] = np.linalg.norm(original_first_diff)
    metrics['first_diff_norm_denoised'] = np.linalg.norm(denoised_first_diff)
    
    # 计算一阶差分相对变化
    if metrics['first_diff_norm_original'] > 0:
        metrics['first_diff_reduction'] = (metrics['first_diff_norm_original'] - metrics['first_diff_norm_denoised']) / metrics['first_diff_norm_original'] * 100
    else:
        metrics['first_diff_reduction'] = 0.0
    
    # 计算一阶差分L1范数（绝对值和）
    metrics['first_diff_l1_original'] = np.sum(np.abs(original_first_diff))
    metrics['first_diff_l1_denoised'] = np.sum(np.abs(denoised_first_diff))
    
    # 一阶差分L1范数相对变化
    if metrics['first_diff_l1_original'] > 0:
        metrics['first_diff_l1_reduction'] = (metrics['first_diff_l1_original'] - metrics['first_diff_l1_denoised']) / metrics['first_diff_l1_original'] * 100
    else:
        metrics['first_diff_l1_reduction'] = 0.0
    
    # 二阶差分范数计算
    original_second_diff = np.diff(original, 2)
    denoised_second_diff = np.diff(denoised, 2)
    
    # 计算二阶差分L2范数（欧几里得范数）
    metrics['second_diff_norm_original'] = np.linalg.norm(original_second_diff)
    metrics['second_diff_norm_denoised'] = np.linalg.norm(denoised_second_diff)
    
    # 计算二阶差分相对变化
    if metrics['second_diff_norm_original'] > 0:
        metrics['second_diff_reduction'] = (metrics['second_diff_norm_original'] - metrics['second_diff_norm_denoised']) / metrics['second_diff_norm_original'] * 100
    else:
        metrics['second_diff_reduction'] = 0.0
    
    # 平滑度比率（原始/去噪）
    if metrics['second_diff_norm_denoised'] > 0:
        metrics['smoothness_ratio'] = metrics['second_diff_norm_original'] / metrics['second_diff_norm_denoised']
    else:
        metrics['smoothness_ratio'] = float('inf')
    
    # 计算二阶差分L1范数（绝对值和）
    metrics['second_diff_l1_original'] = np.sum(np.abs(original_second_diff))
    metrics['second_diff_l1_denoised'] = np.sum(np.abs(denoised_second_diff))
    
    # 二阶差分L1范数相对变化
    if metrics['second_diff_l1_original'] > 0:
        metrics['second_diff_l1_reduction'] = (metrics['second_diff_l1_original'] - metrics['second_diff_l1_denoised']) / metrics['second_diff_l1_original'] * 100
    else:
        metrics['second_diff_l1_reduction'] = 0.0
    
    # 计算R值：二阶差分减少率/一阶差分减少率
    if metrics['first_diff_reduction'] > 0 and metrics['second_diff_reduction'] > 0:
        # 计算减少率（百分比转换为小数）
        first_reduction_rate = metrics['first_diff_reduction'] / 100
        second_reduction_rate = metrics['second_diff_reduction'] / 100
        metrics['R_value'] = second_reduction_rate / first_reduction_rate
    else:
        metrics['R_value'] = float('inf')
    
    # 计算平衡比 R = (二阶差分范数减少率) / (一阶差分范数减少率)
    if metrics['first_diff_norm_original'] > 0 and metrics['second_diff_norm_original'] > 0:
        first_reduction_ratio = 1 - (metrics['first_diff_norm_denoised'] / metrics['first_diff_norm_original'])
        second_reduction_ratio = 1 - (metrics['second_diff_norm_denoised'] / metrics['second_diff_norm_original'])
        
        if first_reduction_ratio > 0:
            metrics['balance_ratio'] = second_reduction_ratio / first_reduction_ratio
        else:
            metrics['balance_ratio'] = float('inf')
    else:
        metrics['balance_ratio'] = float('inf')
    
    return metrics

def calculate_first_difference_norms(data):
    """计算一阶差分的各种范数"""
    first_diff = np.diff(data, 1)
    
    norms = {}
    norms['l2'] = np.linalg.norm(first_diff)  # L2范数（欧几里得范数）
    norms['l1'] = np.sum(np.abs(first_diff))  # L1范数（绝对值和）
    norms['max'] = np.max(np.abs(first_diff))  # L∞范数（最大值）
    norms['rms'] = np.sqrt(np.mean(first_diff**2))  # RMS值
    norms['mean_abs'] = np.mean(np.abs(first_diff))  # 平均绝对值
    
    return norms

def calculate_second_difference_norms(data):
    """计算二阶差分的各种范数"""
    second_diff = np.diff(data, 2)
    
    norms = {}
    norms['l2'] = np.linalg.norm(second_diff)  # L2范数（欧几里得范数）
    norms['l1'] = np.sum(np.abs(second_diff))  # L1范数（绝对值和）
    norms['max'] = np.max(np.abs(second_diff))  # L∞范数（最大值）
    norms['rms'] = np.sqrt(np.mean(second_diff**2))  # RMS值
    norms['mean_abs'] = np.mean(np.abs(second_diff))  # 平均绝对值
    
    return norms

def calculate_residual_autocorrelation(residuals, max_lag=None):
    """Calculate autocorrelation of residuals"""
    n = len(residuals)
    if max_lag is None:
        max_lag = min(n // 4, 100)
    else:
        max_lag = min(max_lag, n-1)
    
    # 中心化残差（去中心平均）
    mean_res = np.mean(residuals)
    residuals_centered = residuals - mean_res
    sum_sq_residuals = np.sum(residuals_centered ** 2)
    
    # 如果平方和为0（全零残差）
    if sum_sq_residuals == 0:
        return list(range(max_lag + 1)), [1.0] + [0.0] * max_lag
    
    autocorr = []
    lags = list(range(max_lag + 1))
    
    for lag in lags:
        if lag == 0:
            autocorr.append(1.0)
        else:
            numerator = np.sum(residuals_centered[lag:] * residuals_centered[:n-lag])
            autocorr.append(numerator / sum_sq_residuals)
    
    return lags, autocorr

# ==================== GUI Application ====================
class DenoiseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("1D Curve Noise2Noise Denoising Test Tool")
        self.root.geometry("1500x1000")  # 进一步增加窗口大小以适应更大的字体
        
        # Initialize variables
        self.model = None
        self.current_file = None
        self.x_data = None  # x-coordinates (1st column)
        self.original_y = None  # Original y-data (2nd column) - in preprocessed space
        self.original_y_raw = None  # Original y-data before preprocessing
        self.denoised_y = None  # Denoised y-data (in preprocessed space)
        self.denoised_results = []  # Store multiple denoising results
        self.denoise_times = 1  # Default denoising times
        self.first_diff_norms = []  # 存储每次去噪的一阶差分范数
        self.second_diff_norms = []  # 存储每次去噪的二阶差分范数
        
        # Load model
        self.load_model()
        
        # Create GUI interface
        self.create_widgets()
        
    def load_model(self):
        """Load denoising model"""
        try:
            if not os.path.exists(config.model_path):
                messagebox.showerror("Error", f"Model file does not exist: {config.model_path}")
                return
            
            print(f"Loading model: {config.model_path}")
            # weights_only=False: these files carry a config dict alongside the
            # weights, which the safe unpickler refuses.
            checkpoint = torch.load(config.model_path, map_location=config.device,
                                    weights_only=False)

            # The checkpoint says which network it was written for, so build
            # that one rather than assuming.
            self.model = build_model_from_checkpoint(checkpoint, in_channels=1)
            self.model = self.model.to(config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            print("✅ Model loaded successfully")
            print(f"Architecture: {getattr(self.model, 'ARCH_TAG', '?')}")
            print(f"Device: {config.device}")
            print(f"Data preprocessing: clip_negative={config.clip_negative}, use_log1p={config.use_log1p}")
            for key in ('val_loss', 'train_loss', 'epoch'):
                if key in checkpoint:
                    print(f"Checkpoint {key}: {checkpoint[key]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            print(f"Failed to load model: {e}")
    
    def create_widgets(self):
        """Create GUI controls"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights - 让图表区域占据更多空间
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)   # 控制按钮区域
        main_frame.columnconfigure(1, weight=5)   # 图表区域 - 增加权重
        main_frame.rowconfigure(1, weight=5)      # 图表区域权重增加
        main_frame.rowconfigure(3, weight=1)      # 信息显示区域权重减小
        
        # Title with larger font
        title_label = ttk.Label(main_frame, text="1D Curve Noise2Noise Denoising Test Tool", 
                               font=("Arial", 18, "bold"))  # 从16加大到18
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        # 左侧控制区域
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 设置控制区域固定宽度
        control_frame.columnconfigure(0, weight=1)
        
        # 创建按钮，使用较大的字体
        button_font = ("Arial", 11)  # 增加按钮字体大小
        button_width = 15
        
        ttk.Button(control_frame, text="Select File", command=self.select_file, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="Load Curve", command=self.load_curve, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="Denoise", command=self.denoise_curve, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="Save Plot as SVG", command=self.save_plot_svg, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 新增: 保存数据按钮
        ttk.Button(control_frame, text="Save Data", command=self.save_data, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=4, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(control_frame, text="Clear Display", command=self.clear_display, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=5, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Button(control_frame, text="Exit", command=self.root.quit, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=6, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 创建按钮样式
        style = ttk.Style()
        style.configure("LargeFont.TButton", font=button_font)
        
        # Denoise times control with larger font
        ttk.Label(control_frame, text="Denoise Times:", font=("Arial", 11)).grid(
            row=7, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.times_var = tk.StringVar(value="1")
        self.times_entry = ttk.Entry(control_frame, textvariable=self.times_var, 
                                    width=button_width, font=("Arial", 11))
        self.times_entry.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(control_frame, text="Set Times", command=self.set_denoise_times, 
                  width=button_width, style="LargeFont.TButton").grid(
            row=9, column=0, padx=5, pady=5, sticky=tk.W)
        
        # 一阶差分范数显示区域
        ttk.Label(control_frame, text="First Difference Norms:", font=("Arial", 11, "bold"), 
                 foreground="darkblue").grid(row=10, column=0, padx=5, pady=(10, 5), sticky=tk.W)
        
        self.first_diff_label = ttk.Label(control_frame, 
                                          text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                          font=("Arial", 11), foreground="darkblue")
        self.first_diff_label.grid(row=11, column=0, padx=5, pady=(0, 5), sticky=tk.W)
        
        # 二阶差分范数显示区域
        ttk.Label(control_frame, text="Second Difference Norms:", font=("Arial", 11, "bold"), 
                 foreground="purple").grid(row=12, column=0, padx=5, pady=(10, 5), sticky=tk.W)
        
        self.second_diff_label = ttk.Label(control_frame, 
                                          text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                          font=("Arial", 11), foreground="darkblue")
        self.second_diff_label.grid(row=13, column=0, padx=5, pady=(0, 5), sticky=tk.W)
        
        # 新增：R值显示区域
        ttk.Label(control_frame, text="Balance Ratio (R):", font=("Arial", 11, "bold"), 
                 foreground="darkred").grid(row=14, column=0, padx=5, pady=(10, 5), sticky=tk.W)
        
        self.r_value_label = ttk.Label(control_frame, 
                                       text="R = --\nInterpretation: --", 
                                       font=("Arial", 11), foreground="darkred")
        self.r_value_label.grid(row=15, column=0, padx=5, pady=(0, 10), sticky=tk.W)
        
        # Add preprocessing info label with larger font
        preprocessing_info = f"Preprocessing:\nclip_negative={config.clip_negative}\nuse_log1p={config.use_log1p}"
        ttk.Label(control_frame, text=preprocessing_info, foreground="blue", 
                 font=("Arial", 11)).grid(
            row=16, column=0, padx=5, pady=20, sticky=tk.W)
        
        # 控制区域的剩余空间填充
        control_frame.rowconfigure(17, weight=1)
        
        # Status information with larger font
        self.status_var = tk.StringVar(value="Ready - Select a file to begin")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, 
                                anchor=tk.W, font=("Arial", 11))
        status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                         pady=(10, 0), ipady=5)
        
        # 右侧图表区域 - 占据更多空间
        plot_frame = ttk.LabelFrame(main_frame, text="Curve Analysis", padding="10")
        plot_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # 设置matplotlib字体大小参数 - 使用整数字体大小
        plt.rcParams.update({
            'font.size': 20,           # 整数
            'axes.titlesize': 23,      # 整数
            'axes.labelsize': 20,      # 整数
            'xtick.labelsize': 17,     # 整数
            'ytick.labelsize': 17,     # 整数
            'legend.fontsize': 17,     # 整数
            'figure.titlesize': 26,    # 整数
        })
        
        # 创建matplotlib图形，调整高度比例，增加图表高度
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, 
            figsize=(14, 14),  # 增加总高度
            gridspec_kw={'height_ratios': [4, 3, 2]},  # 增加第一个图的高度比例
            constrained_layout=True  # 使用constrained_layout自动管理间距
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize charts
        self.initialize_plots()
        
        # Metrics display area - 移动到下方，高度减少50%
        metrics_frame = ttk.LabelFrame(main_frame, text="Evaluation Metrics", padding="10")
        metrics_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(10, 0))
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)
        
        # Create scrolled text for metrics display with larger font，高度减少50%
        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, width=80, height=6,  # 从12改为6
                                                     font=("Courier", 17))  # 使用整数字体大小
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.metrics_text.insert(tk.END, "Metrics will appear here after denoising...")
        self.metrics_text.config(state=tk.DISABLED)  # Make read-only
        
        # Configure grid weights for the main frame
        main_frame.rowconfigure(1, weight=5)  # 图表区域权重更大
        main_frame.rowconfigure(3, weight=1)  # 信息显示区域权重较小
    
    def save_data(self):
        """Save data with difference norms in header"""
        if self.x_data is None or self.original_y_raw is None:
            messagebox.showwarning("Warning", "No data loaded. Please load a curve first.")
            return
        
        if not self.denoised_results:
            messagebox.showwarning("Warning", "No denoised data available. Please run denoising first.")
            return
        
        # 弹出保存文件对话框
        file_path = filedialog.asksaveasfilename(
            title="Save Data with Difference Norms",
            initialdir=".",
            defaultextension=".dat",
            filetypes=[("Data files", "*.dat"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return  # 用户取消保存
        
        try:
            # 获取最后一次去噪的结果
            last_result = self.denoised_results[-1]
            denoised_y_raw = last_result['raw']
            
            # 准备header信息
            header_lines = []
            header_lines.append("# Difference Norm Analysis Results")
            header_lines.append(f"# File: {os.path.basename(self.current_file) if self.current_file else 'Unknown'}")
            header_lines.append(f"# Samples: {len(self.x_data)}")
            header_lines.append(f"# Denoise Iterations: {self.denoise_times}")
            
            # 添加差分范数信息
            if hasattr(self, 'current_metrics') and self.current_metrics:
                header_lines.append("# First Difference Norms:")
                header_lines.append(f"#   Original L2 Norm: {self.current_metrics.get('first_diff_norm_original', 0):.6f}")
                header_lines.append(f"#   Denoised L2 Norm: {self.current_metrics.get('first_diff_norm_denoised', 0):.6f}")
                header_lines.append(f"#   Reduction: {self.current_metrics.get('first_diff_reduction', 0):.2f}%")
                header_lines.append(f"#   Original L1 Norm: {self.current_metrics.get('first_diff_l1_original', 0):.6f}")
                header_lines.append(f"#   Denoised L1 Norm: {self.current_metrics.get('first_diff_l1_denoised', 0):.6f}")
                header_lines.append(f"#   L1 Reduction: {self.current_metrics.get('first_diff_l1_reduction', 0):.2f}%")
                
                header_lines.append("# Second Difference Norms:")
                header_lines.append(f"#   Original L2 Norm: {self.current_metrics.get('second_diff_norm_original', 0):.6f}")
                header_lines.append(f"#   Denoised L2 Norm: {self.current_metrics.get('second_diff_norm_denoised', 0):.6f}")
                header_lines.append(f"#   Reduction: {self.current_metrics.get('second_diff_reduction', 0):.2f}%")
                header_lines.append(f"#   Original L1 Norm: {self.current_metrics.get('second_diff_l1_original', 0):.6f}")
                header_lines.append(f"#   Denoised L1 Norm: {self.current_metrics.get('second_diff_l1_denoised', 0):.6f}")
                header_lines.append(f"#   L1 Reduction: {self.current_metrics.get('second_diff_l1_reduction', 0):.2f}%")
                
                # 添加R值信息
                if 'balance_ratio' in self.current_metrics:
                    r_value = self.current_metrics.get('balance_ratio', 0)
                    header_lines.append("# Balance Ratio (R):")
                    header_lines.append(f"#   R = {r_value:.4f}")
                    header_lines.append(f"#   Interpretation: {self.get_r_interpretation(r_value)}")
                
                # 添加解释
                reduction_1st = self.current_metrics.get('first_diff_reduction', 0)
                reduction_2nd = self.current_metrics.get('second_diff_reduction', 0)
                
                if reduction_2nd > 50:
                    header_lines.append("#   Interpretation: Excellent noise reduction (2nd diff reduction > 50%)")
                elif reduction_2nd > 30:
                    header_lines.append("#   Interpretation: Good noise reduction (2nd diff reduction > 30%)")
                elif reduction_2nd > 10:
                    header_lines.append("#   Interpretation: Moderate noise reduction (2nd diff reduction > 10%)")
                else:
                    header_lines.append("#   Interpretation: Limited noise reduction")
            
            header_lines.append("# Columns: x, original_y, denoised_y")
            header_lines.append("#")
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                # 写入header
                for line in header_lines:
                    f.write(line + "\n")
                
                # 写入数据
                for i in range(len(self.x_data)):
                    x_val = self.x_data[i]
                    y_orig = self.original_y_raw[i]
                    y_denoised = denoised_y_raw[i] if i < len(denoised_y_raw) else 0.0
                    f.write(f"{x_val:.8f}\t{y_orig:.8f}\t{y_denoised:.8f}\n")
            
            self.status_var.set(f"Data saved: {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Data saved successfully as:\n{file_path}")
            print(f"Data with difference norms saved: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            print(f"Failed to save data: {e}")
    
    def get_r_interpretation(self, r_value):
        """根据R值提供解释"""
        if r_value == float('inf'):
            return "Undefined!"
        elif r_value > 2.0:
            return "Excellent!"
        elif r_value > 1.5:
            return "Good!"
        elif r_value > 1.0:
            return "Moderate!"
        elif r_value > 0.5:
            return "Fair!"
        else:
            return "Poor!"
    
    def save_plot_svg(self):
        """Save the current plot as SVG vector format"""
        # 弹出保存文件对话框
        file_path = filedialog.asksaveasfilename(
            title="Save Plot as SVG",
            initialdir=".",
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 使用当前图形保存为SVG
                self.fig.savefig(file_path, format='svg', dpi=600, bbox_inches='tight')
                self.status_var.set(f"Plot saved as SVG: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Plot saved successfully as:\n{file_path}")
                print(f"Plot saved as SVG: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
                print(f"Failed to save plot: {e}")
    
    def set_denoise_times(self):
        """Set denoising times"""
        try:
            times = int(self.times_var.get())
            if times < 1:
                times = 1
            elif times > 10:
                times = 10
                messagebox.showinfo("Info", "Maximum denoise times is 10")
            
            self.denoise_times = times
            self.status_var.set(f"Denoise times set to: {times}")
            print(f"Denoise times set to: {times}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            self.times_var.set("1")
            self.denoise_times = 1
    
    def initialize_plots(self):
        """Initialize charts with larger fonts"""
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        
        # 设置白色背景
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('white')
        
        # First plot: Original vs Denoised with larger fonts
        self.ax1.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
        self.ax1.set_ylabel(r"$I$ ($\mathrm{a.u.}$)", fontsize=20)
        self.ax1.tick_params(axis='both', which='major', labelsize=17)
        self.ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
        
        # Second plot: Residuals with larger fonts
        self.ax2.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
        self.ax2.set_ylabel(r"$I_{\text{orig}} - I_{\text{deno}}$", fontsize=20)
        self.ax2.tick_params(axis='both', which='major', labelsize=17)
        self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
        
        # Third plot: Second Differences with larger fonts
        self.ax3.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)  # 修改这里：从"Index"改为与图1图2相同的q
        self.ax3.set_ylabel(r"$\Delta_y^2$", fontsize=20)
        self.ax3.tick_params(axis='both', which='major', labelsize=17)
        self.ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
        
        # 使用constrained_layout自动管理间距，不额外调整
        self.canvas.draw()
    
    def select_file(self):
        """Select file from disk"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            initialdir=config.data_root,
            filetypes=[("Data files", f"*{config.data_suffix}"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_file = file_path
            file_name = os.path.basename(file_path)
            self.status_var.set(f"Selected file: {file_name}")
    
    def load_curve(self):
        """Load curve data"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        try:
            self.status_var.set("Loading curve...")
            self.root.update()
            
            # Read curve data (x and y columns)
            self.x_data, self.original_y = read_curve_file(self.current_file)
            self.denoised_y = None
            self.denoised_results = []  # Clear previous denoising results
            self.first_diff_norms = []  # Clear previous first difference norms
            self.second_diff_norms = []  # Clear previous second difference norms
            
            # Store raw version for display (inverse transform for visualization)
            self.original_y_raw = inverse_preprocess(self.original_y.copy())
            
            # Update charts
            self.update_plots()
            
            # Update metrics display
            self.update_metrics_display()
            
            # Reset difference display
            self.first_diff_label.config(text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                        foreground="darkblue")
            self.second_diff_label.config(text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                         foreground="darkblue")
            self.r_value_label.config(text="R = --\nInterpretation: --", 
                                     foreground="darkred")
            
            file_name = os.path.basename(self.current_file)
            self.status_var.set(f"Curve loaded: {file_name} ({len(self.original_y)} samples)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load curve: {str(e)}")
            self.status_var.set("Failed to load curve")
            print(f"Failed to load curve: {e}")
    
    def denoise_curve(self):
        """Denoise curve using the trained model"""
        if self.original_y is None:
            messagebox.showwarning("Warning", "Please load curve first")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded, cannot perform denoising")
            return
        
        try:
            self.status_var.set("Denoising...")
            self.root.update()
            
            # Clear previous denoising results
            self.denoised_results = []
            self.first_diff_norms = []
            self.second_diff_norms = []
            
            # Perform multiple denoising iterations
            current_y = self.original_y.copy()
            
            for i in range(self.denoise_times):
                self.status_var.set(f"Denoising iteration {i+1}/{self.denoise_times}...")
                self.root.update()
                
                # Perform denoising (input is already preprocessed)
                denoised_y = denoise_full_curve(
                    self.model, 
                    current_y, 
                    device=config.device
                )
                
                # Store the result (both preprocessed and raw)
                denoised_y_raw = inverse_preprocess(denoised_y.copy())
                self.denoised_results.append({
                    'iteration': i+1,
                    'preprocessed': denoised_y.copy(),
                    'raw': denoised_y_raw
                })
                
                # 计算一阶差分范数
                first_norms = calculate_first_difference_norms(denoised_y)
                first_norms['iteration'] = i+1
                self.first_diff_norms.append(first_norms)
                
                # 计算二阶差分范数
                second_norms = calculate_second_difference_norms(denoised_y)
                second_norms['iteration'] = i+1
                self.second_diff_norms.append(second_norms)
                
                # Use the denoised result as input for next iteration
                current_y = denoised_y.copy()
            
            # Calculate evaluation metrics for the last iteration
            if self.denoised_results:
                last_result = self.denoised_results[-1]['preprocessed']
                metrics = calculate_metrics(self.original_y, last_result)
                self.current_metrics = metrics  # 保存当前指标供保存数据时使用
                
                # 计算原始一阶差分范数
                original_first_norms = calculate_first_difference_norms(self.original_y)
                last_first_norms = self.first_diff_norms[-1]
                
                # 计算原始二阶差分范数
                original_second_norms = calculate_second_difference_norms(self.original_y)
                last_second_norms = self.second_diff_norms[-1]
                
                # 一阶差分减少百分比
                reduction_1st_l2 = (original_first_norms['l2'] - last_first_norms['l2']) / original_first_norms['l2'] * 100
                
                # 二阶差分减少百分比
                reduction_2nd_l2 = (original_second_norms['l2'] - last_second_norms['l2']) / original_second_norms['l2'] * 100
                
                # 更新一阶差分范数显示
                first_diff_text = f"Original L2: {original_first_norms['l2']:.4f}\n"
                first_diff_text += f"Denoised L2: {last_first_norms['l2']:.4f}\n"
                first_diff_text += f"Reduction: {reduction_1st_l2:.2f}%\n"
                
                # 根据一阶差分减少百分比添加英文结论
                if reduction_1st_l2 > 50:
                    first_diff_text += "Excellent (reduction > 50%)"
                    first_color = "darkgreen"
                elif reduction_1st_l2 > 30:
                    first_diff_text += "Good (reduction > 30%)"
                    first_color = "green"
                elif reduction_1st_l2 > 10:
                    first_diff_text += "Moderate (reduction > 10%)"
                    first_color = "orange"
                else:
                    first_diff_text += "Limited (reduction < 10%)"
                    first_color = "red"
                
                self.first_diff_label.config(text=first_diff_text, foreground=first_color)
                
                # 更新二阶差分范数显示
                second_diff_text = f"Original L2: {original_second_norms['l2']:.4f}\n"
                second_diff_text += f"Denoised L2: {last_second_norms['l2']:.4f}\n"
                second_diff_text += f"Reduction: {reduction_2nd_l2:.2f}%\n"
                
                # 根据二阶差分减少百分比添加英文结论
                if reduction_2nd_l2 > 50:
                    second_diff_text += "Excellent (reduction > 50%)"
                    second_color = "darkgreen"
                elif reduction_2nd_l2 > 30:
                    second_diff_text += "Good (reduction > 30%)"
                    second_color = "green"
                elif reduction_2nd_l2 > 10:
                    second_diff_text += "Moderate (reduction > 10%)"
                    second_color = "orange"
                else:
                    second_diff_text += "Limited (reduction < 10%)"
                    second_color = "red"
                
                self.second_diff_label.config(text=second_diff_text, foreground=second_color)
                
                # 更新R值显示
                if 'balance_ratio' in metrics:
                    r_value = metrics['balance_ratio']
                    r_text = f"R = {r_value:.4f}\n"
                    r_text += f"Interpretation: {self.get_r_interpretation(r_value)}"
                    
                    # 根据R值设置颜色
                    if r_value == float('inf'):
                        r_color = "red"
                    elif r_value > 2.0:
                        r_color = "darkgreen"
                    elif r_value > 1.5:
                        r_color = "green"
                    elif r_value > 1.0:
                        r_color = "orange"
                    else:
                        r_color = "red"
                    
                    self.r_value_label.config(text=r_text, foreground=r_color)
                
                # Update charts with all denoising results
                self.update_plots(metrics)
                
                # Update metrics display
                self.update_metrics_display(metrics)
                
                file_name = os.path.basename(self.current_file) if self.current_file else "Unknown file"
                self.status_var.set(f"Denoising completed: {file_name} ({self.denoise_times} iterations)")
                
                # Print metrics to console
                print("\n" + "="*60)
                print(f"Denoising Metrics for {file_name} ({self.denoise_times} iterations):")
                print("-"*60)
                
                # 一阶差分范数信息
                print("\nFirst Difference Norms:")
                print(f"  Original L2 Norm: {metrics['first_diff_norm_original']:.4f}")
                print(f"  Denoised L2 Norm: {metrics['first_diff_norm_denoised']:.4f}")
                print(f"  Reduction: {metrics['first_diff_reduction']:.2f}%")
                print(f"  Original L1 Norm: {metrics['first_diff_l1_original']:.4f}")
                print(f"  Denoised L1 Norm: {metrics['first_diff_l1_denoised']:.4f}")
                print(f"  L1 Reduction: {metrics['first_diff_l1_reduction']:.2f}%")
                
                # 二阶差分范数信息
                print("\nSecond Difference Norms:")
                print(f"  Original L2 Norm: {metrics['second_diff_norm_original']:.4f}")
                print(f"  Denoised L2 Norm: {metrics['second_diff_norm_denoised']:.4f}")
                print(f"  Reduction: {metrics['second_diff_reduction']:.2f}%")
                print(f"  Original L1 Norm: {metrics['second_diff_l1_original']:.4f}")
                print(f"  Denoised L1 Norm: {metrics['second_diff_l1_denoised']:.4f}")
                print(f"  L1 Reduction: {metrics['second_diff_l1_reduction']:.2f}%")
                
                # R值信息
                if 'balance_ratio' in metrics:
                    print("\nBalance Ratio (R):")
                    print(f"  R = {metrics['balance_ratio']:.4f}")
                    print(f"  Interpretation: {self.get_r_interpretation(metrics['balance_ratio'])}")
                
                # 解释
                reduction_2nd = metrics['second_diff_reduction']
                if reduction_2nd > 50:
                    print("\n  → Excellent noise reduction (2nd diff reduction > 50%)")
                elif reduction_2nd > 30:
                    print("\n  → Good noise reduction (2nd diff reduction > 30%)")
                elif reduction_2nd > 10:
                    print("\n  → Moderate noise reduction (2nd diff reduction > 10%)")
                else:
                    print("\n  → Limited noise reduction")
                
                print("="*60)
                
                # Print results for each iteration
                print("\nDenoising Results by Iteration:")
                print("-"*60)
                for i, result in enumerate(self.denoised_results):
                    iteration_metrics = calculate_metrics(self.original_y, result['preprocessed'])
                    first_norms = self.first_diff_norms[i]
                    second_norms = self.second_diff_norms[i]
                    print(f"Iteration {i+1}: 1st L2={first_norms['l2']:.4f}, 2nd L2={second_norms['l2']:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Denoising failed: {str(e)}")
            self.status_var.set("Denoising failed")
            print(f"Denoising failed: {e}")
            import traceback
            traceback.print_exc()
    
    def update_plots(self, metrics=None):
        """Update chart display with larger fonts"""
        # Clear all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        
        # 设置白色背景
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('white')
        
        if self.original_y_raw is not None and self.x_data is not None:
            # ========== First Plot: Original vs Denoised ==========
            # Plot original curve (using inverse preprocessed data for visualization)
            self.ax1.plot(self.x_data, self.original_y_raw, 'k-', alpha=0.7, 
                         label='Original (Noisy) Curve', linewidth=2)
            
            # Plot only the last denoised curve (if denoised_results is not empty)
            if self.denoised_results:
                # Only use the last result
                last_result = self.denoised_results[-1]
                label = f'Denoised (iterations= {last_result["iteration"]})'
                self.ax1.plot(self.x_data, last_result['raw'], 'r-', 
                             label=label, linewidth=2, alpha=0.8)
            
            # 设置图例字体大小
            self.ax1.legend(loc='best', fontsize=17)
            
            self.ax1.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
            self.ax1.set_ylabel(r"$I$ ($\mathrm{a.u.}$)", fontsize=20)
            self.ax1.tick_params(axis='both', which='major', labelsize=17)
            self.ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
            
            # ========== Second Plot: I_orig - I_deno ==========
            if self.denoised_results:
                last_result = self.denoised_results[-1]['raw']
                residuals = self.original_y_raw - last_result
                
                # Plot residuals
                self.ax2.plot(self.x_data, residuals, 'r-', alpha=0.7, 
                             label='$I_{\\text{orig}} - I_{\\text{deno}}$', linewidth=2)
                
                # Add zero line - 加粗虚线
                self.ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Add mean line - 加粗实线
                residual_mean = np.mean(residuals)
                self.ax2.axhline(y=residual_mean, color='b', linestyle='-', 
                                alpha=0.7, linewidth=1.5, label=f'Mean: {residual_mean:.4f}')
                
                # Add ±2σ lines - 加粗虚线
                residual_std = np.std(residuals)
                self.ax2.axhline(y=residual_mean + 2*residual_std, color='g', 
                                linestyle=':', alpha=0.7, linewidth=1.5, label=f'±2σ: {2*residual_std:.4f}')
                self.ax2.axhline(y=residual_mean - 2*residual_std, color='g', 
                                linestyle=':', alpha=0.7, linewidth=1.5)
                
                # 设置图例字体大小，位置在右上角
                self.ax2.legend(loc='upper right', fontsize=17)
                
                self.ax2.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
                self.ax2.set_ylabel(r"$I_{\text{orig}} - I_{\text{deno}}$", fontsize=20)
                self.ax2.tick_params(axis='both', which='major', labelsize=17)
                self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
                
                # ========== Third Plot: Second Differences ==========
                if self.denoised_results:
                    last_result_preprocessed = self.denoised_results[-1]['preprocessed']
                    
                    # 计算二阶差分
                    original_second_diff = np.diff(self.original_y, 2)
                    denoised_second_diff = np.diff(last_result_preprocessed, 2)
                    
                    # 由于二阶差分会使数据点减少2个，我们需要对应的q值
                    # 对于二阶差分，我们使用中间的q值（去掉第一个和最后一个）
                    if len(self.x_data) >= 3:
                        q_values = self.x_data[1:-1]  # 去掉第一个和最后一个q值
                        
                        # 绘制原始二阶差分
                        self.ax3.plot(q_values, original_second_diff, 'k-', alpha=0.7, 
                                     label='Original $\Delta_y^2$', linewidth=1.5)
                        
                        # 绘制去噪后二阶差分
                        self.ax3.plot(q_values, denoised_second_diff, 'r-', alpha=0.8, 
                                     label='Denoised $\Delta_y^2$', linewidth=1.5)
                        
                        # 添加零线
                        self.ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
                        
                        # 设置图例在右侧
                        self.ax3.legend(loc='center right', fontsize=17)
                        
                        # 添加标题显示范数信息
                        if metrics:
                            title_text = f"Second Differences (L2 Norm: {metrics['second_diff_norm_original']:.2f} → {metrics['second_diff_norm_denoised']:.2f})"
                            self.ax3.set_title(title_text, fontsize=18)
                        
                        self.ax3.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
                        self.ax3.set_ylabel(r"$\Delta_y^2$", fontsize=20)
                        self.ax3.tick_params(axis='both', which='major', labelsize=17)
                        self.ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
            else:
                # No denoised results yet
                self.ax2.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
                self.ax2.set_ylabel(r"$I_{\text{orig}} - I_{\text{deno}}$", fontsize=20)
                self.ax2.tick_params(axis='both', which='major', labelsize=17)
                self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
                
                self.ax3.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
                self.ax3.set_ylabel(r"$\Delta_y^2$", fontsize=20)
                self.ax3.tick_params(axis='both', which='major', labelsize=17)
                self.ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
        else:
            # No data loaded yet
            self.ax1.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
            self.ax1.set_ylabel(r"$I$ ($\mathrm{a.u.}$)", fontsize=20)
            self.ax1.tick_params(axis='both', which='major', labelsize=17)
            self.ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
            
            self.ax2.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
            self.ax2.set_ylabel(r"$I_{\text{orig}} - I_{\text{deno}}$", fontsize=20)
            self.ax2.tick_params(axis='both', which='major', labelsize=17)
            self.ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
            
            self.ax3.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=20)
            self.ax3.set_ylabel(r"$\Delta_y^2$", fontsize=20)
            self.ax3.tick_params(axis='both', which='major', labelsize=17)
            self.ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)  # 浅灰色虚线网格
        
        # 使用constrained_layout自动管理间距，不额外调整
        self.canvas.draw()
    
    def update_metrics_display(self, metrics=None):
        """Update metrics display area with larger font"""
        self.metrics_text.config(state=tk.NORMAL)  # Enable editing
        self.metrics_text.delete(1.0, tk.END)  # Clear existing text
        
        if metrics and self.denoised_results:
            # Get file info
            file_name = os.path.basename(self.current_file) if self.current_file else "Unknown"
            num_samples = len(self.original_y) if self.original_y is not None else 0
            
            # Format metrics text
            metrics_text = f"File: {file_name}\n"
            metrics_text += f"Samples: {num_samples}\n"
            metrics_text += f"Denoise Iterations: {self.denoise_times}\n"
            metrics_text += "-" * 60 + "\n"
            metrics_text += "FIRST DIFFERENCE NORMS:\n"
            metrics_text += "-" * 60 + "\n"
            metrics_text += f"Original L2 Norm: {metrics['first_diff_norm_original']:.4f}\n"
            metrics_text += f"Denoised L2 Norm: {metrics['first_diff_norm_denoised']:.4f}\n"
            metrics_text += f"Reduction: {metrics['first_diff_reduction']:.2f}%\n"
            metrics_text += f"Original L1 Norm: {metrics['first_diff_l1_original']:.4f}\n"
            metrics_text += f"Denoised L1 Norm: {metrics['first_diff_l1_denoised']:.4f}\n"
            metrics_text += f"L1 Reduction: {metrics['first_diff_l1_reduction']:.2f}%\n"
            
            # 一阶差分减少程度解释
            reduction_1st = metrics['first_diff_reduction']
            if reduction_1st > 50:
                metrics_text += "\n  → Excellent (1st diff reduction > 50%)\n"
            elif reduction_1st > 30:
                metrics_text += "\n  → Good (1st diff reduction > 30%)\n"
            elif reduction_1st > 10:
                metrics_text += "\n  → Moderate (1st diff reduction > 10%)\n"
            else:
                metrics_text += "\n  → Limited (1st diff reduction < 10%)\n"
            
            metrics_text += "\n" + "-" * 60 + "\n"
            metrics_text += "SECOND DIFFERENCE NORMS:\n"
            metrics_text += "-" * 60 + "\n"
            metrics_text += f"Original L2 Norm: {metrics['second_diff_norm_original']:.4f}\n"
            metrics_text += f"Denoised L2 Norm: {metrics['second_diff_norm_denoised']:.4f}\n"
            metrics_text += f"Reduction: {metrics['second_diff_reduction']:.2f}%\n"
            metrics_text += f"Original L1 Norm: {metrics['second_diff_l1_original']:.4f}\n"
            metrics_text += f"Denoised L1 Norm: {metrics['second_diff_l1_denoised']:.4f}\n"
            metrics_text += f"L1 Reduction: {metrics['second_diff_l1_reduction']:.2f}%\n"
            
            # 二阶差分减少程度解释
            reduction_2nd = metrics['second_diff_reduction']
            if reduction_2nd > 50:
                metrics_text += "\n  → Excellent (2nd diff reduction > 50%)\n"
            elif reduction_2nd > 30:
                metrics_text += "\n  → Good (2nd diff reduction > 30%)\n"
            elif reduction_2nd > 10:
                metrics_text += "\n  → Moderate (2nd diff reduction > 10%)\n"
            else:
                metrics_text += "\n  → Limited (2nd diff reduction < 10%)\n"
            
            # R值信息
            if 'balance_ratio' in metrics:
                metrics_text += "\n" + "-" * 60 + "\n"
                metrics_text += "BALANCE RATIO (R):\n"
                metrics_text += "-" * 60 + "\n"
                metrics_text += f"R = {metrics['balance_ratio']:.4f}\n"
                metrics_text += f"Interpretation: {self.get_r_interpretation(metrics['balance_ratio'])}\n"
                
                # 公式解释
                metrics_text += "\nFormula: R = (1 - ‖Δ²y_den‖₂/‖Δ²y_orig‖₂) / (1 - ‖Δy_den‖₂/‖Δy_orig‖₂)\n"
            
            # 迭代信息
            if len(self.denoised_results) > 1:
                metrics_text += "\n" + "-" * 60 + "\n"
                metrics_text += "ITERATION PROGRESS:\n"
                metrics_text += "-" * 60 + "\n"
                
                for i in range(len(self.denoised_results)):
                    first_norms = self.first_diff_norms[i]
                    second_norms = self.second_diff_norms[i]
                    metrics_text += f"Iteration {i+1}:\n"
                    metrics_text += f"  1st L2 Norm = {first_norms['l2']:.4f}"
                    if i > 0:
                        prev_first_norm = self.first_diff_norms[i-1]['l2']
                        change_first = first_norms['l2'] - prev_first_norm
                        percent_change_first = change_first / prev_first_norm * 100 if prev_first_norm != 0 else 0
                        metrics_text += f" (Δ = {change_first:.4f}, {percent_change_first:.1f}%)"
                    metrics_text += "\n"
                    metrics_text += f"  2nd L2 Norm = {second_norms['l2']:.4f}"
                    if i > 0:
                        prev_second_norm = self.second_diff_norms[i-1]['l2']
                        change_second = second_norms['l2'] - prev_second_norm
                        percent_change_second = change_second / prev_second_norm * 100 if prev_second_norm != 0 else 0
                        metrics_text += f" (Δ = {change_second:.4f}, {percent_change_second:.1f}%)"
                    metrics_text += "\n\n"
            
            # $I_{\text{orig}} - I_{\text{deno}}$ 统计
            if self.denoised_results:
                last_result = self.denoised_results[-1]['raw']
                residuals = self.original_y_raw - last_result
                residual_mean = np.mean(residuals)
                residual_std = np.std(residuals)
                
                metrics_text += "\n" + "-" * 60 + "\n"
                metrics_text += "$I_{\\text{orig}} - I_{\\text{deno}}$ STATISTICS:\n"
                metrics_text += "-" * 60 + "\n"
                metrics_text += f"Mean: {residual_mean:.6f}\n"
                metrics_text += f"Std Dev: {residual_std:.6f}\n"
                metrics_text += f"Min: {np.min(residuals):.6f}\n"
                metrics_text += f"Max: {np.max(residuals):.6f}\n"
                metrics_text += f"Range: {np.max(residuals) - np.min(residuals):.6f}\n"
                metrics_text += f"±2σ Range: [{residual_mean - 2*residual_std:.6f}, {residual_mean + 2*residual_std:.6f}]\n"
                
                # 计算超出±2σ范围的数据点百分比
                outside_2sigma = np.sum(np.abs(residuals - residual_mean) > 2*residual_std)
                outside_percent = outside_2sigma / len(residuals) * 100
                metrics_text += f"Outside ±2σ: {outside_2sigma} points ({outside_percent:.1f}%)\n"
        else:
            metrics_text = "No metrics available yet.\n\n"
            metrics_text += "Load a curve and run denoising to see metrics here."
        
        # Insert the formatted text
        self.metrics_text.insert(tk.END, metrics_text)
        self.metrics_text.config(state=tk.DISABLED)  # Make read-only again
    
    def clear_display(self):
        """Clear display"""
        self.current_file = None
        self.x_data = None
        self.original_y = None
        self.original_y_raw = None
        self.denoised_y = None
        self.denoised_results = []
        self.first_diff_norms = []
        self.second_diff_norms = []
        self.first_diff_label.config(text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                    foreground="darkblue")
        self.second_diff_label.config(text="Original L2: --\nDenoised L2: --\nReduction: --%", 
                                     foreground="darkblue")
        self.r_value_label.config(text="R = --\nInterpretation: --", 
                                 foreground="darkred")
        self.initialize_plots()
        self.update_metrics_display()
        self.status_var.set("Ready - Select a file to begin")

# ==================== Main Function ====================
def main():
    """Main function"""
    print("="*60)
    print("1D Curve Noise2Noise Denoising Test Tool")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Model path: {config.model_path}")
    print(f"Data directory: {config.data_root}")
    print(f"Data preprocessing: clip_negative={config.clip_negative}, use_log1p={config.use_log1p}")
    print("="*60)
    
    # Check model file
    if not os.path.exists(config.model_path):
        print(f"Warning: Model file does not exist: {config.model_path}")
        print("Please train the model first or provide the correct model path.")
    
    # Create GUI application
    root = tk.Tk()
    app = DenoiseApp(root)
    
    # Run application
    root.mainloop()

# ==================== Run Main Program ====================
if __name__ == "__main__":
    main()