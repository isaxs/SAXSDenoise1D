import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
# 设置matplotlib使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox, scrolledtext
import re

# 设置设备（自动检测GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义模型结构（需要与训练时相同）
class GatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(GatedConv1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels * 2,  # 双倍输出通道用于门控
            kernel_size, 
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        # 分成两部分：一部分用于特征提取，一部分用于门控
        gate, feature = torch.chunk(x, 2, dim=1)
        return self.sigmoid(gate) * feature

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

class MultiScaleReceptiveField(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleReceptiveField, self).__init__()
        
        # 不同尺度的卷积核
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=9, padding=4)
        
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        # 并行多尺度卷积
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(x))
        out3 = self.relu(self.conv3(x))
        out4 = self.relu(self.conv4(x))
        
        # 拼接多尺度特征
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.bn(out)
        return out

# 平坦区域检测模块
class FlatRegionDetector(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(FlatRegionDetector, self).__init__()
        
        # 多尺度梯度分析用于平坦区域检测
        self.gradient_analysis = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU()
        )
        
        # 平坦区域分类
        self.flat_classifier = nn.Sequential(
            nn.Conv1d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 局部方差计算用于平坦度测量
        self.variance_conv = nn.Conv1d(1, out_channels, kernel_size=11, padding=5, groups=1)
        
    def forward(self, x):
        # x shape: (batch_size, 1, seq_len)
        
        # 计算局部方差作为平坦度指标
        local_mean = F.avg_pool1d(x, kernel_size=11, stride=1, padding=5)
        local_var = F.avg_pool1d((x - local_mean) ** 2, kernel_size=11, stride=1, padding=5)
        
        # 归一化方差
        var_norm = torch.sigmoid(-local_var * 100)  # 低方差 -> 高平坦度
        
        # 基于梯度的平坦区域检测
        grad_features = self.gradient_analysis(x)
        flat_mask = self.flat_classifier(grad_features)
        
        # 结合方差和梯度信息
        combined_flatness = (flat_mask + var_norm) / 2
        
        return combined_flatness

# 增强的自注意力模块 - 包含平坦区域增强
class SelfAttention1D(nn.Module):
    def __init__(self, in_channels, edge_enhance=True, num_heads=8, flat_region_enhance=True):
        super(SelfAttention1D, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.flat_region_enhance = flat_region_enhance
        
        # 多头注意力投影
        self.query = nn.Conv1d(in_channels, in_channels, 1)
        self.key = nn.Conv1d(in_channels, in_channels, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        
        # 输出投影
        self.out_proj = nn.Conv1d(in_channels, in_channels, 1)
        
        # 可学习的缩放参数
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 边缘增强参数
        self.edge_enhance = edge_enhance
        if edge_enhance:
            self.edge_weight = nn.Parameter(torch.ones(1, in_channels, 1))
        
        # 频率分析组件用于噪声检测
        self.freq_analysis = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 峰值检测组件
        self.peak_detection = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # 平坦区域检测
        if flat_region_enhance:
            self.flat_detection = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 4, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(in_channels // 4, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        batch_size, C, seq_len = x.size()
        
        # 多头注意力
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        
        # 计算注意力分数
        attention_scores = torch.einsum('bhdn,bhdm->bhnm', Q, K) / (self.head_dim ** 0.5)
        
        # 频率分析以识别噪声区域
        freq_weights = self.freq_analysis(x)
        freq_attention = torch.mean(freq_weights, dim=1, keepdim=True)  # (batch_size, 1, seq_len)
        
        # 峰值检测以识别散射峰
        peak_mask = self.peak_detection(x)  # (batch_size, 1, seq_len)
        
        # 平坦区域检测
        if self.flat_region_enhance:
            flat_mask = self.flat_detection(x)  # (batch_size, 1, seq_len)
            # 增强平坦区域的注意力以获得更好的平滑效果
            flat_enhancement = flat_mask * 1.5  # 在平坦区域增强注意力
        else:
            flat_mask = torch.ones_like(peak_mask)
            flat_enhancement = torch.ones_like(peak_mask)
        
        # 结合频率、峰值和平坦信息
        content_aware_mask = (1 - peak_mask) * freq_attention * flat_enhancement
        
        # 将内容感知掩码应用于注意力分数
        content_aware_mask = content_aware_mask.view(batch_size, 1, 1, seq_len)
        attention_scores = attention_scores * content_aware_mask
        
        # 边缘增强
        if self.edge_enhance and seq_len > 10:
            edge_mask = torch.ones_like(attention_scores)
            edge_size = max(5, seq_len // 20)  # 边缘区域大小
            
            # 创建边缘权重掩码 - 增强边缘处的注意力
            edge_mask[:, :, :, :edge_size] *= 2.0  # 左边缘
            edge_mask[:, :, :, -edge_size:] *= 2.0  # 右边缘
            
            attention_scores = attention_scores * edge_mask
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 将注意力应用于值
        out = torch.einsum('bhnm,bhdm->bhdn', attention_weights, V)
        out = out.contiguous().view(batch_size, C, seq_len)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 带有可学习权重的残差连接
        out = self.gamma * out + x
        
        # 进一步边缘增强
        if self.edge_enhance:
            out = out * self.edge_weight
            
        return out, attention_weights, peak_mask.squeeze(1), flat_mask.squeeze(1)

# 增强的自适应平滑网络V2 - 改进平坦区域处理
class AdaptiveSmoothingNetV2(nn.Module):
    def __init__(self, base_channels=64):
        super(AdaptiveSmoothingNetV2, self).__init__()
        
        # 增强的平滑因子处理网络
        self.smooth_factor_fc = nn.Sequential(
            nn.Linear(1, base_channels * 2),
            nn.ReLU(),
            nn.Linear(base_channels * 2, base_channels * 2),
            nn.ReLU(),
            nn.Linear(base_channels * 2, base_channels)
        )
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(base_channels)
        )
        
        # 多尺度感受野模块
        self.multiscale = MultiScaleReceptiveField(base_channels, base_channels * 2)
        
        # 自注意力模块用于全局上下文，包含边缘和平坦区域增强
        self.self_attention = SelfAttention1D(base_channels * 2, edge_enhance=True, flat_region_enhance=True)
        
        # 平坦区域检测器
        self.flat_detector = FlatRegionDetector(1, base_channels // 2)
        
        # 门控残差块序列
        self.residual_blocks = nn.Sequential(
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=1),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=2),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=4),
            ResidualGatedBlock(base_channels * 2, kernel_size=5, dilation=8)
        )
        
        # 边缘保护模块
        self.edge_protection = nn.Sequential(
            nn.Conv1d(1, base_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels//2, base_channels//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels//4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 平坦区域增强模块
        self.flat_enhancement = nn.Sequential(
            nn.Conv1d(1, base_channels//2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels//2, base_channels//4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(base_channels//4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # 增强的自适应平滑模块，包含平坦区域感知
        self.adaptive_smoothing = nn.Sequential(
            nn.Conv1d(base_channels * 2 + base_channels + 1, base_channels * 2, kernel_size=3, padding=1),  # +1 用于平坦特征
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出平滑权重
        )
        
        # 增强的输出层
        self.output_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        )
        
    def forward(self, x, smooth_factor):
        # x 形状: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # 使用与训练代码相同的方式处理平滑因子
        smooth_factor_normalized = smooth_factor  # 与训练代码保持一致，不除以256.0
        
        # 处理平滑因子
        sf_encoded = self.smooth_factor_fc(smooth_factor_normalized.unsqueeze(-1))  # (batch_size, base_channels)
        sf_encoded = sf_encoded.unsqueeze(-1)  # (batch_size, base_channels, 1)
        
        # 添加通道维度: (batch_size, 1, seq_len)
        x_input = x.unsqueeze(1)
        
        # 检测平坦区域
        flat_region_mask = self.flat_detector(x_input)  # (batch_size, 1, seq_len)
        
        # 初始特征提取
        features = self.initial_conv(x_input)
        
        # 多尺度感受野
        features = self.multiscale(features)
        
        # 自注意力用于全局上下文，包含平坦区域感知
        features, attention_weights, peak_mask, flat_mask = self.self_attention(features)
        
        # 门控残差处理
        features = self.residual_blocks(features)
        
        # 扩展平滑因子编码以匹配序列长度
        sf_expanded = sf_encoded.repeat(1, 1, seq_len)  # (batch_size, base_channels, seq_len)
        
        # 扩展平坦区域掩码以匹配特征维度
        flat_features = flat_region_mask  # 已经是 (batch_size, 1, seq_len)
        
        # 将特征与平滑因子编码和平坦特征拼接
        features_with_sf = torch.cat([features, sf_expanded, flat_features], dim=1)
        
        # 带有平滑因子和平坦区域信息的自适应平滑权重
        smooth_weights = self.adaptive_smoothing(features_with_sf)
        
        # 使用峰值掩码保护散射峰免于过度平滑
        peak_protection = 1 - peak_mask.unsqueeze(1)  # 反转掩码：1为非峰值区域，0为峰值区域
        
        # 使用平坦掩码增强平坦区域的平滑
        flat_enhancement = flat_region_mask * 0.4  # 在平坦区域增加额外的平滑
        
        # 应用峰值保护和平坦增强到平滑权重
        smooth_weights = smooth_weights * peak_protection + 0.05 * (1 - peak_protection)
        smooth_weights = smooth_weights + flat_enhancement
        smooth_weights = torch.clamp(smooth_weights, 0, 1)  # 确保权重保持在[0,1]范围内
        
        # 计算边缘保护掩码
        edge_mask = self.edge_protection(x_input)
        
        # 计算平坦区域增强掩码
        flat_enhance_mask = self.flat_enhancement(x_input)
        
        # 应用边缘保护到平滑权重 - 边缘获得适度平滑
        smooth_weights = smooth_weights * (1 - edge_mask) + 0.3 * edge_mask
        
        # 应用平坦区域增强 - 平坦区域获得更多平滑
        smooth_weights = smooth_weights * (1 + 0.5 * flat_enhance_mask)
        smooth_weights = torch.clamp(smooth_weights, 0, 1)  # 确保权重保持在[0,1]范围内
        
        # 应用平滑权重
        # 使用滑动窗口平均作为基础平滑
        base_smooth = F.avg_pool1d(x_input, kernel_size=5, stride=1, padding=2)
        
        # 使用自适应权重组合原始输入和基础平滑
        smoothed = smooth_weights * base_smooth + (1 - smooth_weights) * x_input
        
        # 最终输出
        output = self.output_conv(smoothed).squeeze(1)
        
        return output, smooth_weights.squeeze(1), attention_weights, peak_mask, flat_mask

# 数据预处理函数：删除负值并返回有效的q和i值
def preprocess_data(q_values, i_values):
    """
    预处理数据：删除负值，返回有效的q和i值
    
    参数:
        q_values: 原始的q值数组
        i_values: 原始的强度值数组
    
    返回:
        valid_q: 有效的q值（对应的i值非负）
        valid_i: 有效的i值（非负）
    """
    # 找到非负值的索引
    non_negative_indices = i_values >= 0
    
    if np.sum(non_negative_indices) == 0:
        raise ValueError("所有强度值都是负值，无法处理")
    
    # 删除负值
    valid_q = q_values[non_negative_indices]
    valid_i = i_values[non_negative_indices]
    
    # 打印删除的负值数量
    num_removed = len(i_values) - len(valid_i)
    if num_removed > 0:
        print(f"删除了 {num_removed} 个负值数据点")
        print(f"剩余 {len(valid_i)} 个有效数据点")
    
    return valid_q, valid_i

# 预测函数 - 返回平滑曲线、注意力权重和平坦区域掩码
def predict_v2(model, x_values, smooth_factor=16.0):
    """
    使用训练好的模型进行预测
    
    参数:
        model: 训练好的模型
        x_values: 输入值
        smooth_factor: 平滑因子，控制平滑程度
    
    返回:
        smoothed_curve: 平滑后的曲线
        smooth_weights: 平滑权重
        attention_weights: 注意力权重
        peak_mask: 峰值掩码
        flat_mask: 平坦区域掩码
    """
    model.eval()  # 设置为评估模式
    # 转换为张量并移动到设备
    x_tensor = torch.tensor(x_values, dtype=torch.float32).to(device)
    sf_tensor = torch.tensor([smooth_factor], dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        smoothed_curve, smooth_weights, attention_weights, peak_mask, flat_mask = model(x_tensor.unsqueeze(0), sf_tensor)
        smoothed_curve = smoothed_curve.cpu().numpy()
        smooth_weights = smooth_weights.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()
        peak_mask = peak_mask.cpu().numpy()
        flat_mask = flat_mask.cpu().numpy()
    
    return smoothed_curve.flatten(), smooth_weights.flatten(), attention_weights, peak_mask.flatten(), flat_mask.flatten()

# 多次平滑函数
def predict_multiple_times(model, x_values, smooth_factor=16.0, smooth_times=2):
    """
    多次应用平滑模型
    
    参数:
        model: 训练好的模型
        x_values: 输入值
        smooth_factor: 平滑因子，控制平滑程度
        smooth_times: 平滑次数
    
    返回:
        smoothed_curve: 多次平滑后的曲线
        all_smoothed_curves: 所有次数的平滑曲线列表
        all_attention_weights: 所有次数的注意力权重列表
    """
    current_input = x_values.copy()
    all_smoothed_curves = [current_input]  # 包含原始数据
    all_attention_weights = []
    
    for i in range(smooth_times):
        smoothed_curve, smooth_weights, attention_weights, peak_mask, flat_mask = predict_v2(model, current_input, smooth_factor)
        current_input = smoothed_curve.copy()
        all_smoothed_curves.append(smoothed_curve)
        all_attention_weights.append(attention_weights)
        print(f"第 {i+1} 次平滑完成")
    
    return smoothed_curve, all_smoothed_curves, all_attention_weights

# 读取数据文件函数 - 修复编码问题
def read_data_file(file_path):
    """
    从文件中读取数据，跳过注释行
    
    参数:
        file_path: 文件路径
    
    返回:
        q_values: q值数组
        i_values: I_subtracted值数组
    """
    q_values = []
    i_values = []
    
    # 尝试不同的编码方式
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    # 跳过注释行
                    if line.startswith('#') or line.strip() == '':
                        continue
                    
                    # 尝试解析数据行
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            q_values.append(float(parts[0]))
                            i_values.append(float(parts[1]))
                    except ValueError:
                        # 如果解析失败，跳过这一行
                        continue
            # 如果成功读取数据，跳出循环
            if q_values:
                print(f"成功使用 {encoding} 编码读取文件")
                break
        except UnicodeDecodeError:
            # 如果编码错误，尝试下一种编码
            print(f"使用 {encoding} 编码失败，尝试下一种编码...")
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生错误: {e}")
            continue
    
    # 如果所有编码都失败，尝试二进制模式读取
    if not q_values:
        print("所有编码尝试失败，尝试二进制模式读取...")
        try:
            with open(file_path, 'rb') as f:
                for line in f:
                    try:
                        # 解码为字符串，忽略错误
                        line_str = line.decode('utf-8', errors='ignore')
                        # 跳过注释行
                        if line_str.startswith('#') or line_str.strip() == '':
                            continue
                        
                        # 尝试解析数据行
                        parts = line_str.split()
                        if len(parts) >= 2:
                            q_values.append(float(parts[0]))
                            i_values.append(float(parts[1]))
                    except (ValueError, UnicodeDecodeError):
                        # 如果解析失败，跳过这一行
                        continue
        except Exception as e:
            print(f"二进制模式读取也失败: {e}")
    
    return np.array(q_values), np.array(i_values)

# GUI应用程序类
class SmoothingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("曲线平滑工具")
        self.root.geometry("1400x800")  # 增加窗口宽度以容纳右侧信息窗口
        
        # 初始化变量
        self.model = None
        self.input_files = []
        self.output_dir = ""
        self.smooth_times = tk.IntVar(value=10)
        self.smooth_factor = tk.DoubleVar(value=256.0)
        self.current_file_index = 0
        self.q_values = None
        self.original_curve = None
        self.smoothed_curve = None
        self.current_file_path = None
        self.current_smooth_times = 0
        self.current_smooth_factor = 0.0
        
        # 加载模型
        self.load_model()
        
        # 创建GUI
        self.create_widgets()
        
    def load_model(self):
        """加载训练好的模型"""
        model_path = "final_smoothing_model_NN11_v2.pth"
        
        # 初始化V2模型
        self.model = AdaptiveSmoothingNetV2(base_channels=64)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            print(f"V2 Model loaded successfully from {model_path}")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {e}")
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架 - 使用左右布局
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧区域 - 图表和控制
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 右侧区域 - 信息窗口
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # 防止框架收缩
        
        # 配置左侧区域的网格权重
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)  # 图表区域占据主要空间
        
        # 顶部控制框架
        top_frame = ttk.Frame(left_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 输入文件选择
        ttk.Button(top_frame, text="选择文件", command=self.select_input_files).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(top_frame, text="输入文件:").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.input_files_var = tk.StringVar(value="未选择文件")
        ttk.Label(top_frame, textvariable=self.input_files_var, wraplength=400).grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 输出目录选择
        ttk.Button(top_frame, text="保存目录", command=self.select_output_dir).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(top_frame, text="输出目录:").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        self.output_dir_var = tk.StringVar(value="未选择目录")
        ttk.Label(top_frame, textvariable=self.output_dir_var, wraplength=400).grid(row=1, column=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # 参数设置
        params_frame = ttk.Frame(top_frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(params_frame, text="平滑次数:").grid(row=0, column=0, sticky=tk.W, padx=5)
        smooth_times_entry = ttk.Entry(params_frame, textvariable=self.smooth_times, width=10)
        smooth_times_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(params_frame, text="平滑因子:").grid(row=0, column=2, sticky=tk.W, padx=5)
        smooth_factor_entry = ttk.Entry(params_frame, textvariable=self.smooth_factor, width=10)
        smooth_factor_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # 图表显示区域 - 调整大小
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)  # 减小图表尺寸
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 底部按钮框架 - 确保不会被遮挡
        bottom_frame = ttk.Frame(left_frame)
        bottom_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # 处理按钮
        ttk.Button(bottom_frame, text="平滑当前文件", command=self.process_current_file).grid(row=0, column=0, padx=10, pady=5)
        
        # 保存按钮
        ttk.Button(bottom_frame, text="保存当前图表", command=self.save_current_plot).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(bottom_frame, text="保存当前数据", command=self.save_current_data).grid(row=0, column=2, padx=10, pady=5)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bottom_frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 文件导航（当选择多个文件时）
        self.nav_frame = ttk.Frame(bottom_frame)
        self.nav_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        self.prev_button = ttk.Button(self.nav_frame, text="上一个", command=self.prev_file)
        self.prev_button.grid(row=0, column=0, padx=5)
        
        self.file_info_var = tk.StringVar(value="")
        ttk.Label(self.nav_frame, textvariable=self.file_info_var).grid(row=0, column=1, padx=10)
        
        self.next_button = ttk.Button(self.nav_frame, text="下一个", command=self.next_file)
        self.next_button.grid(row=0, column=2, padx=5)
        
        # 初始隐藏导航按钮
        self.nav_frame.grid_remove()
        
        # 右侧信息窗口
        info_label = ttk.Label(right_frame, text="处理信息", font=('Arial', 12, 'bold'))
        info_label.pack(pady=(0, 10))
        
        # 创建带滚动条的文本框
        self.info_text = scrolledtext.ScrolledText(
            right_frame, 
            width=35, 
            height=30,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 清空信息按钮
        clear_button = ttk.Button(right_frame, text="清空信息", command=self.clear_info)
        clear_button.pack(pady=10)
        
        # 添加初始信息
        self.add_info("欢迎使用曲线平滑工具！")
        self.add_info("请选择输入文件和保存目录。")
    
    def add_info(self, message):
        """向信息窗口添加信息"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, f"{message}\n")
        self.info_text.see(tk.END)  # 自动滚动到底部
        self.info_text.config(state=tk.DISABLED)
    
    def clear_info(self):
        """清空信息窗口"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.config(state=tk.DISABLED)
        self.add_info("信息已清空")
    
    def select_input_files(self):
        """选择输入文件"""
        files = filedialog.askopenfilenames(
            title="选择需要平滑的数据文件",
            filetypes=[("Text files", "*.dat"), ("All files", "*.*")]
        )
        
        if files:
            self.input_files = list(files)
            if len(self.input_files) == 1:
                self.input_files_var.set(f"已选择文件: {os.path.basename(self.input_files[0])}")
                self.add_info(f"已选择文件: {os.path.basename(self.input_files[0])}")
            else:
                self.input_files_var.set(f"已选择 {len(self.input_files)} 个文件")
                self.add_info(f"已选择 {len(self.input_files)} 个文件")
            
            # 如果只选择了一个文件，立即显示
            if len(self.input_files) == 1:
                self.display_file(0)
            else:
                # 显示导航按钮
                self.nav_frame.grid()
                self.display_file(0)
    
    def select_output_dir(self):
        """选择输出目录"""
        directory = filedialog.askdirectory(title="选择保存目录")
        if directory:
            self.output_dir = directory
            self.output_dir_var.set(f"保存目录: {directory}")
            self.add_info(f"已设置保存目录: {directory}")
    
    def display_file(self, index):
        """显示指定索引的文件"""
        if not self.input_files or index < 0 or index >= len(self.input_files):
            return
            
        self.current_file_index = index
        file_path = self.input_files[index]
        self.current_file_path = file_path
        
        # 更新文件信息
        if len(self.input_files) > 1:
            self.file_info_var.set(f"文件 {index+1}/{len(self.input_files)}: {os.path.basename(file_path)}")
        
        # 读取并显示文件数据
        try:
            q_values, i_values = read_data_file(file_path)
            q_values, i_values = preprocess_data(q_values, i_values)
            
            self.q_values = q_values
            self.original_curve = i_values
            
            # 清空图表
            self.ax.clear()
            
            # 绘制原始曲线
            self.ax.plot(q_values, i_values, 'k-', linewidth=1.5, label='Original curve')
            self.ax.set_xlabel('q (A$^{-1}$)')  # 使用LaTeX格式避免上标字符警告
            self.ax.set_ylabel('Intensity (a.u.)')
            self.ax.set_title(f'Curve : {os.path.basename(file_path)}')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # 如果有平滑曲线，也显示
            if self.smoothed_curve is not None and len(self.smoothed_curve) == len(q_values):
                self.ax.plot(q_values, self.smoothed_curve, 'r-', linewidth=1.5, label='Smoothed curve')
                self.ax.legend()
            
            self.canvas.draw()
            self.add_info(f"已显示文件: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败: {e}")
            self.add_info(f"错误: 读取文件失败 - {e}")
    
    def prev_file(self):
        """显示上一个文件"""
        if self.current_file_index > 0:
            self.display_file(self.current_file_index - 1)
    
    def next_file(self):
        """显示下一个文件"""
        if self.current_file_index < len(self.input_files) - 1:
            self.display_file(self.current_file_index + 1)
    
    def process_current_file(self):
        """处理当前显示的文件"""
        if not self.input_files:
            messagebox.showwarning("警告", "请先选择输入文件")
            return
            
        if not self.output_dir:
            messagebox.showwarning("警告", "请先选择保存目录")
            return
        
        # 获取参数
        smooth_times = self.smooth_times.get()
        smooth_factor = self.smooth_factor.get()
        
        # 更新状态
        self.status_var.set("处理中...")
        self.add_info(f"开始处理当前文件: {os.path.basename(self.current_file_path)}")
        self.root.update()
        
        try:
            # 处理当前文件
            if self.process_single_file(self.current_file_path, smooth_times, smooth_factor):
                self.status_var.set("处理完成！")
                self.add_info(f"文件处理成功: {os.path.basename(self.current_file_path)}")
                
                # 更新图表显示平滑结果
                self.display_file(self.current_file_index)
            else:
                self.status_var.set("处理失败")
                
        except Exception as e:
            error_msg = f"处理文件 {os.path.basename(self.current_file_path)} 时出错: {e}"
            messagebox.showerror("错误", error_msg)
            self.add_info(f"错误: {error_msg}")
            self.status_var.set("处理失败")
    
    def process_single_file(self, file_path, smooth_times, smooth_factor):
        """
        处理单个文件：读取、平滑、保存结果
        
        参数:
            file_path: 数据文件路径
            smooth_times: 平滑次数
            smooth_factor: 平滑因子
        
        返回:
            success: 处理是否成功
        """
        try:
            print(f"\n正在处理文件: {file_path}")
            self.add_info(f"正在处理: {os.path.basename(file_path)}")
            
            # 读取数据文件
            q_values, i_values = read_data_file(file_path)
            print(f"成功加载数据，数据点数量: {len(q_values)}")
            self.add_info(f"数据点数量: {len(q_values)}")
            
            # 检查数据是否有效
            if len(q_values) == 0:
                print("文件中没有有效数据，跳过此文件")
                self.add_info("错误: 文件中没有有效数据")
                return False
            
            # 数据预处理：删除负值
            try:
                q_values, i_values = preprocess_data(q_values, i_values)
                print(f"删除负值后，剩余数据点: {len(q_values)}")
                self.add_info(f"删除负值后，剩余数据点: {len(q_values)}")
            except ValueError as e:
                print(f"数据预处理错误: {e}")
                self.add_info(f"数据预处理错误: {e}")
                return False
            
            # 将对数转换后的值用于平滑
            i_values_log = np.log1p(i_values)  # 使用log1p转换到对数坐标
            
            # 使用多次平滑函数
            print(f"开始平滑处理 (平滑次数={smooth_times}, 平滑因子={smooth_factor})...")
            self.add_info(f"开始平滑处理 (次数={smooth_times}, 因子={smooth_factor})")
            smoothed_curve_log, all_smoothed_curves, all_attention_weights = predict_multiple_times(
                self.model, i_values_log, smooth_factor, smooth_times)
            
            # 将对数坐标的平滑结果转换回线性坐标
            final_smoothed_linear = np.expm1(smoothed_curve_log)
            
            # 计算差异（原始曲线与平滑曲线的差异）
            difference = i_values - final_smoothed_linear
            
            # 获取最后一次平滑的注意力分数
            final_attention_weights = all_attention_weights[-1]
            # 计算注意力分数的平均值作为整体注意力分数
            attention_score_2d = np.mean(final_attention_weights, axis=(0, 1))  # 平均所有头和批次
            # 将对角线元素作为每个位置的注意力分数，确保是一维数组
            attention_score = np.diag(attention_score_2d)
            
            # 确保所有数组长度一致
            min_length = min(len(q_values), len(i_values), len(final_smoothed_linear), len(difference), len(attention_score))
            q_values = q_values[:min_length]
            i_values = i_values[:min_length]
            final_smoothed_linear = final_smoothed_linear[:min_length]
            difference = difference[:min_length]
            attention_score = attention_score[:min_length]
            
            # 保存处理结果到内存，不自动保存到文件
            self.q_values = q_values
            self.original_curve = i_values
            self.smoothed_curve = final_smoothed_linear
            self.current_smooth_times = smooth_times
            self.current_smooth_factor = smooth_factor
            self.current_attention_score = attention_score
            self.current_difference = difference
            
            print(f"文件处理完成: {file_path}")
            return True
            
        except Exception as e:
            print(f"处理文件时出错 {file_path}: {e}")
            self.add_info(f"处理错误: {e}")
            return False
    
    def save_current_plot(self):
        """保存当前图表"""
        if not self.output_dir:
            messagebox.showwarning("警告", "请先选择保存目录")
            return
            
        if self.current_file_path is None or self.smoothed_curve is None:
            messagebox.showwarning("警告", "没有可保存的图表数据，请先平滑当前文件")
            return
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        image_output_path = os.path.join(self.output_dir, f"{base_name}_smoothing_plot.png")
        
        # 可视化并保存图像
        self.visualize_final_smoothing(
            self.q_values, 
            self.original_curve, 
            self.smoothed_curve, 
            self.current_smooth_times, 
            image_output_path
        )
        
        self.add_info(f"图表已保存: {os.path.basename(image_output_path)}")
    
    def save_current_data(self):
        """保存当前数据"""
        if not self.output_dir:
            messagebox.showwarning("警告", "请先选择保存目录")
            return
            
        if self.current_file_path is None or self.smoothed_curve is None:
            messagebox.showwarning("警告", "没有可保存的数据，请先平滑当前文件")
            return
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        data_output_path = os.path.join(self.output_dir, f"{base_name}_smoothed.dat")
        
        # 保存结果到文件（5列：q, original, smoothed, difference, attention_score）
        results_df = pd.DataFrame({
            'q_values': self.q_values,
            'original_intensity': self.original_curve,
            'smoothed_intensity': self.smoothed_curve,
            'difference': self.current_difference,
            'attention_score': self.current_attention_score
        })
        
        # 添加元数据作为注释
        metadata = f"""# Smoothing results (V2 Model)
# Original file: {self.current_file_path}
# Smooth factor: {self.current_smooth_factor}
# Smooth times: {self.current_smooth_times}
# Note: Data processed with log1p/expm1 transformation and negative values removed
"""
        
        # 先写入元数据
        with open(data_output_path, 'w') as f:
            f.write(metadata)
        # 再写入数据
        results_df.to_csv(data_output_path, mode='a', index=False)
        
        self.add_info(f"数据已保存: {os.path.basename(data_output_path)}")
    
    def visualize_final_smoothing(self, q_values, original_curve, final_smoothed_curve, smooth_times=2, save_path=None):
        """
        可视化多次平滑的最终结果 - 只显示原始曲线和最后一次平滑曲线
        
        参数:
            q_values: q值数组
            original_curve: 原始曲线
            final_smoothed_curve: 最后一次平滑后的曲线
            smooth_times: 平滑次数
            save_path: 图像保存路径，如果为None则不保存
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制原始曲线
        ax.plot(q_values, original_curve, 'k-', linewidth=2, label='Original curve')
        
        # 绘制最后一次平滑曲线
        ax.plot(q_values, final_smoothed_curve, 'r-', linewidth=2, 
                label=f'Smoothed curve ({smooth_times} times)')
        
        ax.set_xlabel('q (A$^{-1}$)')  # 使用LaTeX格式避免上标字符警告
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'Smoothing Result ({smooth_times} times)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 如果提供了保存路径，则保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Smoothing result saved as '{save_path}'")
        
        plt.close(fig)

# 主函数
def main():
    root = tk.Tk()
    app = SmoothingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()