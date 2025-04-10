#!/usr/bin/env python3
"""
超神量子共生系统 - 深度市场模式识别引擎
使用深度学习识别市场价格形态和隐藏模式
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
from enum import Enum, auto

# 深度学习相关导入
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("DeepPatternRecognition")

class PatternCategory(Enum):
    """市场模式分类"""
    REVERSAL_BULLISH = auto()  # 看涨反转
    REVERSAL_BEARISH = auto()  # 看跌反转
    CONTINUATION_BULLISH = auto()  # 看涨持续
    CONTINUATION_BEARISH = auto()  # 看跌持续
    CONSOLIDATION = auto()     # 盘整
    BREAKOUT_BULLISH = auto()  # 看涨突破
    BREAKOUT_BEARISH = auto()  # 看跌突破
    VOLATILITY_EXPANSION = auto()  # 波动性扩张
    VOLATILITY_CONTRACTION = auto()  # 波动性收缩
    ENERGY_ACCUMULATION = auto()  # 能量积累
    ENERGY_RELEASE = auto()    # 能量释放
    QUANTUM_ANOMALY = auto()   # 量子异常

class MarketPatternDataset(Dataset):
    """市场模式数据集"""
    def __init__(self, data, window_size=50, transform=None):
        """
        初始化数据集
        
        参数:
            data: DataFrame, 包含OHLCV数据
            window_size: 窗口大小
            transform: 数据转换函数
        """
        self.data = data
        self.window_size = window_size
        self.transform = transform
        
        # 对数据进行标准化
        self.scaler = MinMaxScaler()
        self.normalized_data = {}
        
        # 归一化OHLCV数据
        ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
        self.normalized_data['ohlcv'] = self.scaler.fit_transform(ohlcv)
        
        # 计算技术指标
        self._calculate_indicators()
        
        # 生成样本索引
        self.valid_indices = list(range(window_size, len(self.normalized_data['ohlcv'])))
    
    def _calculate_indicators(self):
        """计算常用技术指标"""
        data = self.data
        
        # 移动平均线
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema12'] - data['ema26']
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # 波动率
        data['volatility'] = data['close'].rolling(window=20).std()
        
        # 分析师使用的关键指标
        indicators = data[['ma5', 'ma10', 'ma20', 'rsi', 'macd', 'signal', 'volatility']].values
        # 处理NaN值
        indicators = np.nan_to_num(indicators, nan=0)
        
        # 归一化技术指标
        self.normalized_data['indicators'] = self.scaler.fit_transform(indicators)
    
    def __len__(self):
        """返回样本数量"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """获取样本"""
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        
        # 真实索引
        real_idx = self.valid_indices[idx]
        
        # 获取窗口数据
        ohlcv_window = self.normalized_data['ohlcv'][real_idx-self.window_size:real_idx]
        indicators_window = self.normalized_data['indicators'][real_idx-self.window_size:real_idx]
        
        # 组合输入特征
        X = np.concatenate([
            ohlcv_window, 
            indicators_window
        ], axis=1)
        
        if self.transform:
            X = self.transform(X)
        
        # 转为张量
        X = torch.tensor(X, dtype=torch.float32)
        
        # 标签是下一个收盘价变动方向
        next_close = self.data['close'].iloc[real_idx]
        current_close = self.data['close'].iloc[real_idx-1]
        y = 1 if next_close > current_close else 0
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y

class AttentionBlock(nn.Module):
    """注意力机制模块"""
    def __init__(self, input_dim, attention_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x形状: [batch, seq_len, features]
        attention_weights = self.attention(x)  # [batch, seq_len, 1]
        context_vector = torch.sum(x * attention_weights, dim=1)  # [batch, features]
        return context_vector, attention_weights

class DeepPatternRecognitionModel(nn.Module):
    """深度市场模式识别模型"""
    def __init__(self, input_channels=12, seq_length=50, num_classes=2):
        super(DeepPatternRecognitionModel, self).__init__()
        
        # 卷积层提取空间特征
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 批量归一化
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # LSTM提取时序特征
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = AttentionBlock(256, 64)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x形状: [batch, seq_len, features]
        batch_size, seq_len, features = x.size()
        
        # 转换为卷积输入格式 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # 卷积模块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 转换为LSTM输入格式 [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # LSTM层
        x, _ = self.lstm(x)
        
        # 注意力机制
        x, attention_weights = self.attention(x)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, attention_weights

class PatternRecognitionSystem:
    """市场模式识别系统"""
    def __init__(self, model_path=None):
        """
        初始化模式识别系统
        
        参数:
            model_path: 模型文件路径，如果提供则加载预训练模型
        """
        self.logger = logging.getLogger("PatternRecognitionSystem")
        self.logger.info("初始化深度市场模式识别系统...")
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        
        # 模型参数
        self.input_channels = 12  # OHLCV + 7个指标
        self.sequence_length = 50  # 50个时间步
        self.num_classes = 2      # 上涨/下跌
        
        # 初始化模型
        self.model = DeepPatternRecognitionModel(
            input_channels=self.input_channels,
            seq_length=self.sequence_length,
            num_classes=self.num_classes
        ).to(self.device)
        
        # 如果提供了模型路径，加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # 模式库
        self.pattern_library = self.initialize_pattern_library()
        
        self.logger.info("深度市场模式识别系统初始化完成")
    
    def initialize_pattern_library(self):
        """初始化模式库"""
        patterns = {
            PatternCategory.REVERSAL_BULLISH: {
                "description": "看涨反转模式",
                "examples": ["双底", "头肩底", "碗底", "V形反转", "岛底"],
                "features": {
                    "price_action": "先下跌，然后迅速反转上涨",
                    "volume": "在底部区域放大，确认反转",
                    "indicators": "RSI超卖后回升，MACD金叉"
                },
                "confidence": 0.0  # 动态更新的置信度
            },
            PatternCategory.REVERSAL_BEARISH: {
                "description": "看跌反转模式",
                "examples": ["双顶", "头肩顶", "碗顶", "M顶", "岛顶"],
                "features": {
                    "price_action": "先上涨，然后迅速反转下跌",
                    "volume": "在顶部区域放大，确认反转",
                    "indicators": "RSI超买后回落，MACD死叉"
                },
                "confidence": 0.0
            },
            PatternCategory.CONTINUATION_BULLISH: {
                "description": "看涨持续模式",
                "examples": ["旗形", "三角旗", "楔形上升", "矩形"],
                "features": {
                    "price_action": "在上升趋势中短暂盘整后继续上涨",
                    "volume": "盘整期间量能减弱，突破时放量",
                    "indicators": "均线多头排列，MACD高于零轴"
                },
                "confidence": 0.0
            },
            PatternCategory.CONTINUATION_BEARISH: {
                "description": "看跌持续模式",
                "examples": ["下降旗形", "下降三角旗", "楔形下降", "下降矩形"],
                "features": {
                    "price_action": "在下降趋势中短暂盘整后继续下跌",
                    "volume": "盘整期间量能减弱，突破时放量",
                    "indicators": "均线空头排列，MACD低于零轴"
                },
                "confidence": 0.0
            },
            PatternCategory.BREAKOUT_BULLISH: {
                "description": "看涨突破模式",
                "examples": ["突破阻力位", "突破下降趋势线", "间隙突破"],
                "features": {
                    "price_action": "价格突破重要阻力位或趋势线",
                    "volume": "突破时成交量明显放大",
                    "indicators": "动量指标快速上升"
                },
                "confidence": 0.0
            },
            PatternCategory.ENERGY_ACCUMULATION: {
                "description": "能量积累模式",
                "examples": ["压缩三角形", "窄幅盘整", "低波动率压缩"],
                "features": {
                    "price_action": "价格波动范围逐渐收窄",
                    "volume": "成交量逐渐萎缩",
                    "indicators": "波动率指标降低"
                },
                "confidence": 0.0
            },
            PatternCategory.QUANTUM_ANOMALY: {
                "description": "量子异常模式",
                "examples": ["闪崩", "跳空缺口", "V形异常反转"],
                "features": {
                    "price_action": "价格出现非常规走势，违背传统技术形态",
                    "volume": "成交量突然异常放大或萎缩",
                    "indicators": "指标出现异常背离"
                },
                "confidence": 0.0
            }
        }
        
        return patterns
    
    def load_model(self, model_path):
        """加载预训练模型"""
        try:
            self.logger.info(f"加载预训练模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("模型加载成功")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def save_model(self, model_path):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.logger.info(f"保存模型到: {model_path}")
            torch.save({
                'model_state_dict': self.model.state_dict(),
            }, model_path)
            self.logger.info("模型保存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
        
        返回:
            dict: 训练历史
        """
        self.logger.info("开始训练模型...")
        
        # 设置模型为训练模式
        self.model.train()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 最佳验证损失
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        # 训练循环
        for epoch in range(epochs):
            # 训练
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs, _ = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets.data).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # 验证
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # 前向传播
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # 统计
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets.data).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # 打印进度
            self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model(os.path.join("models", "best_pattern_recognition_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停: 验证损失 {early_stopping_patience} 轮没有改善")
                    break
            
            # 设回训练模式
            self.model.train()
        
        self.logger.info("模型训练完成")
        return history
    
    def recognize_pattern(self, price_data):
        """
        识别价格数据中的模式
        
        参数:
            price_data: DataFrame, 包含OHLCV数据
        
        返回:
            dict: 识别结果
        """
        self.logger.info("分析市场模式...")
        
        try:
            # 设置模型为评估模式
            self.model.eval()
            
            # 准备数据
            dataset = MarketPatternDataset(price_data, window_size=self.sequence_length)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            patterns = []
            attention_maps = []
            
            with torch.no_grad():
                for inputs, _ in dataloader:
                    inputs = inputs.to(self.device)
                    
                    # 前向传播
                    outputs, attention_weights = self.model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    # 获取预测和置信度
                    _, predicted = torch.max(outputs.data, 1)
                    confidence = probabilities[0][predicted.item()].item()
                    
                    # 收集注意力权重
                    attention_maps.append(attention_weights.cpu().numpy())
                    
                    # 模式识别逻辑
                    recognized_patterns = self._identify_specific_patterns(
                        price_data, confidence, attention_weights.cpu().numpy()
                    )
                    
                    patterns.extend(recognized_patterns)
            
            # 整合结果
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "recognized_patterns": patterns,
                "attention_maps": attention_maps
            }
            
            self.logger.info(f"识别到 {len(patterns)} 个市场模式")
            return result
            
        except Exception as e:
            self.logger.error(f"模式识别失败: {str(e)}")
            traceback.print_exc()
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "recognized_patterns": [],
                "error": str(e)
            }
    
    def _identify_specific_patterns(self, price_data, base_confidence, attention_weights):
        """
        识别特定的市场模式
        
        参数:
            price_data: DataFrame, 价格数据
            base_confidence: 基础置信度
            attention_weights: 注意力权重
        
        返回:
            list: 识别出的模式列表
        """
        # 在实际实现中，这里会有更复杂的逻辑来识别特定模式
        # 这里仅作为示例
        
        patterns = []
        
        # 分析最近的价格走势
        recent_data = price_data.iloc[-20:]
        close_prices = recent_data['close'].values
        opens = recent_data['open'].values
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        volumes = recent_data['volume'].values
        
        # 计算简单指标
        price_change = (close_prices[-1] / close_prices[0] - 1) * 100
        max_price = np.max(highs)
        min_price = np.min(lows)
        price_range = max_price - min_price
        
        # 通过简单规则识别基本模式
        if price_change > 5:
            # 强烈上涨
            if np.all(close_prices[-3:] > close_prices[-6:-3]):
                patterns.append({
                    "type": PatternCategory.CONTINUATION_BULLISH.name,
                    "confidence": min(0.8, base_confidence + 0.2),
                    "description": "强劲上升趋势确认",
                    "start_index": len(price_data) - 20,
                    "end_index": len(price_data) - 1
                })
        elif price_change < -5:
            # 强烈下跌
            if np.all(close_prices[-3:] < close_prices[-6:-3]):
                patterns.append({
                    "type": PatternCategory.CONTINUATION_BEARISH.name,
                    "confidence": min(0.8, base_confidence + 0.2),
                    "description": "强劲下降趋势确认",
                    "start_index": len(price_data) - 20,
                    "end_index": len(price_data) - 1
                })
        
        # 识别突破模式
        if close_prices[-1] > np.max(close_prices[:-1]) and volumes[-1] > np.mean(volumes[:-1]) * 1.5:
            patterns.append({
                "type": PatternCategory.BREAKOUT_BULLISH.name,
                "confidence": min(0.85, base_confidence + 0.3),
                "description": "伴随成交量放大的上行突破",
                "start_index": len(price_data) - 10,
                "end_index": len(price_data) - 1
            })
        
        # 识别能量积累
        recent_volatility = np.std(close_prices[-10:]) / np.mean(close_prices[-10:])
        older_volatility = np.std(close_prices[-20:-10]) / np.mean(close_prices[-20:-10])
        
        if recent_volatility < older_volatility * 0.7 and np.mean(volumes[-10:]) < np.mean(volumes[-20:-10]) * 0.8:
            patterns.append({
                "type": PatternCategory.ENERGY_ACCUMULATION.name,
                "confidence": min(0.75, base_confidence + 0.15),
                "description": "波动压缩与能量积累阶段",
                "start_index": len(price_data) - 20,
                "end_index": len(price_data) - 1
            })
        
        # 识别量子异常
        daily_changes = np.abs(close_prices[1:] / close_prices[:-1] - 1)
        avg_daily_change = np.mean(daily_changes)
        max_daily_change = np.max(daily_changes)
        
        if max_daily_change > avg_daily_change * 3:
            max_change_idx = np.argmax(daily_changes)
            patterns.append({
                "type": PatternCategory.QUANTUM_ANOMALY.name,
                "confidence": min(0.7, base_confidence + 0.1),
                "description": "价格异常波动，可能存在量子共振",
                "start_index": len(price_data) - 20 + max_change_idx,
                "end_index": len(price_data) - 20 + max_change_idx + 1
            })
        
        return patterns
    
    def visualize_pattern(self, price_data, pattern, save_path=None):
        """
        可视化识别出的模式
        
        参数:
            price_data: DataFrame, 价格数据
            pattern: dict, 识别出的模式
            save_path: str, 可选，保存路径
        """
        try:
            start_idx = pattern['start_index']
            end_idx = pattern['end_index']
            
            # 提取相关数据
            plot_data = price_data.iloc[start_idx:end_idx+1]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # 绘制K线图
            dates = plot_data.index
            
            # K线图
            for i in range(len(plot_data)):
                date = dates[i]
                open_price = plot_data['open'].iloc[i]
                close_price = plot_data['close'].iloc[i]
                high_price = plot_data['high'].iloc[i]
                low_price = plot_data['low'].iloc[i]
                
                color = 'green' if close_price >= open_price else 'red'
                
                # 绘制实体
                ax1.plot([i, i], [low_price, high_price], color=color)
                ax1.plot([i, i], [open_price, close_price], color=color, linewidth=5)
            
            # 添加移动平均线
            if 'ma5' in plot_data.columns:
                ax1.plot(plot_data['ma5'], color='blue', label='MA5')
            if 'ma10' in plot_data.columns:
                ax1.plot(plot_data['ma10'], color='orange', label='MA10')
            if 'ma20' in plot_data.columns:
                ax1.plot(plot_data['ma20'], color='purple', label='MA20')
            
            # 添加标题和标签
            ax1.set_title(f"识别出的模式: {pattern['type']} (置信度: {pattern['confidence']:.2f})")
            ax1.set_ylabel('价格')
            ax1.legend()
            
            # 在下方绘制成交量
            ax2.bar(range(len(plot_data)), plot_data['volume'], color='blue', alpha=0.5)
            ax2.set_ylabel('成交量')
            ax2.set_xlabel('日期')
            
            # 添加描述
            fig.text(0.5, 0.01, pattern['description'], ha='center', fontsize=12)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"图表已保存到: {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"可视化模式失败: {str(e)}")
            traceback.print_exc()

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建模式识别系统
    pattern_system = PatternRecognitionSystem()
    
    # 输出诊断信息
    print("\n" + "="*60)
    print("超神量子共生系统 - 深度市场模式识别引擎")
    print("="*60 + "\n")
    
    print("模式库包含以下类型:")
    for pattern_type, info in pattern_system.pattern_library.items():
        print(f"- {pattern_type.name}: {info['description']}")
        print(f"  示例: {', '.join(info['examples'])}")
    
    print("\n系统准备就绪，可以开始识别市场模式。")
    print("="*60 + "\n") 