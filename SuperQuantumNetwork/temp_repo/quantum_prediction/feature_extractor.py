#!/usr/bin/env python3
"""
特征提取器 - 超神系统量子预测的特征处理

提取市场数据中的关键特征，为预测模型提供输入
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class FeatureExtractor:
    """特征提取器类
    
    从市场数据中提取和处理关键特征，为预测模型提供输入
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化特征提取器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger("FeatureExtractor")
        
        # 默认配置
        self.default_config = {
            "window_sizes": [5, 10, 20, 50],      # 滑动窗口大小
            "use_momentum": True,                  # 使用动量指标
            "use_volatility": True,                # 使用波动性指标
            "use_trend": True,                     # 使用趋势指标
            "use_volume": True,                    # 使用交易量指标
            "normalize_features": True,            # 规范化特征
            "quantum_feature_extraction": False,   # 量子特征提取 (高级)
            "minimum_history_length": 50           # 最小历史数据长度
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        self.logger.info("特征提取器初始化完成")
    
    def extract_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取市场特征
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]: 提取的特征
        """
        self.logger.debug("开始提取市场特征")
        
        # 验证市场数据
        if not self._validate_market_data(market_data):
            self.logger.warning("市场数据验证失败")
            return {"error": "invalid_market_data"}
            
        # 提取基本特征
        features = self._extract_basic_features(market_data)
        
        # 提取技术指标
        if "price_history" in market_data:
            technical_features = self._extract_technical_indicators(market_data)
            features.update(technical_features)
            
        # 提取模式特征
        pattern_features = self._extract_pattern_features(market_data)
        features.update(pattern_features)
        
        # 规范化特征
        if self.config["normalize_features"]:
            features = self._normalize_features(features)
            
        # 添加元数据
        features["extraction_timestamp"] = datetime.now().isoformat()
        features["extractor_version"] = "1.0.0"
        
        self.logger.debug(f"特征提取完成，提取了 {len(features)} 个特征")
        return features
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """验证市场数据
        
        Args:
            market_data: 市场数据
            
        Returns:
            bool: 数据是否有效
        """
        # 检查必要字段
        required_fields = ["symbol", "price"]
        for field in required_fields:
            if field not in market_data:
                self.logger.warning(f"市场数据缺少必要字段: {field}")
                return False
                
        # 检查价格有效性
        if market_data["price"] <= 0:
            self.logger.warning(f"价格无效: {market_data['price']}")
            return False
            
        # 检查价格历史（如果有）
        if "price_history" in market_data:
            history = market_data["price_history"]
            
            # 检查历史数据类型
            if not isinstance(history, (list, np.ndarray)):
                self.logger.warning("价格历史数据类型无效")
                return False
                
            # 检查历史数据长度
            if len(history) < self.config["minimum_history_length"]:
                self.logger.warning(f"价格历史数据长度不足: {len(history)} < {self.config['minimum_history_length']}")
                return False
        
        return True
    
    def _extract_basic_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取基本特征
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]: 基本特征
        """
        features = {}
        
        # 提取当前价格
        features["current_price"] = market_data["price"]
        
        # 提取交易量（如果有）
        if "volume" in market_data:
            features["volume"] = market_data["volume"]
            
        # 提取时间特征
        if "timestamp" in market_data:
            try:
                # 解析时间戳
                if isinstance(market_data["timestamp"], (int, float)):
                    dt = datetime.fromtimestamp(market_data["timestamp"])
                elif isinstance(market_data["timestamp"], str):
                    dt = datetime.fromisoformat(market_data["timestamp"])
                else:
                    dt = market_data["timestamp"]
                
                # 提取时间组件
                features["hour_of_day"] = dt.hour
                features["day_of_week"] = dt.weekday()
                features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
            except (ValueError, TypeError):
                pass
                
        # 提取其他基本信息
        for key in ["spread", "ask", "bid", "market_cap", "volatility"]:
            if key in market_data:
                features[key] = market_data[key]
                
        return features
    
    def _extract_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取技术指标特征
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]:.技术指标特征
        """
        features = {}
        
        # 获取价格历史
        price_history = np.array(market_data["price_history"])
        
        # 计算移动平均线
        for window in self.config["window_sizes"]:
            if len(price_history) >= window:
                ma = np.mean(price_history[-window:])
                features[f"ma_{window}"] = ma
                
                # 当前价格与移动平均线的关系
                features[f"price_ma_{window}_ratio"] = market_data["price"] / ma
                
        # 计算波动性指标
        if self.config["use_volatility"] and len(price_history) >= 20:
            # 标准差方法
            features["volatility_std"] = np.std(price_history[-20:]) / np.mean(price_history[-20:])
            
            # ATR近似 (平均真实范围)
            if len(price_history) >= 21:
                ranges = []
                for i in range(1, 20):
                    true_range = abs(price_history[-i] - price_history[-i-1])
                    ranges.append(true_range)
                features["volatility_atr"] = np.mean(ranges) / price_history[-1]
                
        # 计算动量指标
        if self.config["use_momentum"] and len(price_history) >= 20:
            # 10天动量
            features["momentum_10"] = price_history[-1] / price_history[-10] - 1
            
            # 20天动量
            features["momentum_20"] = price_history[-1] / price_history[-20] - 1
            
            # RSI近似
            diff = np.diff(price_history[-15:])
            up = np.sum(np.clip(diff, 0, None))
            down = -np.sum(np.clip(diff, None, 0))
            
            if up + down != 0:
                features["rsi_14"] = up / (up + down) * 100
            else:
                features["rsi_14"] = 50
                
        # 计算趋势指标
        if self.config["use_trend"] and len(price_history) >= 50:
            # 短期趋势 (10天)
            short_term = np.polyfit(range(10), price_history[-10:], 1)[0]
            features["trend_short"] = short_term / price_history[-1]
            
            # 中期趋势 (30天)
            if len(price_history) >= 30:
                mid_term = np.polyfit(range(30), price_history[-30:], 1)[0]
                features["trend_mid"] = mid_term / price_history[-1]
            
            # 长期趋势 (50天)
            long_term = np.polyfit(range(50), price_history[-50:], 1)[0]
            features["trend_long"] = long_term / price_history[-1]
            
            # 趋势一致性
            if "trend_mid" in features:
                features["trend_alignment"] = np.sign(features["trend_short"]) == np.sign(features["trend_mid"]) == np.sign(features["trend_long"])
                
        # 提取交易量特征
        if self.config["use_volume"] and "volume_history" in market_data:
            volume_history = np.array(market_data["volume_history"])
            
            if len(volume_history) >= 10:
                # 平均交易量
                features["volume_ma_10"] = np.mean(volume_history[-10:])
                
                # 交易量趋势
                features["volume_trend"] = np.polyfit(range(10), volume_history[-10:], 1)[0] / features["volume_ma_10"]
                
                # 交易量与价格关系
                if len(price_history) >= 10:
                    price_diff = np.diff(price_history[-11:])
                    volume_price_corr = np.corrcoef(volume_history[-10:], np.abs(price_diff))[0, 1]
                    features["volume_price_correlation"] = volume_price_corr
                    
        return features
    
    def _extract_pattern_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取模式特征
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]: 模式特征
        """
        features = {}
        
        # 检查是否有价格历史
        if "price_history" not in market_data or len(market_data["price_history"]) < 20:
            return features
            
        price_history = np.array(market_data["price_history"])
        
        # 检测价格反转模式
        if len(price_history) >= 20:
            # 最近5天的价格变化
            recent_changes = np.diff(price_history[-6:])
            
            # 检测V形反转 (下跌后上涨)
            if np.all(recent_changes[:2] < 0) and np.all(recent_changes[2:] > 0):
                features["pattern_v_reversal"] = 1
            else:
                features["pattern_v_reversal"] = 0
                
            # 检测倒V形反转 (上涨后下跌)
            if np.all(recent_changes[:2] > 0) and np.all(recent_changes[2:] < 0):
                features["pattern_inverted_v_reversal"] = 1
            else:
                features["pattern_inverted_v_reversal"] = 0
                
            # 检测价格突破 (突破前期高点)
            if len(price_history) >= 50:
                prev_high = np.max(price_history[-50:-5])
                if price_history[-1] > prev_high * 1.05:  # 突破5%
                    features["pattern_breakout"] = 1
                else:
                    features["pattern_breakout"] = 0
                    
        # 检测价格运行状态
        if len(price_history) >= 30:
            # 波动区间检测
            upper = np.max(price_history[-30:])
            lower = np.min(price_history[-30:])
            current = price_history[-1]
            
            # 价格范围占比
            price_range = (upper - lower) / lower
            
            # 当前价格在区间中的位置
            if upper > lower:
                range_position = (current - lower) / (upper - lower)
                features["price_range_position"] = range_position
                
                # 接近上轨/下轨判断
                if range_position > 0.8:
                    features["near_upper_band"] = 1
                elif range_position < 0.2:
                    features["near_lower_band"] = 1
                    
        # 检测频道/通道
        if len(price_history) >= 50:
            # 计算线性趋势线
            x = np.arange(50)
            slope, intercept = np.polyfit(x, price_history[-50:], 1)
            
            # 计算趋势线
            trend_line = slope * x + intercept
            
            # 计算价格与趋势线的偏差
            deviations = price_history[-50:] - trend_line
            std_dev = np.std(deviations)
            
            # 计算当前价格与趋势线的偏差 (标准差单位)
            current_dev = (price_history[-1] - (slope * 49 + intercept)) / std_dev
            features["trend_deviation"] = current_dev
            
            # 判断是否突破通道
            if abs(current_dev) > 2:
                features["channel_breakout"] = np.sign(current_dev)
            else:
                features["channel_breakout"] = 0
                
        return features
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """规范化特征
        
        Args:
            features: 原始特征
            
        Returns:
            Dict[str, Any]: 规范化后的特征
        """
        normalized = {}
        
        # 保留元数据和分类特征
        categorical_keys = [
            "hour_of_day", "day_of_week", "is_weekend", 
            "pattern_v_reversal", "pattern_inverted_v_reversal", 
            "pattern_breakout", "near_upper_band", "near_lower_band",
            "channel_breakout", "trend_alignment"
        ]
        
        # 保留这些键不进行规范化
        for key in categorical_keys:
            if key in features:
                normalized[key] = features[key]
                
        # 复制错误信息（如果有）
        if "error" in features:
            normalized["error"] = features["error"]
            return normalized
            
        # 规范化数值特征
        for key, value in features.items():
            # 跳过已处理的键和非数值类型
            if key in normalized or not isinstance(value, (int, float)):
                continue
                
            # 规范化方法取决于特征类型
            if "ratio" in key or "correlation" in key:
                # 比率类特征通常已经规范化
                normalized[key] = value
            elif "trend" in key:
                # 趋势特征可能需要限制在合理范围内
                normalized[key] = max(-1.0, min(1.0, value))
            elif "volatility" in key:
                # 波动性特征通常为正值，限制上限
                normalized[key] = max(0.0, min(1.0, value))
            elif "rsi" in key:
                # RSI已在0-100范围内
                normalized[key] = value / 100.0
            elif "price" in key and key != "current_price":
                # 价格相关特征，如与MA的比率
                normalized[key] = value
            else:
                # 其他数值特征尝试规范化到0-1范围
                # 这里使用启发式方法，实际应用中可能需要更复杂的规范化
                if abs(value) > 1e-10:
                    normalized[key] = value
                else:
                    normalized[key] = value
                    
        return normalized
    
    def extract_quantum_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取量子增强特征 (高级方法)
        
        Args:
            market_data: 市场数据
            
        Returns:
            Dict[str, Any]: 量子特征
        """
        # 首先提取常规特征
        features = self.extract_features(market_data)
        
        # 检查是否启用量子特征提取
        if not self.config["quantum_feature_extraction"]:
            features["quantum_features_enabled"] = False
            return features
            
        # 尝试导入量子核心
        try:
            from quantum_core.quantum_engine import QuantumEngine
            quantum_available = True
        except ImportError:
            self.logger.warning("量子核心不可用，无法提取量子特征")
            quantum_available = False
            
        # 如果量子核心不可用，返回常规特征
        if not quantum_available:
            features["quantum_features_enabled"] = False
            return features
            
        # 创建量子引擎实例
        try:
            quantum_engine = QuantumEngine(dimensions=8)
            quantum_engine.start()
            
            # 准备量子计算任务
            if "price_history" in market_data and len(market_data["price_history"]) >= 20:
                # 使用最近的价格历史数据
                recent_prices = market_data["price_history"][-20:]
                
                # 创建量子特征提取任务
                feature_task = {
                    "type": "feature_extraction",
                    "data": recent_prices,
                    "extraction_params": {
                        "entanglement_level": 0.7,
                        "interference_pattern": True,
                        "quantum_fourier": True
                    }
                }
                
                # 提交任务并获取结果
                task_id = quantum_engine.submit_calculation(feature_task, priority=0.8)
                result = quantum_engine.get_result(task_id, timeout=5.0)
                
                if result and "quantum_features" in result:
                    # 整合量子特征
                    quantum_features = result["quantum_features"]
                    for key, value in quantum_features.items():
                        features[f"quantum_{key}"] = value
                        
                    features["quantum_features_enabled"] = True
                else:
                    features["quantum_features_enabled"] = False
                    
            # 停止量子引擎
            quantum_engine.stop()
            
        except Exception as e:
            self.logger.error(f"量子特征提取失败: {str(e)}")
            features["quantum_features_enabled"] = False
            features["quantum_error"] = str(e)
            
        return features