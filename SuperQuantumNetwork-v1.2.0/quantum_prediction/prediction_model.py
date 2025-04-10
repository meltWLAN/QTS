#!/usr/bin/env python3
"""
预测模型 - 超神系统量子预测的模型实现

提供各种预测算法模型实现和数据处理能力
"""

import logging
import random
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

class PredictionModel:
    """预测模型类
    
    实现各种预测算法，用于市场预测和趋势分析
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """初始化预测模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        self.logger = logging.getLogger(f"PredictionModel.{name}")
        self.logger.info(f"初始化预测模型: {name}")
        
        # 模型基本信息
        self.name = name
        self.version = "1.0.0"
        self.created_at = datetime.now()
        
        # 默认配置
        self.default_config = {
            "confidence_threshold": 0.6,
            "calibration_factor": 0.8,
            "learning_rate": 0.05,
            "memory_length": 24,
            "use_ensemble": True,
            "noise_reduction": 0.3
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 模型状态
        self.state = {
            "calibration_level": 0.7,
            "accuracy": 0.0,
            "predictions_made": 0,
            "last_prediction": None,
            "last_update": datetime.now()
        }
        
        # 模型内存 - 保存近期数据和预测
        self.memory = {
            "recent_data": [],
            "recent_predictions": [],
            "error_history": []
        }
        
        # 特定于模型类型的初始化
        self._initialize_model_specific()
        
        self.logger.info(f"预测模型初始化完成: {name}")
        
    def _initialize_model_specific(self):
        """初始化特定模型类型的参数和方法"""
        # 根据模型名称初始化特定参数
        if self.name == "quantum_wave":
            # 量子波模型特定参数
            self.model_params = {
                "wave_amplitude": 0.7,
                "phase_shift": 0.2,
                "interference_factor": 0.5,
                "wavefront_sensitivity": self.config.get("wavefront_sensitivity", 0.8),
                "dimension_count": self.config.get("dimensions", 12)
            }
            
        elif self.name == "timeline_confluence":
            # 时间线汇聚模型特定参数
            self.model_params = {
                "timeline_count": self.config.get("parallel_timelines", 7),
                "divergence_weight": self.config.get("divergence_weight", 0.4),
                "confluence_threshold": self.config.get("confluence_threshold", 0.65),
                "temporal_momentum": self.config.get("temporal_momentum", 0.6),
                "butterfly_effect": 0.3
            }
            
        elif self.name == "entropic_analysis":
            # 熵分析模型特定参数
            self.model_params = {
                "entropy_sensitivity": self.config.get("entropy_sensitivity", 0.75),
                "chaos_damping": self.config.get("chaos_damping", 0.3),
                "order_emergence_factor": self.config.get("order_emergence_factor", 0.6),
                "pattern_recognition_threshold": self.config.get("pattern_recognition_threshold", 0.55),
                "fractal_depth": 4
            }
            
        elif self.name == "quantum_field_analysis":
            # 量子场分析模型特定参数
            self.model_params = {
                "field_dimensions": self.config.get("field_dimensions", 5),
                "field_strength": self.config.get("field_strength", 0.7),
                "interaction_depth": self.config.get("interaction_depth", 3),
                "coherence_threshold": self.config.get("coherence_threshold", 0.6),
                "field_decay_rate": 0.05
            }
            
        elif self.name == "probability_wave_collapse":
            # 概率波坍缩模型特定参数
            self.model_params = {
                "wave_count": self.config.get("wave_count", 12),
                "collapse_threshold": self.config.get("collapse_threshold", 0.7),
                "observer_effect": self.config.get("observer_effect", 0.2),
                "quantum_tunneling": self.config.get("quantum_tunneling", True),
                "superposition_states": 8
            }
            
        else:
            # 通用模型参数
            self.model_params = {
                "complexity": 0.5,
                "adaptability": 0.7,
                "learning_factor": 0.3,
                "pattern_sensitivity": 0.6
            }
    
    def predict(self, market_data: Dict[str, Any], 
                horizon: int = 10) -> Dict[str, Any]:
        """生成市场预测
        
        Args:
            market_data: 市场数据
            horizon: 预测周期（分钟）
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        self.logger.debug(f"模型 {self.name} 开始生成预测，周期: {horizon}分钟")
        
        # 更新模型内存
        self._update_memory(market_data)
        
        # 根据模型类型选择预测方法
        if self.name == "quantum_wave":
            prediction, confidence = self._predict_quantum_wave(market_data, horizon)
            
        elif self.name == "timeline_confluence":
            prediction, confidence = self._predict_timeline_confluence(market_data, horizon)
            
        elif self.name == "entropic_analysis":
            prediction, confidence = self._predict_entropic_analysis(market_data, horizon)
            
        elif self.name == "quantum_field_analysis":
            prediction, confidence = self._predict_quantum_field(market_data, horizon)
            
        elif self.name == "probability_wave_collapse":
            prediction, confidence = self._predict_probability_collapse(market_data, horizon)
            
        else:
            # 默认预测方法（通用模型）
            prediction, confidence = self._predict_generic(market_data, horizon)
        
        # 应用校准因子
        confidence = self._calibrate_confidence(confidence)
        
        # 更新模型状态
        self.state["predictions_made"] += 1
        self.state["last_prediction"] = {
            "timestamp": datetime.now(),
            "horizon": horizon,
            "prediction": prediction,
            "confidence": confidence
        }
        
        # 记录预测结果到内存
        prediction_record = {
            "timestamp": datetime.now(),
            "horizon": horizon,
            "market_data": {k: v for k, v in market_data.items() if k != "price_history"},
            "prediction": prediction,
            "confidence": confidence
        }
        self.memory["recent_predictions"].append(prediction_record)
        
        # 限制内存大小
        max_memory = self.config.get("memory_length", 24)
        if len(self.memory["recent_predictions"]) > max_memory:
            self.memory["recent_predictions"] = self.memory["recent_predictions"][-max_memory:]
        
        # 构建结果
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "model": self.name,
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon
        }
        
        self.logger.debug(f"模型 {self.name} 预测完成，置信度: {confidence:.2f}")
        return result
    
    def _predict_quantum_wave(self, market_data: Dict[str, Any], 
                            horizon: int) -> Tuple[Dict[str, Any], float]:
        """量子波预测方法
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 提取波动性指标
        volatility = market_data.get("volatility", 0.01)
        
        # 获取量子波参数
        amplitude = self.model_params["wave_amplitude"]
        phase_shift = self.model_params["phase_shift"]
        sensitivity = self.model_params["wavefront_sensitivity"]
        
        # 生成时间因子 (基于预测周期)
        time_factor = horizon / 60.0  # 转换为小时
        
        # 计算波动分量
        wave_component = amplitude * np.sin(phase_shift + time_factor * np.pi)
        
        # 应用随机量子涨落
        quantum_fluctuation = np.random.normal(0, volatility * sensitivity)
        
        # 计算价格变化百分比
        price_change_pct = wave_component * (0.01 + volatility) + quantum_fluctuation
        
        # 计算预测价格
        predicted_price = current_price * (1 + price_change_pct)
        
        # 计算趋势方向 (-1到1)
        trend = np.sign(price_change_pct) * min(1.0, abs(price_change_pct) * 20)
        
        # 预测置信度计算
        # 基于波幅、相位和量子涨落计算
        phase_confidence = 0.7 + 0.3 * np.cos(phase_shift)
        
        # 波动性反比信心度
        volatility_factor = max(0.3, 1.0 - volatility * 5)
        
        # 最终置信度
        confidence = (
            phase_confidence * 0.5 + 
            volatility_factor * 0.3 + 
            sensitivity * 0.2
        )
        
        # 构建预测结果
        prediction = {
            "price": predicted_price,
            "trend": trend,
            "volatility": volatility * (1 + 0.2 * np.random.randn())
        }
        
        return prediction, confidence
    
    def _predict_timeline_confluence(self, market_data: Dict[str, Any], 
                                   horizon: int) -> Tuple[Dict[str, Any], float]:
        """时间线汇聚预测方法
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 提取波动性指标
        volatility = market_data.get("volatility", 0.01)
        
        # 获取时间线参数
        timeline_count = self.model_params["timeline_count"]
        divergence_weight = self.model_params["divergence_weight"]
        confluence_threshold = self.model_params["confluence_threshold"]
        temporal_momentum = self.model_params["temporal_momentum"]
        
        # 生成多条时间线的价格预测
        timeline_prices = []
        for i in range(timeline_count):
            # 每条时间线应用不同的随机偏移
            timeline_seed = int(time.time()) + i
            np.random.seed(timeline_seed)
            
            # 计算时间线特有的偏移
            timeline_bias = np.random.normal(0, volatility * divergence_weight)
            
            # 应用时间动量
            momentum_factor = np.random.uniform(0.8, 1.2) * temporal_momentum
            
            # 获取历史动量（如果有）
            if "trend" in market_data:
                historical_momentum = market_data["trend"] * momentum_factor
            else:
                historical_momentum = 0
                
            # 计算此时间线的价格变化
            timeline_change = (
                historical_momentum * 0.01 + 
                timeline_bias * (horizon / 30.0)  # 按比例缩放变化量
            )
            
            # 计算此时间线的预测价格
            timeline_price = current_price * (1 + timeline_change)
            timeline_prices.append(timeline_price)
        
        # 分析时间线汇聚情况
        mean_price = np.mean(timeline_prices)
        std_price = np.std(timeline_prices)
        
        # 计算汇聚度（标准差的倒数，标准化到0-1）
        if std_price > 0:
            confluence = min(1.0, 1.0 / (std_price / mean_price * 10))
        else:
            confluence = 1.0  # 完全汇聚
            
        # 确定主导时间线 (最接近平均值的时间线)
        closest_idx = np.argmin([abs(p - mean_price) for p in timeline_prices])
        dominant_price = timeline_prices[closest_idx]
        
        # 计算趋势
        price_change_pct = (dominant_price - current_price) / current_price
        trend = np.sign(price_change_pct) * min(1.0, abs(price_change_pct) * 20)
        
        # 计算预测置信度
        # 基于时间线汇聚程度
        if confluence > confluence_threshold:
            base_confidence = 0.5 + 0.5 * (confluence - confluence_threshold) / (1 - confluence_threshold)
        else:
            base_confidence = 0.5 * confluence / confluence_threshold
            
        # 调整置信度
        confidence = base_confidence * (0.8 + 0.2 * temporal_momentum)
        
        # 构建预测结果
        prediction = {
            "price": dominant_price,
            "trend": trend,
            "volatility": volatility * (1 + std_price / mean_price),
            "timeline_divergence": 1.0 - confluence
        }
        
        return prediction, confidence
    
    def _predict_entropic_analysis(self, market_data: Dict[str, Any], 
                                 horizon: int) -> Tuple[Dict[str, Any], float]:
        """熵分析预测方法
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 获取模型参数
        entropy_sensitivity = self.model_params["entropy_sensitivity"]
        chaos_damping = self.model_params["chaos_damping"]
        order_factor = self.model_params["order_emergence_factor"]
        pattern_threshold = self.model_params["pattern_recognition_threshold"]
        
        # 计算市场熵值
        # 如果有价格历史数据，则使用它计算熵
        if "price_history" in market_data and len(market_data["price_history"]) > 5:
            price_history = market_data["price_history"]
            # 计算价格变化率
            changes = [price_history[i] / price_history[i-1] - 1 for i in range(1, len(price_history))]
            
            # 计算变化率的标准偏差作为熵的近似值
            entropy = min(1.0, np.std(changes) * 10)
        else:
            # 没有价格历史，使用默认熵值或波动性
            entropy = market_data.get("volatility", 0.2)
        
        # 应用熵敏感度
        adjusted_entropy = entropy * entropy_sensitivity
        
        # 计算确定性因子 (熵的反函数)
        determinism = 1.0 - adjusted_entropy
        
        # 模拟市场的有序性出现
        if determinism > pattern_threshold:
            # 熵很低，系统较有序，预测更准确
            pattern_strength = (determinism - pattern_threshold) / (1 - pattern_threshold)
            pattern_direction = np.sign(np.random.randn()) if "trend" not in market_data else np.sign(market_data["trend"])
            
            # 基于出现的秩序确定价格变化方向
            price_change = pattern_direction * pattern_strength * order_factor * (horizon / 60.0)
        else:
            # 熵很高，系统较混沌，预测随机性更大
            # 但应用阻尼因子减轻混沌效应
            random_factor = np.random.normal(0, adjusted_entropy * (1 - chaos_damping))
            price_change = random_factor * (horizon / 60.0)
        
        # 计算预测价格
        predicted_price = current_price * (1 + price_change)
        
        # 计算趋势
        trend = np.sign(price_change) * min(1.0, abs(price_change) * 20)
        
        # 计算预测置信度
        # 基于熵和模式识别
        confidence = (
            determinism * 0.7 +  # 低熵高确定性给予更高权重
            (order_factor if determinism > pattern_threshold else 0.3) * 0.3
        )
        
        # 构建预测结果
        prediction = {
            "price": predicted_price,
            "trend": trend,
            "volatility": entropy * (1 + 0.2 * np.random.randn()),
            "entropy": entropy,
            "determinism": determinism
        }
        
        return prediction, confidence
    
    def _predict_quantum_field(self, market_data: Dict[str, Any], 
                              horizon: int) -> Tuple[Dict[str, Any], float]:
        """量子场分析预测方法
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 获取模型参数
        field_dimensions = self.model_params["field_dimensions"]
        field_strength = self.model_params["field_strength"]
        interaction_depth = self.model_params["interaction_depth"]
        coherence_threshold = self.model_params["coherence_threshold"]
        
        # 生成量子场 (多维随机场)
        quantum_field = np.random.randn(field_dimensions)
        field_norm = np.linalg.norm(quantum_field)
        quantum_field = quantum_field / field_norm * field_strength
        
        # 模拟场交互
        field_interactions = []
        for _ in range(interaction_depth):
            # 生成随机交互矩阵
            interaction_matrix = np.random.randn(field_dimensions, field_dimensions)
            # 归一化矩阵
            interaction_matrix = interaction_matrix / np.linalg.norm(interaction_matrix)
            
            # 应用交互
            field_interaction = np.dot(interaction_matrix, quantum_field)
            field_interactions.append(field_interaction)
            
            # 更新场
            quantum_field = 0.7 * quantum_field + 0.3 * field_interaction
        
        # 计算场相干性
        field_coherence = 1.0 - np.std(quantum_field)
        
        # 提取价格影响因子
        # 使用场强度的平均值作为价格变化的基础
        base_price_factor = np.mean(quantum_field)
        
        # 应用时间比例
        time_scale = horizon / 60.0  # 转为小时
        
        # 计算价格变化
        price_change = base_price_factor * time_scale * 0.05  # 缩放到合理范围
        
        # 计算预测价格
        predicted_price = current_price * (1 + price_change)
        
        # 计算趋势
        trend = np.sign(price_change) * min(1.0, abs(price_change) * 20)
        
        # 计算预测置信度
        # 基于场相干性和维度
        if field_coherence > coherence_threshold:
            # 高相干性，更高置信度
            coherence_confidence = 0.5 + 0.5 * (field_coherence - coherence_threshold) / (1 - coherence_threshold)
        else:
            # 低相干性，较低置信度
            coherence_confidence = 0.5 * field_coherence / coherence_threshold
            
        # 维度因子
        dimension_factor = min(1.0, field_dimensions / 10.0)
        
        # 最终置信度
        confidence = (
            coherence_confidence * 0.7 +
            dimension_factor * 0.3
        )
        
        # 构建预测结果
        prediction = {
            "price": predicted_price,
            "trend": trend,
            "volatility": 0.01 * (1 + (1 - field_coherence) * 5),
            "field_coherence": field_coherence
        }
        
        return prediction, confidence
    
    def _predict_probability_collapse(self, market_data: Dict[str, Any], 
                                    horizon: int) -> Tuple[Dict[str, Any], float]:
        """概率波坍缩预测方法
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 获取模型参数
        wave_count = self.model_params["wave_count"]
        collapse_threshold = self.model_params["collapse_threshold"]
        observer_effect = self.model_params["observer_effect"]
        quantum_tunneling = self.model_params["quantum_tunneling"]
        
        # 生成多个概率波（表示不同可能的价格轨迹）
        probability_waves = []
        wave_amplitudes = []
        
        # 当前趋势
        current_trend = market_data.get("trend", 0)
        
        for i in range(wave_count):
            # 每个波具有不同的振幅和方向
            wave_seed = int(time.time()) + i * 1000
            np.random.seed(wave_seed)
            
            # 波振幅（影响价格变化幅度）
            amplitude = np.random.uniform(0.2, 1.0)
            wave_amplitudes.append(amplitude)
            
            # 波方向（影响价格变化方向）
            # 有一定概率跟随当前趋势
            if np.random.random() < 0.6 + 0.2 * abs(current_trend):
                direction = np.sign(current_trend) if current_trend != 0 else np.sign(np.random.randn())
            else:
                direction = np.sign(np.random.randn())
                
            # 计算波的价格变化
            wave_change = direction * amplitude * 0.02 * (horizon / 60.0)
            
            # 应用量子隧穿效应（如果启用）
            if quantum_tunneling and np.random.random() < 0.15:
                # 有小概率出现隧穿效应，导致更大变化
                tunnel_factor = np.random.uniform(1.5, 3.0)
                wave_change *= tunnel_factor
            
            # 计算此波的预测价格
            wave_price = current_price * (1 + wave_change)
            probability_waves.append(wave_price)
        
        # 模拟观察者效应（测量导致波函数坍缩）
        # 通过权重随机选择一个波进行坍缩
        observation_weights = np.array(wave_amplitudes) ** observer_effect
        observation_weights = observation_weights / np.sum(observation_weights)
        
        # 使用加权随机选择波坍缩位置
        collapsed_idx = np.random.choice(range(wave_count), p=observation_weights)
        collapsed_price = probability_waves[collapsed_idx]
        
        # 计算坍缩强度
        # 检查有多少波靠近坍缩位置
        collapse_band = abs(current_price * 0.01)  # 1%范围
        waves_near_collapse = sum(1 for p in probability_waves 
                                 if abs(p - collapsed_price) < collapse_band)
        
        collapse_strength = waves_near_collapse / wave_count
        
        # 计算趋势
        price_change_pct = (collapsed_price - current_price) / current_price
        trend = np.sign(price_change_pct) * min(1.0, abs(price_change_pct) * 20)
        
        # 计算预测置信度
        # 基于坍缩强度
        if collapse_strength > collapse_threshold:
            # 多个波坍缩在同一区域，较高置信度
            base_confidence = 0.5 + 0.5 * (collapse_strength - collapse_threshold) / (1 - collapse_threshold)
        else:
            # 波分散，较低置信度
            base_confidence = 0.5 * collapse_strength / collapse_threshold
            
        # 应用观察者效应调整
        confidence = base_confidence * (1 - observer_effect * 0.5)
        
        # 构建预测结果
        prediction = {
            "price": collapsed_price,
            "trend": trend,
            "volatility": 0.01 * (1 + (1 - collapse_strength) * 3),
            "collapse_strength": collapse_strength
        }
        
        return prediction, confidence
    
    def _predict_generic(self, market_data: Dict[str, Any], 
                        horizon: int) -> Tuple[Dict[str, Any], float]:
        """通用预测方法
        
        用于未指定特殊模型的情况
        
        Args:
            market_data: 市场数据
            horizon: 预测周期
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 提取当前价格
        current_price = market_data.get("price", 0)
        
        # 提取或设置波动性
        volatility = market_data.get("volatility", 0.01)
        
        # 提取当前趋势
        current_trend = market_data.get("trend", 0)
        
        # 获取模型参数
        complexity = self.model_params["complexity"]
        adaptability = self.model_params["adaptability"]
        pattern_sensitivity = self.model_params["pattern_sensitivity"]
        
        # 计算预测变化
        # 基础随机变化
        random_change = np.random.normal(0, volatility)
        
        # 趋势延续效应
        trend_factor = current_trend * adaptability * (horizon / 60.0)
        
        # 应用复杂性调整
        complexity_adjustment = np.random.normal(0, complexity * 0.01)
        
        # 总体价格变化
        price_change = random_change + trend_factor + complexity_adjustment
        
        # 计算预测价格
        predicted_price = current_price * (1 + price_change)
        
        # 计算新趋势
        new_trend = np.sign(price_change) * min(1.0, abs(price_change) * 20)
        
        # 计算趋势变化（用于置信度计算）
        trend_shift = abs(new_trend - current_trend)
        
        # 计算预测置信度
        # 反比于波动性和趋势变化
        confidence = max(0.3, 1.0 - (volatility * 5 + trend_shift * 0.2))
        
        # 应用模式敏感度
        confidence *= (0.7 + 0.3 * pattern_sensitivity)
        
        # 构建预测结果
        prediction = {
            "price": predicted_price,
            "trend": new_trend,
            "volatility": volatility * (1 + 0.2 * np.random.randn())
        }
        
        return prediction, confidence
    
    def _update_memory(self, market_data: Dict[str, Any]):
        """更新模型内存
        
        保存最近的市场数据
        
        Args:
            market_data: 市场数据
        """
        # 创建内存记录
        memory_record = {
            "timestamp": datetime.now(),
            "data": {k: v for k, v in market_data.items() if k != "price_history"}
        }
        
        # 添加到内存
        self.memory["recent_data"].append(memory_record)
        
        # 限制内存大小
        max_memory = self.config.get("memory_length", 24)
        if len(self.memory["recent_data"]) > max_memory:
            self.memory["recent_data"] = self.memory["recent_data"][-max_memory:]
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """校准预测置信度
        
        Args:
            raw_confidence: 原始置信度
            
        Returns:
            float: 校准后的置信度
        """
        # 应用校准因子
        calibration_factor = self.state["calibration_level"]
        calibrated = raw_confidence * (0.7 + 0.3 * calibration_factor)
        
        # 确保在有效范围内
        return max(0.1, min(0.99, calibrated))
    
    def calibrate(self, calibration_level: float = None) -> bool:
        """校准模型
        
        Args:
            calibration_level: 校准级别 (None则使用默认值)
            
        Returns:
            bool: 校准是否成功
        """
        self.logger.debug(f"校准模型 {self.name}")
        
        if calibration_level is not None:
            self.state["calibration_level"] = max(0.1, min(1.0, calibration_level))
        else:
            # 使用默认校准级别
            self.state["calibration_level"] = self.config.get("calibration_factor", 0.8)
            
        # 根据模型类型执行特定校准
        if self.name == "quantum_wave":
            # 调整波参数
            self.model_params["wave_amplitude"] *= (0.9 + 0.2 * self.state["calibration_level"])
            self.model_params["phase_shift"] *= (0.9 + 0.2 * self.state["calibration_level"])
            
        elif self.name == "timeline_confluence":
            # 调整时间线参数
            self.model_params["divergence_weight"] *= (0.8 + 0.4 * self.state["calibration_level"])
            
        elif self.name == "entropic_analysis":
            # 调整熵分析参数
            self.model_params["entropy_sensitivity"] *= (0.9 + 0.2 * self.state["calibration_level"])
            
        # 更新状态
        self.state["last_update"] = datetime.now()
        
        self.logger.debug(f"模型 {self.name} 校准完成，级别: {self.state['calibration_level']:.2f}")
        return True
    
    def evaluate(self, prediction: Dict[str, Any], 
                actual_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估预测准确性
        
        Args:
            prediction: 预测结果
            actual_data: 实际市场数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.debug(f"评估模型 {self.name} 预测")
        
        # 提取预测和实际价格
        predicted_price = prediction.get("prediction", {}).get("price", 0)
        actual_price = actual_data.get("price", 0)
        
        if predicted_price <= 0 or actual_price <= 0:
            return {
                "success": False,
                "error": "invalid_price_data"
            }
        
        # 计算价格误差
        absolute_error = abs(predicted_price - actual_price)
        relative_error = absolute_error / actual_price
        
        # 计算价格准确性
        price_accuracy = max(0.0, 1.0 - relative_error)
        
        # 提取预测和实际趋势
        predicted_trend = prediction.get("prediction", {}).get("trend", 0)
        actual_trend = actual_data.get("trend", 0)
        
        # 计算趋势准确性
        if (predicted_trend > 0 and actual_trend > 0) or (predicted_trend < 0 and actual_trend < 0):
            # 趋势方向一致
            trend_accuracy = 1.0
        else:
            # 趋势方向不一致
            trend_accuracy = 0.0
            
        # 计算总体准确性
        overall_accuracy = (price_accuracy * 0.7 + trend_accuracy * 0.3)
        
        # 记录误差
        error_record = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual": actual_data,
            "price_error": relative_error,
            "trend_error": 0 if trend_accuracy == 1.0 else 1,
            "overall_accuracy": overall_accuracy
        }
        
        self.memory["error_history"].append(error_record)
        
        # 限制历史大小
        max_history = self.config.get("memory_length", 24)
        if len(self.memory["error_history"]) > max_history:
            self.memory["error_history"] = self.memory["error_history"][-max_history:]
            
        # 更新模型准确率
        self.state["accuracy"] = (
            self.state["accuracy"] * 0.9 +  # 90%的旧值
            overall_accuracy * 0.1          # 10%的新值
        )
        
        # 构建评估结果
        evaluation = {
            "success": True,
            "price_accuracy": price_accuracy,
            "trend_accuracy": trend_accuracy,
            "overall_accuracy": overall_accuracy,
            "relative_error": relative_error,
            "absolute_error": absolute_error,
            "model": self.name,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"模型 {self.name} 评估完成，准确率: {overall_accuracy:.2f}")
        return evaluation
    
    def get_state(self) -> Dict[str, Any]:
        """获取模型状态
        
        Returns:
            Dict[str, Any]: 模型状态
        """
        # 复制状态，避免修改原始数据
        state_copy = self.state.copy()
        
        # 处理日期时间
        if isinstance(state_copy["last_update"], datetime):
            state_copy["last_update"] = state_copy["last_update"].isoformat()
            
        # 添加额外信息
        state_copy["model_name"] = self.name
        state_copy["model_version"] = self.version
        state_copy["memory_size"] = {
            "recent_data": len(self.memory["recent_data"]),
            "recent_predictions": len(self.memory["recent_predictions"]),
            "error_history": len(self.memory["error_history"])
        }
        
        return state_copy 