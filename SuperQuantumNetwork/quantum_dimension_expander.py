#!/usr/bin/env python3
"""
超神量子共生系统 - 量子维度扩展模块
用于将市场分析从11维扩展到21维，增强分析深度和预测能力
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import math

# 设置日志
logger = logging.getLogger("QuantumDimensionExpander")

class QuantumDimensionExpander:
    """量子维度扩展器，将市场数据从基础维度扩展到更高维度空间"""
    
    def __init__(self, config=None):
        """初始化量子维度扩展器
        
        参数:
            config: 配置参数
        """
        self.logger = logger
        self.config = config or {}
        self.base_dimensions = 11
        self.target_dimensions = 21
        self.initialized = False
        self.dimension_weights = None
        self.expansion_matrix = None
        self.entanglement_factor = self.config.get('entanglement_factor', 0.42)
        self.precision_factor = self.config.get('precision_factor', 1.0)  # 精度系数
        self.adaptive_mode = self.config.get('adaptive_mode', True)  # 自适应模式
        self.market_condition = 'normal'  # 市场状态：normal, volatile, trending
        self.expansion_quality = 0.0  # 扩展质量评分
        
    def initialize(self):
        """初始化扩展器，使用更精确的维度建模"""
        self.logger.info(f"初始化量子维度扩展器: {self.base_dimensions} -> {self.target_dimensions}")
        
        # 创建维度权重 - 使用非线性递减
        self.dimension_weights = np.exp(-np.linspace(0, 1, self.target_dimensions)) * 0.7 + 0.3
        
        # 创建扩展矩阵（高维映射）
        np.random.seed(42)  # 确保可重复性
        self.expansion_matrix = np.random.normal(0, 0.1, (self.target_dimensions, self.base_dimensions))
        
        # 修正扩展矩阵，确保基本维度直接映射
        for i in range(min(self.base_dimensions, self.target_dimensions)):
            self.expansion_matrix[i] = np.zeros(self.base_dimensions)
            self.expansion_matrix[i, i] = 1.0
        
        # 扩展维度的正交化处理，提高独立性
        for i in range(self.base_dimensions, self.target_dimensions):
            # 对扩展维度进行正交化
            for j in range(self.base_dimensions):
                # 减少与基础维度的重合
                projection = np.dot(self.expansion_matrix[i], self.expansion_matrix[j])
                self.expansion_matrix[i] = self.expansion_matrix[i] - projection * self.expansion_matrix[j]
            
            # 归一化
            norm = np.linalg.norm(self.expansion_matrix[i])
            if norm > 1e-10:  # 避免除以零
                self.expansion_matrix[i] = self.expansion_matrix[i] / norm
            
        self.initialized = True
        self.logger.info("量子维度扩展器初始化完成")
        return True
        
    def expand_dimensions(self, market_state):
        """将市场状态从11维扩展到21维，采用增强的量子算法
        
        参数:
            market_state: 市场状态向量或字典
            
        返回:
            dict: 扩展后的市场状态
        """
        if not self.initialized:
            self.initialize()
            
        self.logger.info("开始精确量子维度扩展")
        
        # 转换输入为向量形式
        if isinstance(market_state, dict):
            # 从字典中提取向量
            base_vector = self._dict_to_vector(market_state)
            # 自动检测市场状态
            self._detect_market_condition(market_state)
        else:
            base_vector = np.array(market_state)
            
        # 保存原始向量用于质量评估
        original_vector = base_vector.copy()
            
        # 确保向量维度正确
        if len(base_vector) != self.base_dimensions:
            self.logger.warning(f"输入向量维度不匹配: 预期{self.base_dimensions}, 实际{len(base_vector)}")
            # 对齐向量维度
            if len(base_vector) < self.base_dimensions:
                # 扩展向量
                padding = np.zeros(self.base_dimensions - len(base_vector))
                base_vector = np.concatenate([base_vector, padding])
            else:
                # 截断向量
                base_vector = base_vector[:self.base_dimensions]
                
        # 执行量子维度扩展
        expanded_vector = self._quantum_expansion(base_vector)
        
        # 转换回字典形式
        expanded_state = self._vector_to_expanded_dict(expanded_vector, market_state)
        
        self.logger.info(f"精确维度扩展完成: {self.base_dimensions}D -> {self.target_dimensions}D")
        return expanded_state
        
    def _quantum_expansion(self, base_vector):
        """量子扩展算法，将基础向量扩展到更高维度
        
        参数:
            base_vector: 基础维度向量
            
        返回:
            np.ndarray: 扩展维度向量
        """
        # 保存原始向量副本
        orig_base_vector = base_vector.copy()
        
        # 量子扩展前的归一化 - 避免量值过大或过小
        normalized_base = np.tanh(base_vector * 0.1) * 2
        
        # 基本线性变换
        expanded = np.dot(self.expansion_matrix, normalized_base)
        
        # 市场状态感知的量子纠缠系数
        entanglement_factor = self.entanglement_factor
        if self.adaptive_mode:
            if self.market_condition == 'volatile':
                entanglement_factor *= 1.2  # 波动市场增强纠缠
            elif self.market_condition == 'trending':
                entanglement_factor *= 0.8  # 趋势市场减弱纠缠
        
        # 1. 量子纠缠：某些维度之间相互关联，更精确的控制
        for i in range(self.base_dimensions, self.target_dimensions):
            for j in range(i+1, self.target_dimensions):
                # 使用相位差异产生更复杂的纠缠效应
                phase_diff = np.sin(np.pi * (i - j) / self.target_dimensions)
                entanglement = entanglement_factor * normalized_base[min(i % self.base_dimensions, self.base_dimensions-1)] * expanded[j] * phase_diff
                expanded[i] += entanglement
                expanded[j] += entanglement * 0.7  # 非对称纠缠
                
        # 2. 量子叠加：新维度是多个基础维度的叠加态
        for i in range(self.base_dimensions, self.target_dimensions):
            # 复杂波函数，不仅使用sin还使用cos增加复杂性
            wave_function = np.sin(normalized_base * np.pi * (i % 5 + 1) / 10)
            phase_shift = np.cos(normalized_base * np.pi * ((i + 2) % 7 + 1) / 14)
            
            # 叠加波函数
            superposition = (wave_function * self.dimension_weights[i] + 
                           phase_shift * self.dimension_weights[i] * 0.5)
            
            expanded[i] += np.sum(superposition) / len(superposition) * self.precision_factor
            
        # 3. 非线性变换：增强的量子隧穿效应 - 只应用于扩展维度
        for i in range(self.base_dimensions, self.target_dimensions):
            # 非线性S型曲线变换，保持在合理范围内
            expanded[i] = 2.0 * (1.0 / (1.0 + np.exp(-expanded[i] * 0.5)) - 0.5)
            
        # 确保基础维度保持不变
        expanded[:self.base_dimensions] = orig_base_vector
            
        return expanded
        
    def _dict_to_vector(self, market_state):
        """将市场状态字典转换为向量
        
        参数:
            market_state: 市场状态字典
            
        返回:
            np.ndarray: 市场状态向量
        """
        # 定义基础维度的标准字段（顺序很重要）
        base_fields = [
            "price", "volume", "momentum", "volatility", "trend", 
            "oscillator", "sentiment", "liquidity", "correlation", 
            "divergence", "cycle_phase"
        ]
        
        # 创建向量
        vector = np.zeros(self.base_dimensions)
        
        # 填充向量
        for i, field in enumerate(base_fields):
            if i < self.base_dimensions:
                if field in market_state:
                    vector[i] = float(market_state[field])
                    
        return vector
        
    def _vector_to_expanded_dict(self, expanded_vector, original_state):
        """将扩展向量转换回字典
        
        参数:
            expanded_vector: 扩展后的向量
            original_state: 原始市场状态字典
            
        返回:
            dict: 扩展后的市场状态字典
        """
        # 定义扩展维度的字段名
        expanded_fields = [
            "price", "volume", "momentum", "volatility", "trend", 
            "oscillator", "sentiment", "liquidity", "correlation", 
            "divergence", "cycle_phase",
            # 扩展维度 (11-20)
            "quantum_momentum", "phase_coherence", "entropy", 
            "fractal_dimension", "resonance", "quantum_sentiment",
            "harmonic_pattern", "dimensional_flow", "attractor_strength",
            "quantum_potential"
        ]
        
        # 创建扩展状态字典
        if isinstance(original_state, dict):
            expanded_state = original_state.copy()
        else:
            expanded_state = {}
            
        # 填充扩展维度
        for i, field in enumerate(expanded_fields):
            if i < len(expanded_vector):
                expanded_state[field] = float(expanded_vector[i])
                
        # 添加元数据
        expanded_state["dimensions"] = self.target_dimensions
        expanded_state["expansion_time"] = datetime.now().isoformat()
        
        return expanded_state
    
    def collapse_dimensions(self, expanded_state):
        """精确的维度折叠算法，将扩展的市场状态从21维折叠回11维
        
        参数:
            expanded_state: 扩展后的市场状态
            
        返回:
            dict: 折叠后的市场状态
        """
        self.logger.info("开始精确量子维度折叠")
        
        # 如果是字典，转换为向量
        if isinstance(expanded_state, dict):
            expanded_vector = np.zeros(self.target_dimensions)
            # 定义扩展维度的字段名
            expanded_fields = [
                "price", "volume", "momentum", "volatility", "trend", 
                "oscillator", "sentiment", "liquidity", "correlation", 
                "divergence", "cycle_phase",
                # 扩展维度 (11-20)
                "quantum_momentum", "phase_coherence", "entropy", 
                "fractal_dimension", "resonance", "quantum_sentiment",
                "harmonic_pattern", "dimensional_flow", "attractor_strength",
                "quantum_potential"
            ]
            
            for i, field in enumerate(expanded_fields):
                if i < self.target_dimensions and field in expanded_state:
                    expanded_vector[i] = expanded_state[field]
        else:
            expanded_vector = expanded_state
            
        # 确保维度正确
        if len(expanded_vector) != self.target_dimensions:
            self.logger.warning(f"扩展向量维度不匹配: 预期{self.target_dimensions}, 实际{len(expanded_vector)}")
            if len(expanded_vector) < self.target_dimensions:
                # 扩展向量
                padding = np.zeros(self.target_dimensions - len(expanded_vector))
                expanded_vector = np.concatenate([expanded_vector, padding])
            else:
                # 截断向量
                expanded_vector = expanded_vector[:self.target_dimensions]
        
        # 精确的维度折叠
        collapsed_vector = np.zeros(self.base_dimensions)
        
        # 1. 直接保留基础维度
        collapsed_vector = expanded_vector[:self.base_dimensions].copy()
        
        # 2. 计算扩展维度对基础维度的影响
        influence_factor = 0.05 * self.precision_factor  # 影响因子，精度越高影响越大
        
        if self.adaptive_mode:
            # 在不同市场状态下调整影响因子
            if self.market_condition == 'volatile':
                influence_factor *= 0.8  # 波动市场减少影响以增强稳定性
            elif self.market_condition == 'trending':
                influence_factor *= 1.2  # 趋势市场增强信号
        
        # 高维空间对基础空间的影响
        for i in range(self.base_dimensions):
            quantum_influence = 0
            
            # 基于扩展矩阵计算影响权重
            for j in range(self.base_dimensions, self.target_dimensions):
                # 权重基于扩展矩阵中的对应关系
                weight = self.expansion_matrix[j, i]
                contribution = expanded_vector[j] * weight
                quantum_influence += contribution
            
            # 使用tanh函数限制量子影响的范围
            max_influence = abs(collapsed_vector[i]) * 0.1
            scaled_influence = np.tanh(quantum_influence) * max_influence * influence_factor
            
            # 应用量子影响，但保持值的合理性
            collapsed_vector[i] += scaled_influence
                
        # 转换回字典
        collapsed_state = {}
        base_fields = [
            "price", "volume", "momentum", "volatility", "trend", 
            "oscillator", "sentiment", "liquidity", "correlation", 
            "divergence", "cycle_phase"
        ]
        
        for i, field in enumerate(base_fields):
            if i < self.base_dimensions:
                collapsed_state[field] = float(collapsed_vector[i])
                
        # 保留原始状态的非维度字段
        if isinstance(expanded_state, dict):
            for key, value in expanded_state.items():
                if key not in expanded_fields and key not in ["dimensions", "expansion_time"]:
                    collapsed_state[key] = value
                    
        # 添加元数据
        collapsed_state["dimensions"] = self.base_dimensions
        collapsed_state["collapse_time"] = datetime.now().isoformat()
        collapsed_state["market_condition"] = self.market_condition
        
        self.logger.info(f"精确维度折叠完成: {self.target_dimensions}D -> {self.base_dimensions}D")
        return collapsed_state
    
    def _detect_market_condition(self, market_state):
        """根据市场状态自动检测市场条件
        
        参数:
            market_state: 市场状态字典
        """
        # 默认为normal
        condition = 'normal'
        
        # 检查波动性和趋势
        volatility = market_state.get('volatility', 0)
        trend = market_state.get('trend', 0)
        momentum = market_state.get('momentum', 0)
        
        # 高波动性市场
        if abs(volatility) > 0.3 or abs(momentum) > 0.4:
            condition = 'volatile'
            
        # 强趋势市场
        if abs(trend) > 0.3 and abs(momentum) > 0.2:
            condition = 'trending'
            
        # 设置检测到的市场状态
        if condition != self.market_condition:
            self.market_condition = condition
            self.logger.info(f"自动检测到市场状态: {condition}")
        
        return condition

# 创建扩展器工厂函数
def get_dimension_expander(config=None):
    """获取量子维度扩展器实例
    
    参数:
        config: 配置参数
        
    返回:
        QuantumDimensionExpander: 扩展器实例
    """
    return QuantumDimensionExpander(config)

def test_dimension_expander():
    """测试量子维度扩展器"""
    # 初始化扩展器
    expander = get_dimension_expander()
    
    # 创建测试市场状态
    market_state = {
        "symbol": "000001.SH",
        "price": 3100.0,
        "volume": 0.75,
        "momentum": 0.2,
        "volatility": 0.15,
        "trend": 0.5,
        "oscillator": 0.3,
        "sentiment": -0.1,
        "liquidity": 0.8,
        "correlation": 0.4,
        "divergence": 0.1,
        "cycle_phase": 0.25,
        "timestamp": "2025-04-08T08:00:00"
    }
    
    # 执行维度扩展
    expanded_state = expander.expand_dimensions(market_state)
    
    # 显示结果
    print("\n原始市场状态 (11维):")
    base_dims = ["price", "volume", "momentum", "volatility", "trend", 
                "oscillator", "sentiment", "liquidity", "correlation", 
                "divergence", "cycle_phase"]
    for dim in base_dims:
        if dim in market_state:
            print(f"  {dim}: {market_state[dim]}")
    
    print("\n扩展后市场状态 (21维):")
    expanded_dims = base_dims + [
        "quantum_momentum", "phase_coherence", "entropy", 
        "fractal_dimension", "resonance", "quantum_sentiment",
        "harmonic_pattern", "dimensional_flow", "attractor_strength",
        "quantum_potential"
    ]
    
    for dim in expanded_dims:
        if dim in expanded_state:
            print(f"  {dim}: {expanded_state[dim]:.4f}")
    
    # 测试维度折叠
    collapsed_state = expander.collapse_dimensions(expanded_state)
    
    print("\n折叠后市场状态 (11维):")
    for dim in base_dims:
        if dim in collapsed_state:
            print(f"  {dim}: {collapsed_state[dim]:.4f}")
    
    return expanded_state, collapsed_state

if __name__ == "__main__":
    # 设置日志输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("测试量子维度扩展器...")
    expanded, collapsed = test_dimension_expander()
    print("\n测试完成!") 