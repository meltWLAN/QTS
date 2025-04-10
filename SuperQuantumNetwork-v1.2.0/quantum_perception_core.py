#!/usr/bin/env python3
"""
超神量子共生系统 - 量子感知核心
实现高维市场感知与预测能力的基础模块
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("QuantumPerception")

class PerceptionDimension(Enum):
    """感知维度"""
    PRICE_MOMENTUM = auto()     # 价格动量
    VOLUME_FLOW = auto()        # 成交量流动
    TIME_SERIES = auto()        # 时间序列
    MARKET_STRUCTURE = auto()   # 市场结构
    NEWS_IMPACT = auto()        # 新闻影响
    POLICY_WAVE = auto()        # 政策波动
    CAPITAL_CURRENT = auto()    # 资金流向
    EMOTIONAL_FIELD = auto()    # 情绪场
    COSMIC_RESONANCE = auto()   # 宇宙共振
    QUANTUM_ENTANGLEMENT = auto() # 量子纠缠

class MarketState(Enum):
    """市场状态"""
    ACCUMULATION = auto()       # 积累期
    MARKUP = auto()             # 上涨期
    DISTRIBUTION = auto()       # 分配期
    MARKDOWN = auto()           # 下跌期
    CHAOS = auto()              # 混沌期
    QUANTUM_SHIFT = auto()      # 量子跃迁期

class QuantumPerceptionCore:
    """量子感知核心"""
    
    def __init__(self, config=None):
        """
        初始化量子感知核心
        
        参数:
            config: 配置字典
        """
        self.logger = logging.getLogger("QuantumPerceptionCore")
        self.logger.info("初始化量子感知核心...")
        
        # 默认配置
        self.config = {
            'perception_dimensions': 11,
            'consciousness_depth': 7,
            'quantum_field_resolution': 0.01,
            'temporal_sensitivity': 0.85,
            'field_coherence_threshold': 0.7,
            'entanglement_detection': True,
            'cosmic_resonance_amplification': 2.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化感知场
        self.perception_field = self._initialize_perception_field()
        
        # 初始化状态跟踪
        self.current_state = MarketState.ACCUMULATION
        self.state_probabilities = {state: 0.0 for state in MarketState}
        self.state_probabilities[MarketState.ACCUMULATION] = 1.0
        
        # 初始化维度权重
        self.dimension_weights = {dim: 1.0 for dim in PerceptionDimension}
        
        # 初始化量子场态
        self.quantum_field = {
            'coherence': 0.8,
            'energy_level': 0.5,
            'phase': 0.0,
            'entanglement_strength': 0.3,
            'field_integrity': 0.9
        }
        
        # 初始化感知模型
        self.perception_model = self._build_perception_model()
        
        # 时间序列记忆
        self.time_memory = []
        self.max_memory_length = 100
        
        # 市场结构记忆
        self.structure_memory = {}
        
        # 初始化成功
        self.logger.info(f"量子感知核心初始化成功，运行于 {self.config['device']} 设备")
    
    def _initialize_perception_field(self):
        """初始化感知场"""
        field = {
            'dimensions': {},
            'harmonics': np.zeros(5),
            'resonance_points': [],
            'field_strength': 0.5,
            'coherence_matrix': np.eye(len(PerceptionDimension)),
            'phase_alignment': 0.0
        }
        
        # 初始化各维度
        for dim in PerceptionDimension:
            field['dimensions'][dim.name] = {
                'sensitivity': np.random.uniform(0.7, 0.9),
                'current_value': 0.0,
                'historical_values': [],
                'resonance_frequency': np.random.uniform(5.0, 12.0),
                'phase': np.random.uniform(0, 2*np.pi)
            }
        
        return field
    
    def _build_perception_model(self):
        """构建感知模型"""
        
        class QuantumPerceptionNetwork(nn.Module):
            """量子感知神经网络"""
            def __init__(self, input_dim, hidden_dims, output_dim):
                super(QuantumPerceptionNetwork, self).__init__()
                
                self.input_layer = nn.Linear(input_dim, hidden_dims[0])
                
                self.hidden_layers = nn.ModuleList()
                for i in range(len(hidden_dims) - 1):
                    self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                
                self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
                
                # 量子注意力机制
                self.attention = nn.MultiheadAttention(hidden_dims[-1], num_heads=4)
                
                # 场相干性层
                self.coherence_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1])
                
                # 量子纠缠模拟层
                self.entanglement_layer = nn.Linear(hidden_dims[-1], hidden_dims[-1])
            
            def forward(self, x):
                x = F.relu(self.input_layer(x))
                
                for layer in self.hidden_layers:
                    x = F.relu(layer(x))
                
                # 应用注意力机制
                x_reshaped = x.unsqueeze(0)
                attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
                x = x + attn_output.squeeze(0)
                
                # 应用场相干性
                coherence = torch.sigmoid(self.coherence_layer(x))
                x = x * coherence
                
                # 应用量子纠缠模拟
                entanglement = self.entanglement_layer(x)
                x = x + torch.tanh(entanglement) * 0.1
                
                return self.output_layer(x)
        
        # 构建模型
        in_dim = len(PerceptionDimension)
        hidden_dims = [64, 128, 256, 128, 64]
        out_dim = len(MarketState)
        
        model = QuantumPerceptionNetwork(in_dim, hidden_dims, out_dim)
        model.to(self.config['device'])
        
        return model
    
    def perceive(self, market_data, additional_data=None):
        """
        感知市场状态
        
        参数:
            market_data: DataFrame, 市场数据
            additional_data: dict, 可选，附加数据
            
        返回:
            dict: 感知结果
        """
        self.logger.info("开始量子感知过程...")
        
        try:
            # 确保市场数据有足够的历史
            if len(market_data) < 5:
                self.logger.warning("市场数据不足，无法可靠进行量子感知")
                return {'error': '数据不足'}
            
            # 计算基本特征
            features = self._extract_features(market_data)
            
            # 更新感知场
            self._update_perception_field(features, additional_data)
            
            # 更新量子场态
            self._update_quantum_field(features)
            
            # 计算市场状态概率
            state_probs = self._calculate_state_probabilities()
            
            # 找出最可能的状态
            current_state = max(state_probs.items(), key=lambda x: x[1])[0]
            
            # 检测是否为量子跃迁期
            if self._detect_quantum_shift(features):
                current_state = MarketState.QUANTUM_SHIFT
                state_probs[MarketState.QUANTUM_SHIFT] = max(state_probs.values()) * 1.2
            
            # 更新当前状态
            self.current_state = current_state
            self.state_probabilities = state_probs
            
            # 检测共振点
            resonance_points = self._detect_resonance_points(features)
            
            # 整合感知结果
            perception_result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_state': self.current_state.name,
                'state_probabilities': {k.name: v for k, v in state_probs.items()},
                'quantum_field': self.quantum_field,
                'resonance_points': resonance_points,
                'field_coherence': self.perception_field['coherence_matrix'].diagonal().mean(),
                'most_sensitive_dimension': max(
                    self.perception_field['dimensions'].items(), 
                    key=lambda x: x[1]['sensitivity']
                )[0]
            }
            
            # 更新时间序列记忆
            self._update_memory(perception_result)
            
            self.logger.info(f"量子感知完成，当前市场状态: {self.current_state.name}")
            return perception_result
            
        except Exception as e:
            self.logger.error(f"量子感知过程失败: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}
    
    def _extract_features(self, market_data):
        """提取市场特征"""
        # 提取价格动量特征
        close_prices = market_data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        
        # 提取成交量特征
        if 'volume' in market_data.columns:
            volumes = market_data['volume'].values
            volume_change = volumes[-1] / np.mean(volumes[-6:-1]) if len(volumes) >= 6 else 1.0
        else:
            volume_change = 1.0
        
        # 提取市场结构特征
        if len(close_prices) >= 20:
            ma5 = np.mean(close_prices[-5:])
            ma10 = np.mean(close_prices[-10:])
            ma20 = np.mean(close_prices[-20:])
            structure_feature = (ma5 / ma10 - 1) + (ma5 / ma20 - 1)
        else:
            structure_feature = 0
        
        # 计算波动率
        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
        
        # 整合特征
        features = {
            PerceptionDimension.PRICE_MOMENTUM.name: momentum,
            PerceptionDimension.VOLUME_FLOW.name: volume_change - 1,
            PerceptionDimension.TIME_SERIES.name: np.mean(returns[-3:]) if len(returns) >= 3 else 0,
            PerceptionDimension.MARKET_STRUCTURE.name: structure_feature,
            # 其他特征初始值为0，等待附加数据更新
            PerceptionDimension.NEWS_IMPACT.name: 0,
            PerceptionDimension.POLICY_WAVE.name: 0,
            PerceptionDimension.CAPITAL_CURRENT.name: 0,
            PerceptionDimension.EMOTIONAL_FIELD.name: 0,
            PerceptionDimension.COSMIC_RESONANCE.name: 0,
            PerceptionDimension.QUANTUM_ENTANGLEMENT.name: 0
        }
        
        return features
    
    def _update_perception_field(self, features, additional_data=None):
        """更新感知场"""
        # 更新各维度当前值
        for dim_name, value in features.items():
            if dim_name in self.perception_field['dimensions']:
                dim = self.perception_field['dimensions'][dim_name]
                dim['current_value'] = value
                dim['historical_values'].append(value)
                
                # 保持历史记录在合理范围内
                if len(dim['historical_values']) > self.max_memory_length:
                    dim['historical_values'] = dim['historical_values'][-self.max_memory_length:]
        
        # 如果有附加数据，更新相应维度
        if additional_data:
            for dim_name, value in additional_data.items():
                if dim_name in self.perception_field['dimensions']:
                    self.perception_field['dimensions'][dim_name]['current_value'] = value
                    self.perception_field['dimensions'][dim_name]['historical_values'].append(value)
                    
                    # 保持历史记录在合理范围内
                    hist_values = self.perception_field['dimensions'][dim_name]['historical_values']
                    if len(hist_values) > self.max_memory_length:
                        self.perception_field['dimensions'][dim_name]['historical_values'] = hist_values[-self.max_memory_length:]
        
        # 更新维度之间的相干性矩阵
        dim_values = np.array([
            dim['current_value'] 
            for dim in self.perception_field['dimensions'].values()
        ])
        
        # 使用外积更新相干性矩阵
        coherence_update = np.outer(dim_values, dim_values)
        coherence_update = coherence_update / (np.linalg.norm(coherence_update) + 1e-8)
        
        # 平滑更新
        alpha = 0.1  # 更新速率
        self.perception_field['coherence_matrix'] = (1 - alpha) * self.perception_field['coherence_matrix'] + alpha * coherence_update
        
        # 更新场强度
        self.perception_field['field_strength'] = np.mean(np.abs(dim_values))
        
        # 更新相位对齐
        phases = np.array([dim['phase'] for dim in self.perception_field['dimensions'].values()])
        phase_variance = np.var(phases)
        self.perception_field['phase_alignment'] = 1.0 / (1.0 + phase_variance)
        
        # 更新谐波
        for i in range(len(self.perception_field['harmonics'])):
            freq = i + 1
            harmonic_value = np.sum([
                np.sin(freq * dim['phase']) * dim['current_value']
                for dim in self.perception_field['dimensions'].values()
            ])
            self.perception_field['harmonics'][i] = harmonic_value
    
    def _update_quantum_field(self, features):
        """更新量子场态"""
        # 计算场相干性
        coherence_diagonal = np.diagonal(self.perception_field['coherence_matrix'])
        self.quantum_field['coherence'] = np.mean(coherence_diagonal)
        
        # 计算能量水平
        energy_contribution = np.mean([
            abs(features[dim.name]) * self.dimension_weights[dim] 
            for dim in PerceptionDimension 
            if dim.name in features
        ])
        
        # 平滑更新能量水平
        alpha = 0.2  # 更新速率
        self.quantum_field['energy_level'] = (1 - alpha) * self.quantum_field['energy_level'] + alpha * energy_contribution
        
        # 更新相位
        phase_shift = np.mean([
            np.sin(self.perception_field['dimensions'][dim.name]['phase']) 
            for dim in PerceptionDimension 
            if dim.name in self.perception_field['dimensions']
        ])
        self.quantum_field['phase'] = (self.quantum_field['phase'] + phase_shift * 0.1) % (2 * np.pi)
        
        # 更新纠缠强度
        off_diagonal_mean = np.mean(self.perception_field['coherence_matrix'] - np.diag(coherence_diagonal))
        self.quantum_field['entanglement_strength'] = abs(off_diagonal_mean) * 2
        
        # 更新场完整性
        integrity_factors = [
            self.quantum_field['coherence'],
            1 - abs(np.mean(list(features.values()))),
            self.perception_field['phase_alignment']
        ]
        self.quantum_field['field_integrity'] = np.mean(integrity_factors)
    
    def _calculate_state_probabilities(self):
        """计算市场状态概率"""
        # 从感知场中提取特征向量
        feature_vector = np.array([
            dim['current_value'] for dim in self.perception_field['dimensions'].values()
        ])
        
        # 转换为PyTorch张量
        tensor_input = torch.FloatTensor(feature_vector).to(self.config['device'])
        
        # 使用感知模型预测状态概率
        with torch.no_grad():
            model_output = self.perception_model(tensor_input)
            probabilities = F.softmax(model_output, dim=0).cpu().numpy()
        
        # 创建状态概率字典
        state_probs = {state: probabilities[i] for i, state in enumerate(MarketState) if i < len(probabilities)}
        
        # 应用量子场修正
        for state in state_probs:
            modifier = 1.0
            
            if state == MarketState.MARKUP:
                # 上涨期受能量水平正向影响
                modifier += self.quantum_field['energy_level'] * 0.5
                
            elif state == MarketState.MARKDOWN:
                # 下跌期受能量水平负向影响
                modifier += (1 - self.quantum_field['energy_level']) * 0.5
                
            elif state == MarketState.ACCUMULATION or state == MarketState.DISTRIBUTION:
                # 积累期和分配期受相干性影响
                modifier += self.quantum_field['coherence'] * 0.3
                
            elif state == MarketState.CHAOS:
                # 混沌期受纠缠强度影响
                modifier += self.quantum_field['entanglement_strength'] * 0.7
            
            state_probs[state] *= modifier
        
        # 重新归一化
        total_prob = sum(state_probs.values())
        if total_prob > 0:
            state_probs = {state: prob / total_prob for state, prob in state_probs.items()}
        
        return state_probs
    
    def _detect_quantum_shift(self, features):
        """检测量子跃迁"""
        # 量子跃迁的条件
        conditions = [
            self.quantum_field['entanglement_strength'] > 0.7,
            self.quantum_field['energy_level'] > 0.8 or self.quantum_field['energy_level'] < 0.2,
            abs(features.get(PerceptionDimension.PRICE_MOMENTUM.name, 0)) > 0.03,
            self.quantum_field['coherence'] < 0.4
        ]
        
        # 如果满足至少3个条件，判断为量子跃迁期
        return sum(conditions) >= 3
    
    def _detect_resonance_points(self, features):
        """检测共振点"""
        resonance_points = []
        
        # 检查各维度是否存在共振
        for dim_name, dim_info in self.perception_field['dimensions'].items():
            # 至少需要有一定的历史数据
            if len(dim_info['historical_values']) < 5:
                continue
            
            # 取最近的值
            recent_values = dim_info['historical_values'][-5:]
            
            # 计算相对变化率
            changes = np.diff(recent_values) / (np.abs(recent_values[:-1]) + 1e-8)
            
            # 检查是否有明显波动后的稳定
            if np.std(changes) > 0.2 and np.std(recent_values[-2:]) < np.std(recent_values) * 0.5:
                # 检查是否与其他维度有共振
                for other_dim, other_info in self.perception_field['dimensions'].items():
                    if other_dim == dim_name or len(other_info['historical_values']) < 5:
                        continue
                    
                    other_recent = other_info['historical_values'][-5:]
                    correlation = np.corrcoef(recent_values, other_recent)[0, 1]
                    
                    if abs(correlation) > 0.7:
                        resonance_points.append({
                            'primary_dimension': dim_name,
                            'resonating_dimension': other_dim,
                            'correlation': correlation,
                            'strength': abs(correlation) * dim_info['sensitivity'],
                            'phase_alignment': 1 - abs(dim_info['phase'] - other_info['phase']) / np.pi
                        })
        
        return resonance_points
    
    def _update_memory(self, perception_result):
        """更新系统记忆"""
        # 更新时间序列记忆
        self.time_memory.append({
            'timestamp': perception_result['timestamp'],
            'state': perception_result['current_state'],
            'field_coherence': perception_result['field_coherence'],
            'energy_level': self.quantum_field['energy_level']
        })
        
        # 保持记忆在合理大小范围内
        if len(self.time_memory) > self.max_memory_length:
            self.time_memory = self.time_memory[-self.max_memory_length:]
        
        # 更新市场结构记忆
        self.structure_memory[perception_result['timestamp']] = {
            'state': perception_result['current_state'],
            'resonance_points': len(perception_result['resonance_points']),
            'field_state': {
                'coherence': self.quantum_field['coherence'],
                'energy': self.quantum_field['energy_level'],
                'entanglement': self.quantum_field['entanglement_strength']
            }
        }
    
    def adjust_sensitivities(self, performance_feedback=None):
        """
        根据反馈调整维度敏感度
        
        参数:
            performance_feedback: dict, 性能反馈
        """
        try:
            # 基于时间自然衰减
            for dim_name, dim_info in self.perception_field['dimensions'].items():
                # 略微降低敏感度
                dim_info['sensitivity'] *= 0.99
                # 确保敏感度在合理范围内
                dim_info['sensitivity'] = max(0.4, min(0.95, dim_info['sensitivity']))
            
            # 如果有性能反馈，基于反馈调整
            if performance_feedback and 'dimension_performance' in performance_feedback:
                for dim_name, performance in performance_feedback['dimension_performance'].items():
                    if dim_name not in self.perception_field['dimensions']:
                        continue
                    
                    # 性能好的维度增加敏感度
                    if performance > 0:
                        self.perception_field['dimensions'][dim_name]['sensitivity'] += performance * 0.05
                    else:
                        self.perception_field['dimensions'][dim_name]['sensitivity'] += performance * 0.03
                    
                    # 确保敏感度在合理范围内
                    self.perception_field['dimensions'][dim_name]['sensitivity'] = max(0.4, min(0.95, self.perception_field['dimensions'][dim_name]['sensitivity']))
            
            # 随机探索 - 偶尔随机调整一个维度的敏感度
            if np.random.random() < 0.1:
                random_dim = np.random.choice(list(self.perception_field['dimensions'].keys()))
                random_adjustment = np.random.uniform(-0.05, 0.05)
                self.perception_field['dimensions'][random_dim]['sensitivity'] += random_adjustment
                self.perception_field['dimensions'][random_dim]['sensitivity'] = max(0.4, min(0.95, self.perception_field['dimensions'][random_dim]['sensitivity']))
                
                self.logger.info(f"随机调整维度 {random_dim} 敏感度，调整量: {random_adjustment:.3f}")
            
            # 记录当前敏感度
            self.logger.info("当前维度敏感度:")
            for dim_name, dim_info in self.perception_field['dimensions'].items():
                self.logger.info(f"  {dim_name}: {dim_info['sensitivity']:.3f}")
                
        except Exception as e:
            self.logger.error(f"调整敏感度失败: {str(e)}")
            traceback.print_exc()
    
    def get_field_state(self):
        """获取当前场态"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'quantum_field': self.quantum_field,
            'perception_field': {
                'field_strength': self.perception_field['field_strength'],
                'phase_alignment': self.perception_field['phase_alignment'],
                'dimensions': {
                    name: {
                        'sensitivity': info['sensitivity'],
                        'current_value': info['current_value'],
                        'phase': info['phase']
                    }
                    for name, info in self.perception_field['dimensions'].items()
                }
            },
            'current_state': self.current_state.name,
            'state_probabilities': {k.name: v for k, v in self.state_probabilities.items()},
            'memory_length': len(self.time_memory)
        }
    
    def save_state(self, file_path):
        """
        保存当前状态
        
        参数:
            file_path: 保存路径
        """
        try:
            state = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config,
                'quantum_field': self.quantum_field,
                'current_state': self.current_state.name,
                'state_probabilities': {k.name: v for k, v in self.state_probabilities.items()},
                'perception_field': {
                    'field_strength': self.perception_field['field_strength'],
                    'phase_alignment': self.perception_field['phase_alignment'],
                    'harmonics': self.perception_field['harmonics'].tolist(),
                    'dimensions': {
                        name: {
                            'sensitivity': info['sensitivity'],
                            'current_value': info['current_value'],
                            'phase': info['phase'],
                            'resonance_frequency': info['resonance_frequency']
                        }
                        for name, info in self.perception_field['dimensions'].items()
                    }
                },
                'dimension_weights': {k.name: v for k, v in self.dimension_weights.items()},
                'model_state': self.perception_model.state_dict()
            }
            
            # 保存到文件
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(state, file_path)
            
            self.logger.info(f"量子感知核心状态已保存到: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_state(self, file_path):
        """
        加载状态
        
        参数:
            file_path: 加载路径
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"状态文件不存在: {file_path}")
                return False
            
            # 加载状态
            state = torch.load(file_path, map_location=self.config['device'])
            
            # 恢复配置
            self.config.update(state['config'])
            
            # 恢复量子场
            self.quantum_field = state['quantum_field']
            
            # 恢复当前状态
            self.current_state = MarketState[state['current_state']]
            self.state_probabilities = {MarketState[k]: v for k, v in state['state_probabilities'].items()}
            
            # 恢复感知场
            self.perception_field['field_strength'] = state['perception_field']['field_strength']
            self.perception_field['phase_alignment'] = state['perception_field']['phase_alignment']
            self.perception_field['harmonics'] = np.array(state['perception_field']['harmonics'])
            
            for name, info in state['perception_field']['dimensions'].items():
                if name in self.perception_field['dimensions']:
                    self.perception_field['dimensions'][name]['sensitivity'] = info['sensitivity']
                    self.perception_field['dimensions'][name]['current_value'] = info['current_value']
                    self.perception_field['dimensions'][name]['phase'] = info['phase']
                    self.perception_field['dimensions'][name]['resonance_frequency'] = info['resonance_frequency']
            
            # 恢复维度权重
            for k, v in state['dimension_weights'].items():
                if hasattr(PerceptionDimension, k):
                    self.dimension_weights[getattr(PerceptionDimension, k)] = v
            
            # 恢复模型状态
            self.perception_model.load_state_dict(state['model_state'])
            
            self.logger.info(f"量子感知核心状态已从 {file_path} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载状态失败: {str(e)}")
            traceback.print_exc()
            return False

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建量子感知核心
    perception_core = QuantumPerceptionCore()
    
    # 输出诊断信息
    print("\n" + "="*60)
    print("超神量子共生系统 - 量子感知核心")
    print("="*60 + "\n")
    
    print("感知维度:")
    for dim in PerceptionDimension:
        sensitivity = perception_core.perception_field['dimensions'][dim.name]['sensitivity']
        print(f"- {dim.name}: 敏感度 {sensitivity:.3f}")
    
    print("\n量子场状态:")
    for key, value in perception_core.quantum_field.items():
        print(f"- {key}: {value:.3f}")
    
    print("\n市场状态概率:")
    for state, prob in perception_core.state_probabilities.items():
        print(f"- {state.name}: {prob:.3f}")
    
    print("\n系统准备就绪，可以开始感知市场量子场态。")
    print("="*60 + "\n") 