"""
market_to_quantum - 量子核心组件
市场数据到量子态转换器 - 将市场数据转换为量子计算表示
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class MarketToQuantumConverter:
    """市场到量子转换器 - 将市场数据转换为量子计算表示"""
    
    def __init__(self):
        self.is_running = False
        self.encoding_methods = {}
        self._register_default_encoders()
        logger.info("市场到量子转换器初始化完成")
        
    def start(self):
        """启动转换器"""
        if self.is_running:
            logger.warning("转换器已在运行")
            return
            
        logger.info("启动市场到量子转换器...")
        self.is_running = True
        logger.info("市场到量子转换器启动完成")
        
    def stop(self):
        """停止转换器"""
        if not self.is_running:
            logger.warning("转换器已停止")
            return
            
        logger.info("停止市场到量子转换器...")
        self.is_running = False
        logger.info("市场到量子转换器已停止")
        
    def _register_default_encoders(self):
        """注册默认编码方法"""
        self.register_encoding_method('amplitude', self._amplitude_encoding, 
                                 '振幅编码 - 将数据归一化后编码到量子态的振幅')
        self.register_encoding_method('angle', self._angle_encoding,
                                '角度编码 - 将数据映射到旋转角度')
        self.register_encoding_method('binary', self._binary_encoding,
                                '二进制编码 - 将数据转换为二进制表示')
        
        logger.info(f"注册了 {len(self.encoding_methods)} 个默认编码方法")
        
    def register_encoding_method(self, name: str, encoder_func, description: str = ""):
        """注册编码方法"""
        if name in self.encoding_methods:
            logger.warning(f"编码方法 '{name}' 已存在，将被替换")
            
        self.encoding_methods[name] = {
            'function': encoder_func,
            'description': description
        }
        
        logger.info(f"注册编码方法: {name}")
        return True
        
    def unregister_encoding_method(self, name: str):
        """注销编码方法"""
        if name not in self.encoding_methods:
            logger.warning(f"编码方法 '{name}' 不存在")
            return False
            
        del self.encoding_methods[name]
        logger.info(f"注销编码方法: {name}")
        return True
        
    def convert(self, market_data: Dict[str, pd.DataFrame], method: str = 'amplitude', 
              **kwargs) -> Dict[str, Any]:
        """将市场数据转换为量子表示"""
        if not self.is_running:
            logger.warning("转换器未运行，无法执行转换")
            return {'status': 'error', 'message': '转换器未运行'}
            
        if method not in self.encoding_methods:
            logger.warning(f"编码方法 '{method}' 不存在")
            return {'status': 'error', 'message': f"编码方法 '{method}' 不存在"}
            
        logger.info(f"开始使用 '{method}' 方法转换市场数据")
        
        try:
            encoder_func = self.encoding_methods[method]['function']
            result = encoder_func(market_data, **kwargs)
            
            logger.info(f"市场数据转换完成，使用方法: {method}")
            
            return {
                'status': 'success',
                'method': method,
                'quantum_data': result
            }
            
        except Exception as e:
            logger.error(f"转换市场数据时出错: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _amplitude_encoding(self, market_data: Dict[str, pd.DataFrame], 
                         feature: str = 'close', 
                         num_qubits: int = 4,
                         lookback: int = 10) -> Dict[str, Any]:
        """振幅编码方法"""
        results = {}
        
        for symbol, df in market_data.items():
            if feature not in df.columns or len(df) < lookback:
                logger.warning(f"无法为 {symbol} 执行振幅编码: 特征不存在或数据不足")
                continue
                
            # 获取最近的数据点
            data_points = df[feature].values[-lookback:]
            
            # 归一化数据
            min_val = np.min(data_points)
            max_val = np.max(data_points)
            
            if max_val > min_val:
                normalized_data = (data_points - min_val) / (max_val - min_val)
            else:
                normalized_data = np.ones_like(data_points) * 0.5
                
            # 根据量子比特数量，决定可以编码的数据点数量
            max_states = 2 ** num_qubits
            if len(normalized_data) > max_states:
                # 如果数据点过多，只保留最近的
                normalized_data = normalized_data[-max_states:]
                
            # 扩展到2^n大小并填充剩余部分为0
            padded_data = np.zeros(max_states)
            padded_data[:len(normalized_data)] = normalized_data
            
            # 归一化以确保平方和为1（量子态要求）
            l2_norm = np.sqrt(np.sum(padded_data ** 2))
            if l2_norm > 0:
                quantum_amplitudes = padded_data / l2_norm
            else:
                # 如果所有数据都是0，使用均匀分布
                quantum_amplitudes = np.ones(max_states) / np.sqrt(max_states)
                
            # 保存结果
            results[symbol] = {
                'method': 'amplitude',
                'num_qubits': num_qubits,
                'amplitudes': quantum_amplitudes.tolist(),
                'original_data': data_points.tolist(),
                'min_val': float(min_val),
                'max_val': float(max_val)
            }
            
        return results
        
    def _angle_encoding(self, market_data: Dict[str, pd.DataFrame], 
                     features: List[str] = None, 
                     scaling_factor: float = np.pi) -> Dict[str, Any]:
        """角度编码方法"""
        if features is None:
            features = ['open', 'high', 'low', 'close']
            
        results = {}
        
        for symbol, df in market_data.items():
            # 检查特征是否存在
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                logger.warning(f"无法为 {symbol} 执行角度编码: 没有可用特征")
                continue
                
            # 获取最新的数据点
            latest_data = df.iloc[-1]
            
            # 计算每个特征的角度
            angles = {}
            for feature in available_features:
                # 获取最近的值
                value = latest_data[feature]
                
                # 将值归一化到0-1范围（可根据需要调整归一化方法）
                if len(df) > 1:
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    if max_val > min_val:
                        norm_value = (value - min_val) / (max_val - min_val)
                    else:
                        norm_value = 0.5
                else:
                    norm_value = 0.5
                    
                # 转换为角度（0到pi或更广范围）
                angle = norm_value * scaling_factor
                
                angles[feature] = float(angle)
                
            # 保存结果
            results[symbol] = {
                'method': 'angle',
                'angles': angles,
                'scaling_factor': scaling_factor
            }
            
        return results
        
    def _binary_encoding(self, market_data: Dict[str, pd.DataFrame], 
                      feature: str = 'close', 
                      threshold: float = None) -> Dict[str, Any]:
        """二进制编码方法"""
        results = {}
        
        for symbol, df in market_data.items():
            if feature not in df.columns or len(df) < 2:
                logger.warning(f"无法为 {symbol} 执行二进制编码: 特征不存在或数据不足")
                continue
                
            # 获取最新值和前一个值
            current_value = df[feature].iloc[-1]
            previous_value = df[feature].iloc[-2]
            
            # 如果没有提供阈值，使用前一个值作为阈值
            if threshold is None:
                threshold = previous_value
                
            # 二进制编码：1表示高于阈值，0表示低于或等于阈值
            binary_state = 1 if current_value > threshold else 0
            
            # 计算变化百分比
            if previous_value != 0:
                percent_change = (current_value - previous_value) / previous_value * 100
            else:
                percent_change = 0
                
            # 保存结果
            results[symbol] = {
                'method': 'binary',
                'binary_state': binary_state,
                'current_value': float(current_value),
                'threshold': float(threshold),
                'percent_change': float(percent_change)
            }
            
        return results
        
    def get_encoding_methods(self) -> Dict[str, str]:
        """获取所有可用的编码方法"""
        return {name: info['description'] for name, info in self.encoding_methods.items()}

