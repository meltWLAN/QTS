"""
quantum_interpreter - 量子核心组件
量子结果解释器 - 将量子计算结果解释为市场分析结果
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class QuantumInterpreter:
    """量子解释器 - 将量子计算结果转换为可用的市场分析"""
    
    def __init__(self):
        self.is_running = False
        self.interpretation_methods = {}
        self._register_default_interpreters()
        logger.info("量子解释器初始化完成")
        
    def start(self):
        """启动解释器"""
        if self.is_running:
            logger.warning("解释器已在运行")
            return
            
        logger.info("启动量子解释器...")
        self.is_running = True
        logger.info("量子解释器启动完成")
        
    def stop(self):
        """停止解释器"""
        if not self.is_running:
            logger.warning("解释器已停止")
            return
            
        logger.info("停止量子解释器...")
        self.is_running = False
        logger.info("量子解释器已停止")
        
    def _register_default_interpreters(self):
        """注册默认解释方法"""
        self.register_interpretation_method('probability', self._probability_interpretation,
                                    '概率解释 - 根据概率分布解释量子结果')
        self.register_interpretation_method('threshold', self._threshold_interpretation,
                                    '阈值解释 - 使用阈值确定信号方向')
        self.register_interpretation_method('relative', self._relative_interpretation,
                                    '相对解释 - 比较不同状态的概率差异')
        
        logger.info(f"注册了 {len(self.interpretation_methods)} 个默认解释方法")
        
    def register_interpretation_method(self, name: str, interpreter_func, description: str = ""):
        """注册解释方法"""
        if name in self.interpretation_methods:
            logger.warning(f"解释方法 '{name}' 已存在，将被替换")
            
        self.interpretation_methods[name] = {
            'function': interpreter_func,
            'description': description
        }
        
        logger.info(f"注册解释方法: {name}")
        return True
        
    def unregister_interpretation_method(self, name: str):
        """注销解释方法"""
        if name not in self.interpretation_methods:
            logger.warning(f"解释方法 '{name}' 不存在")
            return False
            
        del self.interpretation_methods[name]
        logger.info(f"注销解释方法: {name}")
        return True
        
    def interpret(self, quantum_result: Dict[str, Any], method: str = 'probability',
                **kwargs) -> Dict[str, Any]:
        """解释量子计算结果"""
        if not self.is_running:
            logger.warning("解释器未运行，无法执行解释")
            return {'status': 'error', 'message': '解释器未运行'}
            
        if method not in self.interpretation_methods:
            logger.warning(f"解释方法 '{method}' 不存在")
            return {'status': 'error', 'message': f"解释方法 '{method}' 不存在"}
            
        if 'counts' not in quantum_result:
            logger.warning("量子结果缺少必要的计数数据")
            return {'status': 'error', 'message': '量子结果格式不正确，缺少counts字段'}
            
        logger.info(f"开始使用 '{method}' 方法解释量子结果")
        
        try:
            interpreter_func = self.interpretation_methods[method]['function']
            interpretation = interpreter_func(quantum_result, **kwargs)
            
            logger.info(f"量子结果解释完成，使用方法: {method}")
            
            return {
                'status': 'success',
                'method': method,
                'interpretation': interpretation,
                'original_counts': quantum_result.get('counts', {})
            }
            
        except Exception as e:
            logger.error(f"解释量子结果时出错: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _probability_interpretation(self, quantum_result: Dict[str, Any], 
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """概率解释方法"""
        counts = quantum_result.get('counts', {})
        shots = quantum_result.get('shots', sum(counts.values()))
        
        if shots == 0:
            return {'error': '没有有效的测量结果'}
            
        # 计算每个状态的概率
        probabilities = {state: count / shots for state, count in counts.items()}
        
        # 按概率排序
        sorted_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # 获取最可能的状态
        most_probable_state = sorted_states[0][0] if sorted_states else None
        
        # 计算上升/下降的总概率
        # 假设最高位为1表示上升趋势，为0表示下降趋势
        up_probability = sum(prob for state, prob in probabilities.items() 
                           if state and state[0] == '1')
        down_probability = sum(prob for state, prob in probabilities.items() 
                             if state and state[0] == '0')
        
        # 确定信号方向
        if up_probability > threshold:
            signal = 'strong_buy'
        elif up_probability > 0.5:
            signal = 'buy'
        elif down_probability > threshold:
            signal = 'strong_sell'
        elif down_probability > 0.5:
            signal = 'sell'
        else:
            signal = 'neutral'
            
        # 计算信号强度 (0-100)
        if signal in ['strong_buy', 'buy']:
            strength = up_probability * 100
        elif signal in ['strong_sell', 'sell']:
            strength = down_probability * 100
        else:
            strength = 50.0
            
        return {
            'signal': signal,
            'strength': float(strength),
            'up_probability': float(up_probability),
            'down_probability': float(down_probability),
            'most_probable_state': most_probable_state,
            'probabilities': probabilities
        }
        
    def _threshold_interpretation(self, quantum_result: Dict[str, Any], 
                               up_states: List[str] = None,
                               down_states: List[str] = None) -> Dict[str, Any]:
        """阈值解释方法"""
        counts = quantum_result.get('counts', {})
        shots = quantum_result.get('shots', sum(counts.values()))
        
        if shots == 0:
            return {'error': '没有有效的测量结果'}
            
        # 如果没有指定上升/下降状态，使用默认规则
        if up_states is None:
            # 假设二进制表示中，最高位为1的状态表示上升趋势
            up_states = [state for state in counts.keys() if state and state[0] == '1']
            
        if down_states is None:
            # 假设二进制表示中，最高位为0的状态表示下降趋势
            down_states = [state for state in counts.keys() if state and state[0] == '0']
            
        # 计算上升/下降状态的总计数
        up_counts = sum(counts.get(state, 0) for state in up_states)
        down_counts = sum(counts.get(state, 0) for state in down_states)
        
        # 计算比率
        if shots > 0:
            up_ratio = up_counts / shots
            down_ratio = down_counts / shots
        else:
            up_ratio = down_ratio = 0
            
        # 计算信号强度 (0-100)
        if up_ratio > down_ratio:
            signal = 'buy' if up_ratio < 0.7 else 'strong_buy'
            strength = up_ratio * 100
        elif down_ratio > up_ratio:
            signal = 'sell' if down_ratio < 0.7 else 'strong_sell'
            strength = down_ratio * 100
        else:
            signal = 'neutral'
            strength = 50.0
            
        return {
            'signal': signal,
            'strength': float(strength),
            'up_ratio': float(up_ratio),
            'down_ratio': float(down_ratio),
            'up_states': up_states,
            'down_states': down_states
        }
        
    def _relative_interpretation(self, quantum_result: Dict[str, Any], 
                              reference_state: str = None) -> Dict[str, Any]:
        """相对解释方法"""
        counts = quantum_result.get('counts', {})
        shots = quantum_result.get('shots', sum(counts.values()))
        
        if shots == 0:
            return {'error': '没有有效的测量结果'}
            
        # 如果没有指定参考状态，使用最频繁的状态
        if reference_state is None:
            reference_state = max(counts.items(), key=lambda x: x[1])[0]
            
        # 计算每个状态相对于参考状态的比率
        reference_count = counts.get(reference_state, 0)
        
        if reference_count > 0:
            relative_ratios = {state: count / reference_count 
                             for state, count in counts.items()}
        else:
            relative_ratios = {state: 0 for state in counts.keys()}
            
        # 计算平均偏差
        if len(relative_ratios) > 1:
            avg_deviation = np.mean([abs(ratio - 1.0) for state, ratio in relative_ratios.items() 
                                  if state != reference_state])
        else:
            avg_deviation = 0.0
            
        # 根据平均偏差确定一致性
        if avg_deviation < 0.2:
            consistency = 'high'
        elif avg_deviation < 0.5:
            consistency = 'medium'
        else:
            consistency = 'low'
            
        # 确定最显著的状态（除参考状态外）
        significant_states = {state: ratio for state, ratio in relative_ratios.items() 
                            if state != reference_state and ratio > 1.0}
        
        if significant_states:
            most_significant = max(significant_states.items(), key=lambda x: x[1])
        else:
            most_significant = (reference_state, 1.0)
            
        return {
            'reference_state': reference_state,
            'relative_ratios': relative_ratios,
            'avg_deviation': float(avg_deviation),
            'consistency': consistency,
            'most_significant_state': most_significant[0],
            'significance_ratio': float(most_significant[1])
        }
        
    def get_interpretation_methods(self) -> Dict[str, str]:
        """获取所有可用的解释方法"""
        return {name: info['description'] for name, info in self.interpretation_methods.items()}

