from typing import Dict, List
import numpy as np

class SymbiosisManager:
    """共生系统管理器 - 处理共生效应和量子计算交互"""
    
    def __init__(self, quantum_engine):
        self.quantum_engine = quantum_engine
        self.metrics = {
            'coherence': 1.0,      # 量子相干性
            'resonance': 1.0,      # 量子共振
            'synergy': 1.0,        # 量子协同
            'stability': 1.0       # 系统稳定性
        }
        self.symbiosis_state = 'initializing'
        
        # 历史记录
        self.history = {
            'coherence': [],
            'resonance': [],
            'synergy': [],
            'stability': []
        }
        
    def update_metrics(self, new_metrics: Dict[str, float]) -> None:
        """更新共生效应指标
        
        Args:
            new_metrics: 新的指标值字典
        """
        for key, value in new_metrics.items():
            if key in self.metrics:
                # 确保值在[0,1]范围内
                self.metrics[key] = max(0.0, min(1.0, value))
                # 记录历史
                self.history[key].append(self.metrics[key])
                
    def _update_symbiosis_state(self):
        """更新共生状态"""
        avg_metric = sum(self.metrics.values()) / len(self.metrics)
        if avg_metric > 0.8:
            self.symbiosis_state = 'optimal'
        elif avg_metric > 0.5:
            self.symbiosis_state = 'stable'
        elif avg_metric > 0.2:
            self.symbiosis_state = 'developing'
        else:
            self.symbiosis_state = 'unstable'
            
    def get_quantum_metrics(self):
        """获取量子计算相关的指标"""
        return {
            'entanglement_level': self.quantum_engine.get_entanglement_level(),
            'noise_level': self.quantum_engine.get_noise_level(),
            'quantum_state': self.quantum_engine.get_quantum_state()
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """获取当前共生效应指标
        
        Returns:
            当前指标值字典
        """
        return self.metrics.copy()
        
    def get_history(self) -> Dict[str, List[float]]:
        """获取历史记录
        
        Returns:
            历史记录字典
        """
        return self.history.copy()
        
    def calculate_symbiosis_score(self) -> float:
        """计算总体共生得分
        
        Returns:
            0到1之间的得分
        """
        weights = {
            'coherence': 0.3,
            'resonance': 0.2,
            'synergy': 0.3,
            'stability': 0.2
        }
        
        score = sum(self.metrics[key] * weight 
                   for key, weight in weights.items())
        return score
        
    def get_report(self) -> Dict:
        """生成共生系统报告
        
        Returns:
            包含详细信息的报告字典
        """
        quantum_metrics = self.get_quantum_metrics()
        return {
            'current_metrics': self.get_metrics(),
            'symbiosis_score': self.calculate_symbiosis_score(),
            'trends': {
                key: np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
                for key, values in self.history.items()
            },
            'symbiosis_state': self.symbiosis_state,
            'quantum_metrics': quantum_metrics,
            'timestamp': datetime.now().isoformat()
        } 