#!/usr/bin/env python3
"""
量子状态 - 管理和存储量子系统状态

提供量子状态的表示、存储和操作
"""

import logging
import numpy as np
import random
from datetime import datetime
import json
import os

class QuantumState:
    """量子状态类
    
    管理和存储量子系统的状态
    """
    
    def __init__(self, name, dimensions=12, initial_state=None):
        """初始化量子状态
        
        Args:
            name: 状态名称
            dimensions: 量子维度
            initial_state: 初始状态字典
        """
        self.logger = logging.getLogger("QuantumState")
        self.name = name
        self.dimensions = dimensions
        
        # 创建状态
        self.state = initial_state if initial_state else {
            "quantum_vector": self._create_random_state_vector(dimensions),
            "coherence": random.uniform(0.7, 0.9),
            "entanglement": random.uniform(0.6, 0.8),
            "dimensional_access": list(range(3, min(dimensions + 1, 12))),
            "stability": random.uniform(0.7, 0.9),
            "created_at": datetime.now(),
            "last_update": datetime.now(),
            "evolution_step": 0,
            "metadata": {}
        }
        
        self.logger.debug(f"创建量子状态 '{name}', 维度: {dimensions}")
    
    def evolve(self, evolution_operator=None):
        """演化量子状态
        
        Args:
            evolution_operator: 演化算子，如果为None则使用随机演化
            
        Returns:
            bool: 演化是否成功
        """
        try:
            # 增加演化步数
            self.state["evolution_step"] += 1
            
            # 随机波动
            coherence_change = random.uniform(-0.05, 0.05)
            self.state["coherence"] = max(0.3, min(0.95, 
                self.state["coherence"] + coherence_change))
                
            entanglement_change = random.uniform(-0.03, 0.03)
            self.state["entanglement"] = max(0.2, min(0.9, 
                self.state["entanglement"] + entanglement_change))
                
            stability_change = random.uniform(-0.04, 0.04)
            self.state["stability"] = max(0.4, min(0.95, 
                self.state["stability"] + stability_change))
            
            # 应用演化算子
            if evolution_operator:
                # 使用提供的演化算子
                self.state["quantum_vector"] = evolution_operator(self.state["quantum_vector"])
            else:
                # 随机演化，保持规范化
                phase_shifts = np.exp(1j * np.random.uniform(0, 0.1, len(self.state["quantum_vector"])))
                evolved_vector = self.state["quantum_vector"] * phase_shifts
                self.state["quantum_vector"] = self._normalize_vector(evolved_vector)
            
            # 更新时间戳
            self.state["last_update"] = datetime.now()
            
            self.logger.debug(f"量子状态 '{self.name}' 演化到步骤 {self.state['evolution_step']}")
            return True
            
        except Exception as e:
            self.logger.error(f"量子状态演化失败: {str(e)}")
            return False
    
    def collapse(self, measurement_basis=None):
        """坍缩量子状态
        
        Args:
            measurement_basis: 测量基，如果为None则使用标准基
            
        Returns:
            tuple: (测量结果, 坍缩状态)
        """
        try:
            # 执行量子测量
            if measurement_basis:
                # 使用指定测量基
                probabilities = []
                for basis_vector in measurement_basis:
                    # 计算投影概率
                    probability = abs(np.vdot(basis_vector, self.state["quantum_vector"])) ** 2
                    probabilities.append(probability)
                    
                # 归一化
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                else:
                    # 如果总概率为0，使用均匀分布
                    probabilities = [1.0 / len(measurement_basis) for _ in measurement_basis]
                    
                # 随机选择结果
                result_index = np.random.choice(len(measurement_basis), p=probabilities)
                result = result_index
                
                # 坍缩到测量结果
                self.state["quantum_vector"] = measurement_basis[result_index]
                
            else:
                # 使用标准基测量
                probabilities = [abs(amplitude) ** 2 for amplitude in self.state["quantum_vector"]]
                
                # 归一化
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                else:
                    # 如果总概率为0，使用均匀分布
                    probabilities = [1.0 / len(probabilities) for _ in probabilities]
                    
                # 随机选择结果
                result_index = np.random.choice(len(probabilities), p=probabilities)
                result = result_index
                
                # 坍缩到测量结果
                collapsed_vector = np.zeros_like(self.state["quantum_vector"])
                collapsed_vector[result_index] = 1.0
                self.state["quantum_vector"] = collapsed_vector
            
            # 坍缩后的状态变化
            self.state["coherence"] *= random.uniform(0.7, 0.9)  # 减少相干性
            self.state["entanglement"] *= random.uniform(0.6, 0.8)  # 减少纠缠
            self.state["stability"] = max(0.95, self.state["stability"] * random.uniform(1.1, 1.3))  # 增加稳定性
            
            # 更新时间戳
            self.state["last_update"] = datetime.now()
            
            # 添加测量记录
            if "measurements" not in self.state["metadata"]:
                self.state["metadata"]["measurements"] = []
                
            self.state["metadata"]["measurements"].append({
                "result": result,
                "time": datetime.now(),
                "stability_after": self.state["stability"]
            })
            
            self.logger.debug(f"量子状态 '{self.name}' 坍缩到结果 {result}")
            
            # 返回结果和坍缩后的状态
            return result, self.state
            
        except Exception as e:
            self.logger.error(f"量子状态坍缩失败: {str(e)}")
            return None, None
    
    def entangle(self, other_state):
        """与另一个量子状态纠缠
        
        Args:
            other_state: 另一个量子状态对象
            
        Returns:
            bool: 纠缠是否成功
        """
        try:
            # 检查维度兼容性
            if self.dimensions != other_state.dimensions:
                self.logger.warning(f"维度不兼容: {self.dimensions} vs {other_state.dimensions}")
                return False
                
            # 模拟纠缠过程
            entanglement_strength = min(self.state["entanglement"], other_state.state["entanglement"])
            
            # 随机纠缠
            if random.random() < entanglement_strength:
                # 增强纠缠度
                self.state["entanglement"] = min(0.95, self.state["entanglement"] * random.uniform(1.05, 1.1))
                other_state.state["entanglement"] = min(0.95, other_state.state["entanglement"] * random.uniform(1.05, 1.1))
                
                # 记录纠缠关系
                if "entangled_with" not in self.state["metadata"]:
                    self.state["metadata"]["entangled_with"] = []
                    
                if "entangled_with" not in other_state.state["metadata"]:
                    other_state.state["metadata"]["entangled_with"] = []
                    
                # 添加纠缠记录
                if other_state.name not in self.state["metadata"]["entangled_with"]:
                    self.state["metadata"]["entangled_with"].append(other_state.name)
                    
                if self.name not in other_state.state["metadata"]["entangled_with"]:
                    other_state.state["metadata"]["entangled_with"].append(self.name)
                    
                # 更新时间戳
                self.state["last_update"] = datetime.now()
                other_state.state["last_update"] = datetime.now()
                
                self.logger.debug(f"量子状态 '{self.name}' 与 '{other_state.name}' 成功纠缠")
                return True
            else:
                self.logger.debug(f"量子状态 '{self.name}' 与 '{other_state.name}' 纠缠失败")
                return False
                
        except Exception as e:
            self.logger.error(f"量子状态纠缠失败: {str(e)}")
            return False
    
    def superposition(self, other_states, weights=None):
        """创建与其他状态的叠加态
        
        Args:
            other_states: 其他量子状态对象的列表
            weights: 权重列表，如果为None则使用均匀权重
            
        Returns:
            QuantumState: 新的叠加态
        """
        try:
            # 检查维度兼容性
            for state in other_states:
                if self.dimensions != state.dimensions:
                    self.logger.warning(f"维度不兼容: {self.dimensions} vs {state.dimensions}")
                    return None
            
            # 创建叠加态
            if weights is None:
                # 使用均匀权重
                weights = [1.0] * (len(other_states) + 1)
                
            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # 创建初始叠加态向量
            superposition_vector = self.state["quantum_vector"] * weights[0]
            
            # 添加其他状态
            for i, state in enumerate(other_states):
                superposition_vector += state.state["quantum_vector"] * weights[i + 1]
                
            # 归一化
            superposition_vector = self._normalize_vector(superposition_vector)
            
            # 计算平均相干性和纠缠度
            coherence = self.state["coherence"] * weights[0]
            entanglement = self.state["entanglement"] * weights[0]
            stability = self.state["stability"] * weights[0]
            
            for i, state in enumerate(other_states):
                coherence += state.state["coherence"] * weights[i + 1]
                entanglement += state.state["entanglement"] * weights[i + 1]
                stability += state.state["stability"] * weights[i + 1]
                
            # 创建新状态对象
            combined_names = self.name + "_" + "_".join([s.name for s in other_states])
            super_state = QuantumState(
                name=f"superposition_{combined_names}",
                dimensions=self.dimensions,
                initial_state={
                    "quantum_vector": superposition_vector,
                    "coherence": coherence,
                    "entanglement": entanglement,
                    "dimensional_access": self.state["dimensional_access"],
                    "stability": stability * random.uniform(0.8, 0.9),  # 叠加态通常不太稳定
                    "created_at": datetime.now(),
                    "last_update": datetime.now(),
                    "evolution_step": 0,
                    "metadata": {
                        "parent_states": [self.name] + [s.name for s in other_states],
                        "weights": weights
                    }
                }
            )
            
            self.logger.debug(f"创建叠加态 '{super_state.name}'")
            return super_state
            
        except Exception as e:
            self.logger.error(f"创建叠加态失败: {str(e)}")
            return None
    
    def save(self, directory="quantum_states"):
        """保存量子状态到文件
        
        Args:
            directory: 保存目录
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(directory, exist_ok=True)
            
            # 创建文件路径
            filepath = os.path.join(directory, f"{self.name}.json")
            
            # 准备状态数据
            state_data = self.state.copy()
            
            # 转换numpy数组
            state_data["quantum_vector"] = [complex(v.real, v.imag) for v in self.state["quantum_vector"]]
            
            # 转换日期时间
            state_data["created_at"] = state_data["created_at"].isoformat()
            state_data["last_update"] = state_data["last_update"].isoformat()
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.debug(f"量子状态 '{self.name}' 已保存到 {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存量子状态失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, name, directory="quantum_states"):
        """从文件加载量子状态
        
        Args:
            name: 状态名称
            directory: 保存目录
            
        Returns:
            QuantumState: 加载的量子状态对象，如果失败则返回None
        """
        try:
            # 创建文件路径
            filepath = os.path.join(directory, f"{name}.json")
            
            # 检查文件是否存在
            if not os.path.exists(filepath):
                logging.warning(f"量子状态文件不存在: {filepath}")
                return None
                
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                
            # 转换复数
            state_data["quantum_vector"] = np.array([complex(v[0], v[1]) if isinstance(v, list) else complex(v.real, v.imag) for v in state_data["quantum_vector"]])
            
            # 转换日期时间
            state_data["created_at"] = datetime.fromisoformat(state_data["created_at"])
            state_data["last_update"] = datetime.fromisoformat(state_data["last_update"])
            
            # 创建维度
            dimensions = len(state_data["dimensional_access"])
            
            # 创建量子状态对象
            quantum_state = cls(name=name, dimensions=dimensions, initial_state=state_data)
            
            logging.debug(f"量子状态 '{name}' 已从 {filepath} 加载")
            return quantum_state
            
        except Exception as e:
            logging.error(f"加载量子状态失败: {str(e)}")
            return None
    
    def _create_random_state_vector(self, dimensions):
        """创建随机状态向量
        
        Args:
            dimensions: 向量维度
            
        Returns:
            numpy.ndarray: 随机状态向量
        """
        # 创建随机复数向量
        vector = np.random.normal(0, 1, dimensions) + 1j * np.random.normal(0, 1, dimensions)
        
        # 归一化
        return self._normalize_vector(vector)
    
    def _normalize_vector(self, vector):
        """归一化向量
        
        Args:
            vector: 输入向量
            
        Returns:
            numpy.ndarray: 归一化后的向量
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            # 如果范数为0，返回标准基的第一个向量
            result = np.zeros_like(vector)
            result[0] = 1.0
            return result 