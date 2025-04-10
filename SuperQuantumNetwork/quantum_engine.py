import numpy as np
from typing import Dict, List, Tuple

class QuantumEngine:
    """量子计算引擎 - 处理量子计算和共生效应"""
    
    def __init__(self, num_qubits: int):
        """初始化量子计算引擎
        
        Args:
            num_qubits: 量子比特数量
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0  # 初始化为|0⟩态
        
        # 量子门操作
        self.hadamard = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                                [1/np.sqrt(2), -1/np.sqrt(2)]])
        self.cnot = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])
        
        # 性能指标
        self.entanglement = 0.0
        self.coherence_time = 1.0
        self.gate_fidelity = 0.99
        
        self.qubits = []
        self.entanglement_matrix = None
        self.noise_level = 0.0
        self.symbiosis_effects = {
            'coherence_boost': 0.0,
            'resonance_boost': 0.0,
            'synergy_boost': 0.0,
            'stability_boost': 0.0
        }
        
    def initialize_qubits(self, num_qubits):
        """初始化量子比特"""
        self.qubits = [Qubit() for _ in range(num_qubits)]
        self.entanglement_matrix = np.zeros((num_qubits, num_qubits))
        
    def apply_symbiosis_effects(self, metrics):
        """应用共生效应到量子计算"""
        # 更新共生效应
        self.symbiosis_effects = {
            'coherence_boost': metrics['coherence'] * 0.1,
            'resonance_boost': metrics['resonance'] * 0.1,
            'synergy_boost': metrics['synergy'] * 0.1,
            'stability_boost': metrics['stability'] * 0.1
        }
        
        # 应用效应到量子比特
        for qubit in self.qubits:
            # 提高相干性
            qubit.coherence += self.symbiosis_effects['coherence_boost']
            # 提高稳定性
            qubit.stability += self.symbiosis_effects['stability_boost']
            
        # 更新纠缠矩阵
        self._update_entanglement_matrix()
        
    def _update_entanglement_matrix(self):
        """更新量子比特之间的纠缠关系"""
        num_qubits = len(self.qubits)
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                # 基于共振和协同性计算纠缠强度
                resonance = self.symbiosis_effects['resonance_boost']
                synergy = self.symbiosis_effects['synergy_boost']
                entanglement = (resonance + synergy) / 2
                
                self.entanglement_matrix[i][j] = entanglement
                self.entanglement_matrix[j][i] = entanglement
                
    def get_quantum_state(self):
        """获取当前量子状态"""
        state = "|ψ⟩ = "
        for i, qubit in enumerate(self.qubits):
            state += f"|{qubit.state}⟩"
            if i < len(self.qubits) - 1:
                state += " ⊗ "
        return state
        
    def get_entanglement_level(self):
        """获取整体纠缠度"""
        if self.entanglement_matrix is None:
            return 0.0
        return np.mean(self.entanglement_matrix)
        
    def get_noise_level(self):
        """获取量子噪声水平"""
        # 基于共生效应调整噪声水平
        base_noise = 0.1
        stability_factor = 1 - self.symbiosis_effects['stability_boost']
        return base_noise * stability_factor
        
    def apply_hadamard(self, qubit: int) -> None:
        """对指定量子比特应用Hadamard门
        
        Args:
            qubit: 目标量子比特的索引
        """
        if qubit >= self.num_qubits:
            raise ValueError("量子比特索引超出范围")
            
        # 更新量子态
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            target_bit = (i >> qubit) & 1
            new_i = i ^ (1 << qubit)
            new_state[i] += self.state[i] / np.sqrt(2)
            new_state[new_i] += (-1)**target_bit * self.state[i] / np.sqrt(2)
            
        self.state = new_state
        self._update_metrics()
        
    def apply_cnot(self, control: int, target: int) -> None:
        """应用CNOT门
        
        Args:
            control: 控制量子比特的索引
            target: 目标量子比特的索引
        """
        if control >= self.num_qubits or target >= self.num_qubits:
            raise ValueError("量子比特索引超出范围")
            
        # 更新量子态
        new_state = np.zeros_like(self.state)
        for i in range(len(self.state)):
            control_bit = (i >> control) & 1
            target_bit = (i >> target) & 1
            if control_bit == 1:
                new_i = i ^ (1 << target)
                new_state[new_i] = self.state[i]
            else:
                new_state[i] = self.state[i]
                
        self.state = new_state
        self._update_metrics()
        
    def measure(self) -> int:
        """测量量子态
        
        Returns:
            测量结果（整数）
        """
        probabilities = np.abs(self.state)**2
        result = np.random.choice(len(self.state), p=probabilities)
        return result
        
    def _update_metrics(self) -> None:
        """更新性能指标"""
        # 计算纠缠度
        density_matrix = np.outer(self.state, self.state.conj())
        self.entanglement = np.abs(np.trace(density_matrix @ density_matrix) - 1)
        
        # 更新相干时间（模拟衰减）
        self.coherence_time *= 0.95
        
        # 更新门保真度（模拟噪声影响）
        self.gate_fidelity *= 0.995
        
    def get_metrics(self) -> Dict:
        """获取性能指标
        
        Returns:
            包含各项指标的字典
        """
        return {
            'entanglement': self.entanglement,
            'coherence_time': self.coherence_time,
            'gate_fidelity': self.gate_fidelity,
            'num_qubits': self.num_qubits
        }

    def apply_phase_gate(self, qubit: int, phase: float) -> None:
        """应用相位门到指定量子比特
        
        Args:
            qubit: 目标量子比特的索引
            phase: 相位角度（弧度）
        """
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"量子比特索引 {qubit} 超出范围")
            
        # 相位门矩阵
        phase_matrix = np.array([[1, 0], [0, np.exp(1j * phase)]])
        
        # 更新量子态
        self._apply_single_qubit_gate(qubit, phase_matrix)
        self._update_metrics()

    def apply_swap_gate(self, qubit1: int, qubit2: int) -> None:
        """应用SWAP门交换两个量子比特的状态
        
        Args:
            qubit1: 第一个量子比特的索引
            qubit2: 第二个量子比特的索引
        """
        if not (0 <= qubit1 < self.num_qubits and 0 <= qubit2 < self.num_qubits):
            raise ValueError("量子比特索引超出范围")
        if qubit1 == qubit2:
            return
            
        # 构建SWAP门矩阵
        swap_matrix = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]])
        
        # 更新量子态
        self._apply_two_qubit_gate(qubit1, qubit2, swap_matrix)
        self._update_metrics()

    def get_state_vector(self) -> np.ndarray:
        """返回当前量子态向量"""
        return self.state.copy()

    def get_density_matrix(self) -> np.ndarray:
        """返回密度矩阵表示"""
        state_vector = self.state.reshape(-1, 1)
        return np.outer(state_vector, state_vector.conj())

    def apply_custom_gate(self, qubits: List[int], gate_matrix: np.ndarray) -> None:
        """应用自定义量子门
        
        Args:
            qubits: 目标量子比特的索引列表
            gate_matrix: 量子门矩阵
        """
        if not all(0 <= q < self.num_qubits for q in qubits):
            raise ValueError("量子比特索引超出范围")
            
        # 验证门矩阵维度
        expected_dim = 2 ** len(qubits)
        if gate_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"门矩阵维度必须为 {expected_dim}x{expected_dim}")
            
        # 更新量子态
        self._apply_multi_qubit_gate(qubits, gate_matrix)
        self._update_metrics()

    def _apply_multi_qubit_gate(self, qubits: List[int], gate_matrix: np.ndarray) -> None:
        """应用多量子比特门
        
        Args:
            qubits: 目标量子比特的索引列表
            gate_matrix: 量子门矩阵
        """
        # 构建完整的变换矩阵
        full_matrix = np.eye(2 ** self.num_qubits, dtype=complex)
        
        # 计算目标比特的掩码
        target_mask = sum(1 << q for q in qubits)
        
        # 应用门操作
        for i in range(2 ** self.num_qubits):
            if (i & target_mask) == i:  # 只处理目标比特
                for j in range(2 ** self.num_qubits):
                    if (j & target_mask) == j:  # 只处理目标比特
                        # 计算门矩阵的索引
                        gate_i = sum(((i >> q) & 1) << idx for idx, q in enumerate(qubits))
                        gate_j = sum(((j >> q) & 1) << idx for idx, q in enumerate(qubits))
                        full_matrix[i, j] = gate_matrix[gate_i, gate_j]
        
        # 更新量子态
        self.state = full_matrix @ self.state

    def get_entanglement_measure(self) -> float:
        """计算量子态的纠缠度量
        
        Returns:
            float: 纠缠度量值 (0-1之间)
        """
        # 使用von Neumann熵作为纠缠度量
        density_matrix = self.get_density_matrix()
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]  # 只考虑正特征值
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        max_entropy = np.log2(2 ** self.num_qubits)
        return entropy / max_entropy

    def get_state_fidelity(self, target_state: np.ndarray) -> float:
        """计算与目标态的保真度
        
        Args:
            target_state: 目标量子态向量
            
        Returns:
            float: 保真度 (0-1之间)
        """
        if target_state.shape != self.state.shape:
            raise ValueError("目标态维度不匹配")
            
        # 计算保真度 F = |<ψ|φ>|²
        fidelity = np.abs(np.vdot(target_state, self.state)) ** 2
        return float(fidelity)

class Qubit:
    """量子比特类"""
    
    def __init__(self):
        self.state = 0  # 0 或 1
        self.coherence = 1.0
        self.stability = 1.0 