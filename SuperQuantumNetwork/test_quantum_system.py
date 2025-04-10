import unittest
import numpy as np
from quantum_engine import QuantumEngine
from symbiosis_manager import SymbiosisManager

class TestQuantumEngine(unittest.TestCase):
    """测试量子计算引擎的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.quantum_engine = QuantumEngine(num_qubits=2)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.quantum_engine.num_qubits, 2)
        self.assertEqual(len(self.quantum_engine.state), 4)  # 2^2 = 4
        self.assertTrue(np.allclose(self.quantum_engine.state[0], 1.0))
        self.assertTrue(np.allclose(np.sum(np.abs(self.quantum_engine.state)**2), 1.0))
        
    def test_hadamard_gate(self):
        """测试Hadamard门操作"""
        # 对第一个量子比特应用Hadamard门
        self.quantum_engine.apply_hadamard(0)
        # 检查状态是否正确
        expected_state = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])
        self.assertTrue(np.allclose(self.quantum_engine.state, expected_state))
        
    def test_cnot_gate(self):
        """测试CNOT门操作"""
        # 先对第一个量子比特应用Hadamard门
        self.quantum_engine.apply_hadamard(0)
        # 然后应用CNOT门
        self.quantum_engine.apply_cnot(0, 1)
        # 检查状态是否正确
        expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        self.assertTrue(np.allclose(self.quantum_engine.state, expected_state))
        
    def test_measurement(self):
        """测试量子测量"""
        # 对第一个量子比特应用Hadamard门
        self.quantum_engine.apply_hadamard(0)
        # 进行多次测量，检查结果分布
        results = [self.quantum_engine.measure() for _ in range(1000)]
        # 检查结果是否在合理范围内
        self.assertTrue(all(0 <= result < 4 for result in results))
        
    def test_metrics(self):
        """测试性能指标"""
        # 应用一些量子门操作
        self.quantum_engine.apply_hadamard(0)
        self.quantum_engine.apply_cnot(0, 1)
        # 获取指标
        metrics = self.quantum_engine.get_metrics()
        # 检查指标是否存在且合理
        self.assertIn('entanglement', metrics)
        self.assertIn('coherence_time', metrics)
        self.assertIn('gate_fidelity', metrics)
        self.assertTrue(0 <= metrics['entanglement'] <= 1)
        self.assertTrue(0 <= metrics['coherence_time'] <= 1)
        self.assertTrue(0 <= metrics['gate_fidelity'] <= 1)

class TestSymbiosisManager(unittest.TestCase):
    """测试共生系统管理器的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.quantum_engine = QuantumEngine(num_qubits=2)
        self.symbiosis_manager = SymbiosisManager(self.quantum_engine)
        
    def test_initialization(self):
        """测试初始化"""
        metrics = self.symbiosis_manager.get_metrics()
        self.assertEqual(metrics['coherence'], 1.0)
        self.assertEqual(metrics['resonance'], 1.0)
        self.assertEqual(metrics['synergy'], 1.0)
        self.assertEqual(metrics['stability'], 1.0)
        
    def test_update_metrics(self):
        """测试更新指标"""
        new_metrics = {
            'coherence': 0.8,
            'resonance': 0.7,
            'synergy': 0.9,
            'stability': 0.6
        }
        self.symbiosis_manager.update_metrics(new_metrics)
        updated_metrics = self.symbiosis_manager.get_metrics()
        self.assertEqual(updated_metrics, new_metrics)
        
    def test_history_recording(self):
        """测试历史记录"""
        # 更新两次指标
        self.symbiosis_manager.update_metrics({
            'coherence': 0.8,
            'resonance': 0.7,
            'synergy': 0.9,
            'stability': 0.6
        })
        self.symbiosis_manager.update_metrics({
            'coherence': 0.7,
            'resonance': 0.6,
            'synergy': 0.8,
            'stability': 0.5
        })
        # 获取历史记录
        history = self.symbiosis_manager.get_history()
        # 检查历史记录长度
        self.assertEqual(len(history['coherence']), 2)
        self.assertEqual(len(history['resonance']), 2)
        self.assertEqual(len(history['synergy']), 2)
        self.assertEqual(len(history['stability']), 2)
        # 检查历史记录值
        self.assertEqual(history['coherence'][0], 0.8)
        self.assertEqual(history['coherence'][1], 0.7)
        
    def test_symbiosis_score(self):
        """测试共生得分计算"""
        # 设置指标
        self.symbiosis_manager.update_metrics({
            'coherence': 0.8,
            'resonance': 0.7,
            'synergy': 0.9,
            'stability': 0.6
        })
        # 计算得分
        score = self.symbiosis_manager.calculate_symbiosis_score()
        # 检查得分是否在合理范围内
        self.assertTrue(0 <= score <= 1)
        
    def test_report_generation(self):
        """测试报告生成"""
        # 更新指标
        self.symbiosis_manager.update_metrics({
            'coherence': 0.8,
            'resonance': 0.7,
            'synergy': 0.9,
            'stability': 0.6
        })
        # 生成报告
        report = self.symbiosis_manager.get_report()
        # 检查报告内容
        self.assertIn('current_metrics', report)
        self.assertIn('symbiosis_score', report)
        self.assertIn('trends', report)
        self.assertIn('symbiosis_state', report)
        self.assertIn('quantum_metrics', report)
        self.assertIn('timestamp', report)

if __name__ == '__main__':
    unittest.main() 