"""
量子核心模块 - 超神系统的计算引擎

提供量子计算和高维数据处理的核心能力
"""

from .quantum_engine import QuantumEngine, get_quantum_engine
from .quantum_state import QuantumState
from .quantum_processor import QuantumProcessor, get_quantum_processor

__all__ = [
    'QuantumEngine', 'get_quantum_engine',
    'QuantumState',
    'QuantumProcessor', 'get_quantum_processor'
] 