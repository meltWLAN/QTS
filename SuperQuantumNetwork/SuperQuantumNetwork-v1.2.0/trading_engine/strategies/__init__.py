"""
交易引擎 - 策略模块
此包包含各种交易策略的实现
"""
from typing import Dict, List, Type

# 导入策略类
try:
    from .quantum_strategy import QuantumStrategy
except ImportError:
    pass

# 策略注册表
STRATEGY_REGISTRY = {}

def register_strategy(strategy_class):
    """注册策略类"""
    STRATEGY_REGISTRY[strategy_class.__name__] = strategy_class
    return strategy_class

# 注册内置策略
for strategy_class_name in list(locals().keys()):
    strategy_class = locals()[strategy_class_name]
    if isinstance(strategy_class, type) and strategy_class_name.endswith('Strategy'):
        register_strategy(strategy_class)

def get_strategy_class(strategy_name: str) -> Type:
    """获取策略类"""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {strategy_name}")
    return STRATEGY_REGISTRY[strategy_name]

def get_available_strategies() -> List[str]:
    """获取可用策略列表"""
    return list(STRATEGY_REGISTRY.keys())

def create_strategy(strategy_name: str, config: Dict = None):
    """创建策略实例"""
    strategy_class = get_strategy_class(strategy_name)
    return strategy_class(config)