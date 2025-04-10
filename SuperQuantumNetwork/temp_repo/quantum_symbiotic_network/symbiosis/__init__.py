#!/usr/bin/env python3
"""
超神量子共生网络 - 共生核心
实现各个模块间的深度联动和共生增强
"""

from .symbiosis_core import SymbiosisCore
from .hyperunity import HyperUnityField

# 全局实例
_symbiosis_core = None
_hyperunity_field = None

def get_symbiosis_core():
    """获取共生核心的单例实例

    Returns:
        SymbiosisCore: 共生核心实例
    """
    global _symbiosis_core
    if _symbiosis_core is None:
        _symbiosis_core = SymbiosisCore()
    return _symbiosis_core

def get_hyperunity_field():
    """获取高维统一场单例实例
    
    Returns:
        HyperUnityField: 高维统一场实例
    """
    global _hyperunity_field
    if _hyperunity_field is None:
        _hyperunity_field = HyperUnityField()
    return _hyperunity_field

__all__ = ["SymbiosisCore", "get_symbiosis_core", "get_hyperunity_field"] 