#!/usr/bin/env python3
"""
超神量子共生系统 - 交易引擎包
实现真实交易功能的核心组件
"""

__version__ = "1.0.0"
__author__ = "SupergodTeam"

# 导出主要组件
from .trading_core import TradingEngine, OrderType, OrderStatus, Position
from .risk_manager import RiskManager
from .performance_analyzer import PerformanceAnalyzer

# 模块初始化日志
import logging
logger = logging.getLogger("TradingEngine")
logger.info("超神交易引擎模块已加载") 