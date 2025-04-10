"""
超神量子共生系统 - 包初始化文件
"""

from .quantum_errors import QuantumError, ModuleError, DataError, SystemError
from .quantum_engine import QuantumEngine
from .supergod_cockpit import SupergodCockpit
from .event import EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent
from .data import DataHandler
from .portfolio import SimplePortfolio
from .backtest import BacktestEngine

__version__ = '1.2.0'
__author__ = 'SuperGod Team' 