from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

class EventType(Enum):
    MARKET_DATA = "MARKET_DATA"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO = "PORTFOLIO"

@dataclass
class MarketDataEvent:
    type: EventType = EventType.MARKET_DATA
    symbol: str = ""
    timestamp: datetime = None
    data: Dict[str, Any] = None

@dataclass
class SignalEvent:
    type: EventType = EventType.SIGNAL
    symbol: str = ""
    timestamp: datetime = None
    signal_type: str = ""
    strength: float = 0.0
    direction: str = ""
    price: float = 0.0

@dataclass
class OrderEvent:
    type: EventType = EventType.ORDER
    symbol: str = ""
    timestamp: datetime = None
    order_type: str = ""
    quantity: int = 0
    direction: str = ""
    price: float = 0.0

@dataclass
class FillEvent:
    type: EventType = EventType.FILL
    symbol: str = ""
    timestamp: datetime = None
    quantity: int = 0
    direction: str = ""
    price: float = 0.0
    commission: float = 0.0

@dataclass
class PortfolioEvent:
    type: EventType = EventType.PORTFOLIO
    timestamp: datetime = None
    positions: Dict[str, int] = None
    equity: float = 0.0
    cash: float = 0.0 