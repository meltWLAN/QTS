from typing import Dict, List, Any, Optional
from datetime import datetime
from event import PortfolioEvent, EventType
from data import DataHandler

class SimplePortfolio:
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # 持仓: {symbol: quantity}
        self.position_values = {}  # 持仓市值: {symbol: value}
        self.equity = initial_capital  # 权益 = 现金 + 持仓市值
        self.history = []  # 投资组合历史状态
        self.data_handler = None

    def update(self, fill_event) -> None:
        """根据成交事件更新投资组合"""
        symbol = fill_event.symbol
        quantity = fill_event.quantity
        direction = fill_event.direction
        price = fill_event.price
        commission = fill_event.commission
        
        # 更新现金和持仓
        if direction == 'BUY':
            cost = price * quantity + commission
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif direction == 'SELL':
            proceed = price * quantity - commission
            self.cash += proceed
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            
            # 如果持仓为0，从字典中删除
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
        
        # 更新持仓市值和权益
        self._update_position_values()
        self._update_equity()
        
        # 记录投资组合状态
        self._record_portfolio_event(fill_event.timestamp)
        
    def _update_position_values(self) -> None:
        """更新持仓市值"""
        self.position_values = {}
        
        if not self.data_handler:
            return
            
        for symbol, quantity in self.positions.items():
            latest_data = self.data_handler.get_latest_data(symbol)
            if latest_data:
                price = latest_data.get('close', 0)
                self.position_values[symbol] = price * quantity
                
    def _update_equity(self) -> None:
        """更新权益"""
        position_value = sum(self.position_values.values())
        self.equity = self.cash + position_value
    
    def _record_portfolio_event(self, timestamp: datetime) -> None:
        """记录投资组合状态"""
        event = PortfolioEvent(
            timestamp=timestamp,
            positions=self.positions.copy(),
            equity=self.equity,
            cash=self.cash
        )
        self.history.append(event)
        
    def update_daily(self, date: datetime, market_data: Dict[str, Dict[str, Any]]) -> None:
        """更新每日投资组合状态"""
        # 更新持仓市值
        self.position_values = {}
        
        for symbol, quantity in self.positions.items():
            if symbol in market_data:
                price = float(market_data[symbol].get('close', 0))
                self.position_values[symbol] = price * quantity
        
        # 更新权益
        self._update_equity()
        
        # 记录投资组合状态
        self._record_portfolio_event(date)

    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        return self.positions.get(symbol, 0)

    def get_cash(self) -> float:
        """获取可用现金"""
        return self.cash

    def get_equity(self) -> float:
        """获取总资产"""
        return self.equity 