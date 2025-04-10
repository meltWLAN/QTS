#!/usr/bin/env python3
"""
超神量子共生系统 - 交易引擎核心
实现交易执行、订单管理和持仓管理
"""

import logging
import uuid
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

# 设置日志
logger = logging.getLogger("TradingEngine.Core")

# 订单类型枚举
class OrderType(Enum):
    """订单类型"""
    MARKET = "市价单"
    LIMIT = "限价单"
    STOP = "止损单"
    STOP_LIMIT = "止损限价单"

# 订单状态枚举
class OrderStatus(Enum):
    """订单状态"""
    PENDING = "等待中"
    PARTIAL = "部分成交"
    FILLED = "已成交"
    CANCELED = "已取消"
    REJECTED = "已拒绝"
    EXPIRED = "已过期"

# 交易方向枚举
class OrderDirection(Enum):
    """交易方向"""
    BUY = "买入"
    SELL = "卖出"

class Order:
    """订单类"""
    
    def __init__(self, symbol: str, direction: OrderDirection, quantity: float, 
                 order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
                 stop_price: Optional[float] = None, client_order_id: Optional[str] = None):
        """
        初始化订单
        
        参数:
            symbol: 股票代码
            direction: 交易方向
            quantity: 数量
            order_type: 订单类型
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
            client_order_id: 客户端订单ID
        """
        self.order_id = str(uuid.uuid4())[:12]
        self.client_order_id = client_order_id or f"supergod_{int(time.time())}"
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        self.commission = 0.0
        self.created_time = datetime.now()
        self.updated_time = self.created_time
        self.filled_time = None
        self.cancel_time = None
        self.rejection_reason = None
        
    def update_status(self, status: OrderStatus, 
                      filled_quantity: Optional[float] = None, 
                      fill_price: Optional[float] = None,
                      rejection_reason: Optional[str] = None):
        """
        更新订单状态
        
        参数:
            status: 新状态
            filled_quantity: 成交数量增量
            fill_price: 成交价格
            rejection_reason: 拒绝原因
        """
        self.status = status
        self.updated_time = datetime.now()
        
        if filled_quantity and filled_quantity > 0:
            # 计算新的平均成交价格
            total_value = self.average_fill_price * self.filled_quantity
            new_value = fill_price * filled_quantity
            
            self.filled_quantity += filled_quantity
            
            if self.filled_quantity > 0:
                self.average_fill_price = (total_value + new_value) / self.filled_quantity
            
            # 如果完全成交
            if self.filled_quantity >= self.quantity:
                self.status = OrderStatus.FILLED
                self.filled_time = datetime.now()
        
        if status == OrderStatus.REJECTED:
            self.rejection_reason = rejection_reason
            
        if status == OrderStatus.CANCELED:
            self.cancel_time = datetime.now()
        
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "created_time": self.created_time.isoformat(),
            "updated_time": self.updated_time.isoformat(),
            "filled_time": self.filled_time.isoformat() if self.filled_time else None,
            "cancel_time": self.cancel_time.isoformat() if self.cancel_time else None,
            "rejection_reason": self.rejection_reason
        }

class Position:
    """持仓类"""
    
    def __init__(self, symbol: str, quantity: float = 0.0, 
                 average_cost: float = 0.0, market_price: float = 0.0):
        """
        初始化持仓
        
        参数:
            symbol: 股票代码
            quantity: 持仓数量
            average_cost: 平均成本
            market_price: 市场价格
        """
        self.symbol = symbol
        self.quantity = quantity
        self.average_cost = average_cost
        self.market_price = market_price
        self.market_value = quantity * market_price
        self.unrealized_pnl = quantity * (market_price - average_cost)
        self.unrealized_pnl_percent = (market_price / average_cost - 1) * 100 if average_cost > 0 else 0
        self.realized_pnl = 0.0
        self.last_update_time = datetime.now()
        
    def update_market_price(self, market_price: float):
        """更新市场价格"""
        self.market_price = market_price
        self.market_value = self.quantity * market_price
        self.unrealized_pnl = self.quantity * (market_price - self.average_cost)
        self.unrealized_pnl_percent = (market_price / self.average_cost - 1) * 100 if self.average_cost > 0 else 0
        self.last_update_time = datetime.now()
        
    def apply_fill(self, quantity: float, price: float) -> float:
        """
        应用成交
        
        参数:
            quantity: 成交数量（买入为正，卖出为负）
            price: 成交价格
            
        返回:
            float: 实现盈亏
        """
        realized_pnl = 0.0
        
        if quantity > 0:  # 买入
            # 计算新的平均成本
            total_cost = self.average_cost * self.quantity
            new_cost = price * quantity
            
            self.quantity += quantity
            if self.quantity > 0:
                self.average_cost = (total_cost + new_cost) / self.quantity
            
        else:  # 卖出
            quantity_abs = abs(quantity)
            if quantity_abs > self.quantity:
                # 不允许超卖
                quantity_abs = self.quantity
                
            # 计算实现盈亏
            realized_pnl = quantity_abs * (price - self.average_cost)
            self.realized_pnl += realized_pnl
            
            # 更新持仓
            self.quantity -= quantity_abs
            # 如果完全平仓，重置平均成本
            if self.quantity <= 0:
                self.average_cost = 0.0
        
        # 更新市场价值和未实现盈亏
        self.update_market_price(price)
        
        return realized_pnl
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_cost": self.average_cost,
            "market_price": self.market_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "realized_pnl": self.realized_pnl,
            "last_update_time": self.last_update_time.isoformat()
        }

class Account:
    """账户类"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        """
        初始化账户
        
        参数:
            initial_cash: 初始资金
        """
        self.account_id = str(uuid.uuid4())[:8]
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions = {}  # symbol -> Position
        self.market_value = 0.0
        self.total_value = initial_cash
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0
        self.leverage = 1.0
        self.last_update_time = datetime.now()
        
    def update(self, market_data: Optional[Dict[str, float]] = None):
        """
        更新账户状态
        
        参数:
            market_data: 市场数据，格式为 {symbol: price}
        """
        if market_data:
            for symbol, price in market_data.items():
                if symbol in self.positions:
                    self.positions[symbol].update_market_price(price)
        
        # 更新市场价值和未实现盈亏
        self.market_value = sum(position.market_value for position in self.positions.values())
        self.unrealized_pnl = sum(position.unrealized_pnl for position in self.positions.values())
        self.total_value = self.cash + self.market_value
        self.last_update_time = datetime.now()
        
    def apply_fill(self, order: Order, fill_price: float, commission: float = 0.0) -> Tuple[float, float]:
        """
        应用成交到账户
        
        参数:
            order: 订单对象
            fill_price: 成交价格
            commission: 佣金
            
        返回:
            Tuple[float, float]: (实现盈亏, 成交金额)
        """
        symbol = order.symbol
        
        # 计算成交金额和方向
        fill_quantity = order.filled_quantity
        if order.direction == OrderDirection.SELL:
            fill_quantity = -fill_quantity
            
        fill_amount = abs(order.filled_quantity) * fill_price
        
        # 更新现金
        if order.direction == OrderDirection.BUY:
            self.cash -= (fill_amount + commission)
        else:
            self.cash += (fill_amount - commission)
        
        # 更新持仓
        realized_pnl = 0.0
        if symbol not in self.positions:
            if fill_quantity > 0:  # 只有买入才创建新持仓
                self.positions[symbol] = Position(symbol, fill_quantity, fill_price, fill_price)
        else:
            realized_pnl = self.positions[symbol].apply_fill(fill_quantity, fill_price)
            # 如果持仓为0，删除持仓记录
            if self.positions[symbol].quantity <= 0:
                del self.positions[symbol]
        
        # 更新已实现盈亏
        self.realized_pnl += realized_pnl
        
        # 更新账户状态
        self.update({symbol: fill_price})
        
        return realized_pnl, fill_amount
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取特定持仓"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """获取所有持仓"""
        return list(self.positions.values())
    
    def to_dict(self) -> Dict:
        """转换为字典表示"""
        return {
            "account_id": self.account_id,
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "market_value": self.market_value,
            "total_value": self.total_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_used": self.margin_used,
            "leverage": self.leverage,
            "last_update_time": self.last_update_time.isoformat(),
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()}
        }
    
    def get_performance_metrics(self) -> Dict:
        """获取账户绩效指标"""
        # 计算收益率
        total_return = (self.total_value / self.initial_cash - 1) * 100
        
        return {
            "total_return_pct": total_return,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "cash_ratio": self.cash / self.total_value * 100,
            "position_count": len(self.positions)
        }

class TradingEngine:
    """交易引擎核心类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化交易引擎
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        self.account = Account(self.config.get("initial_cash", 1000000.0))
        self.orders = []
        self.order_history = []
        self.active_orders = []
        self.filled_orders = []
        self.canceled_orders = []
        self.rejected_orders = []
        self.execution_records = []
        self.market_data = {}
        self.last_update_time = datetime.now()
        
        # 设置交易费率
        self.commission_rate = self.config.get("commission_rate", 0.0003)
        self.min_commission = self.config.get("min_commission", 5.0)
        
        # 模拟延迟
        self.execution_delay = self.config.get("execution_delay", 0.0)
        
        # 引入风险管理器和性能分析器
        self.risk_manager = None
        self.performance_analyzer = None
        
        logger.info(f"交易引擎初始化完成，初始资金: {self.account.initial_cash}")
    
    def place_order(self, symbol: str, direction: OrderDirection, quantity: float, 
                   order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
                   stop_price: Optional[float] = None, client_order_id: Optional[str] = None) -> Order:
        """
        下单
        
        参数:
            symbol: 股票代码
            direction: 交易方向
            quantity: 数量
            order_type: 订单类型
            price: 价格（限价单需要）
            stop_price: 止损价格（止损单需要）
            client_order_id: 客户端订单ID
            
        返回:
            Order: 订单对象
        """
        # 创建订单
        order = Order(symbol, direction, quantity, order_type, price, stop_price, client_order_id)
        
        # 检查订单有效性
        if order_type == OrderType.LIMIT and price is None:
            order.update_status(OrderStatus.REJECTED, rejection_reason="限价单需要指定价格")
            self.rejected_orders.append(order)
            return order
        
        # 添加到订单列表
        self.orders.append(order)
        self.active_orders.append(order)
        
        # 异步处理订单
        if self.config.get("async_execution", True):
            # 在实际系统中，这里会调用异步处理
            # 在模拟环境中，我们直接处理
            self._process_order(order)
        
        return order
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        取消订单
        
        参数:
            order_id: 订单ID
            
        返回:
            Tuple[bool, str]: (成功标志, 消息)
        """
        for order in self.active_orders:
            if order.order_id == order_id:
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                    return False, f"订单已经处于{order.status.value}状态，无法取消"
                
                order.update_status(OrderStatus.CANCELED)
                self.active_orders.remove(order)
                self.canceled_orders.append(order)
                return True, "订单取消成功"
        
        return False, "未找到指定的活动订单"
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        参数:
            order_id: 订单ID
            
        返回:
            Optional[Order]: 订单对象，如果不存在则返回None
        """
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None
    
    def get_active_orders(self) -> List[Order]:
        """获取活动订单列表"""
        return self.active_orders
    
    def update_market_data(self, market_data: Dict[str, float]):
        """
        更新市场数据
        
        参数:
            market_data: 市场数据，格式为 {symbol: price}
        """
        self.market_data.update(market_data)
        self.last_update_time = datetime.now()
        
        # 更新账户
        self.account.update(market_data)
        
        # 处理限价单和止损单
        self._process_conditional_orders()
    
    def _process_order(self, order: Order):
        """
        处理订单（模拟执行）
        
        参数:
            order: 订单对象
        """
        # 模拟执行延迟
        if self.execution_delay > 0:
            time.sleep(self.execution_delay)
        
        # 获取当前市场价格
        market_price = self.market_data.get(order.symbol)
        if market_price is None:
            # 尝试从订单价格获取
            if order.price:
                market_price = order.price
            else:
                # 无法执行
                order.update_status(OrderStatus.REJECTED, rejection_reason="无法获取市场价格")
                self.rejected_orders.append(order)
                if order in self.active_orders:
                    self.active_orders.remove(order)
                return
        
        # 根据订单类型处理
        if order.order_type == OrderType.MARKET:
            # 市价单直接执行
            self._execute_order(order, market_price)
        elif order.order_type == OrderType.LIMIT:
            # 限价单检查条件
            if (order.direction == OrderDirection.BUY and market_price <= order.price) or \
               (order.direction == OrderDirection.SELL and market_price >= order.price):
                self._execute_order(order, market_price)
            # 否则保持等待状态
        elif order.order_type == OrderType.STOP:
            # 止损单检查条件
            if (order.direction == OrderDirection.BUY and market_price >= order.stop_price) or \
               (order.direction == OrderDirection.SELL and market_price <= order.stop_price):
                self._execute_order(order, market_price)
            # 否则保持等待状态
        elif order.order_type == OrderType.STOP_LIMIT:
            # 止损限价单检查条件
            if (order.direction == OrderDirection.BUY and market_price >= order.stop_price) or \
               (order.direction == OrderDirection.SELL and market_price <= order.stop_price):
                # 触发后转为限价单
                if (order.direction == OrderDirection.BUY and market_price <= order.price) or \
                   (order.direction == OrderDirection.SELL and market_price >= order.price):
                    self._execute_order(order, market_price)
            # 否则保持等待状态
    
    def _process_conditional_orders(self):
        """处理条件单（限价单和止损单）"""
        for order in list(self.active_orders):
            if order.status != OrderStatus.PENDING:
                continue
                
            market_price = self.market_data.get(order.symbol)
            if market_price is None:
                continue
                
            # 根据订单类型和条件执行
            if order.order_type == OrderType.LIMIT:
                if (order.direction == OrderDirection.BUY and market_price <= order.price) or \
                   (order.direction == OrderDirection.SELL and market_price >= order.price):
                    self._execute_order(order, market_price)
            elif order.order_type == OrderType.STOP:
                if (order.direction == OrderDirection.BUY and market_price >= order.stop_price) or \
                   (order.direction == OrderDirection.SELL and market_price <= order.stop_price):
                    self._execute_order(order, market_price)
            elif order.order_type == OrderType.STOP_LIMIT:
                if (order.direction == OrderDirection.BUY and market_price >= order.stop_price) or \
                   (order.direction == OrderDirection.SELL and market_price <= order.stop_price):
                    # 触发后转为限价单
                    if (order.direction == OrderDirection.BUY and market_price <= order.price) or \
                       (order.direction == OrderDirection.SELL and market_price >= order.price):
                        self._execute_order(order, market_price)
    
    def _execute_order(self, order: Order, execution_price: float):
        """
        执行订单
        
        参数:
            order: 订单对象
            execution_price: 执行价格
        """
        # 检查资金是否足够
        if order.direction == OrderDirection.BUY:
            required_cash = order.quantity * execution_price
            if required_cash > self.account.cash:
                # 资金不足，拒绝订单
                order.update_status(OrderStatus.REJECTED, rejection_reason="资金不足")
                self.rejected_orders.append(order)
                if order in self.active_orders:
                    self.active_orders.remove(order)
                return
        
        # 计算佣金
        commission = max(execution_price * order.quantity * self.commission_rate, self.min_commission)
        
        # 更新订单状态
        order.update_status(OrderStatus.FILLED, order.quantity, execution_price)
        order.commission = commission
        
        # 更新账户
        realized_pnl, fill_amount = self.account.apply_fill(order, execution_price, commission)
        
        # 记录成交记录
        execution_record = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "direction": order.direction.value,
            "quantity": order.quantity,
            "price": execution_price,
            "commission": commission,
            "realized_pnl": realized_pnl,
            "fill_amount": fill_amount,
            "execution_time": datetime.now().isoformat()
        }
        self.execution_records.append(execution_record)
        
        # 更新订单列表
        if order in self.active_orders:
            self.active_orders.remove(order)
        self.filled_orders.append(order)
        
        logger.info(f"订单执行成功: {order.symbol} {order.direction.value} {order.quantity}股 价格:{execution_price}")
    
    def get_account_summary(self) -> Dict:
        """获取账户摘要"""
        return self.account.to_dict()
    
    def get_positions(self) -> List[Dict]:
        """获取持仓列表"""
        return [pos.to_dict() for pos in self.account.get_all_positions()]
    
    def get_order_history(self) -> List[Dict]:
        """获取订单历史"""
        return [order.to_dict() for order in self.orders]
    
    def get_execution_history(self) -> List[Dict]:
        """获取成交历史"""
        return self.execution_records
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if self.performance_analyzer:
            return self.performance_analyzer.get_metrics()
        return self.account.get_performance_metrics()
    
    def reset(self):
        """重置交易引擎"""
        self.account = Account(self.config.get("initial_cash", 1000000.0))
        self.orders = []
        self.order_history = []
        self.active_orders = []
        self.filled_orders = []
        self.canceled_orders = []
        self.rejected_orders = []
        self.execution_records = []
        self.market_data = {}
        self.last_update_time = datetime.now()
        logger.info("交易引擎已重置")

# 测试函数
def test_trading_engine():
    """测试交易引擎"""
    # 创建交易引擎
    engine = TradingEngine({"initial_cash": 100000.0})
    
    # 设置市场数据
    market_data = {
        "000001.SZ": 10.0,
        "600000.SH": 15.0
    }
    engine.update_market_data(market_data)
    
    # 下单
    buy_order = engine.place_order(
        symbol="000001.SZ",
        direction=OrderDirection.BUY,
        quantity=1000,
        order_type=OrderType.MARKET
    )
    
    # 查看账户状态
    account_summary = engine.get_account_summary()
    positions = engine.get_positions()
    
    print(f"账户摘要: {account_summary}")
    print(f"持仓列表: {positions}")
    
    # 卖出部分持仓
    sell_order = engine.place_order(
        symbol="000001.SZ",
        direction=OrderDirection.SELL,
        quantity=500,
        order_type=OrderType.LIMIT,
        price=11.0
    )
    
    # 更新市场数据以触发限价单
    market_data = {
        "000001.SZ": 11.0,
        "600000.SH": 15.5
    }
    engine.update_market_data(market_data)
    
    # 再次查看账户状态
    account_summary = engine.get_account_summary()
    positions = engine.get_positions()
    
    print(f"更新后账户摘要: {account_summary}")
    print(f"更新后持仓列表: {positions}")
    
    # 获取订单历史和成交历史
    order_history = engine.get_order_history()
    execution_history = engine.get_execution_history()
    
    print(f"订单历史: {order_history}")
    print(f"成交历史: {execution_history}")
    
    return engine

if __name__ == "__main__":
    # 设置日志输出
    logging.basicConfig(level=logging.INFO)
    
    # 测试交易引擎
    test_engine = test_trading_engine() 