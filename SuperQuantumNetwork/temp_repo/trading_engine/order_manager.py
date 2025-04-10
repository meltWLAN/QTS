#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Order Manager - 超神量子共生系统订单管理器
负责处理订单的生成、执行、跟踪和管理。
"""

import logging
import uuid
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

# 设置日志
logger = logging.getLogger("OrderManager")

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"         # 市价单
    LIMIT = "limit"           # 限价单
    STOP = "stop"             # 止损单
    STOP_LIMIT = "stop_limit" # 止损限价单
    TRAIL_STOP = "trail_stop" # 追踪止损单


class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = "buy"               # 买入
    SELL = "sell"             # 卖出


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"       # 待处理
    SUBMITTED = "submitted"   # 已提交
    PARTIAL = "partial"       # 部分成交
    FILLED = "filled"         # 完全成交
    CANCELED = "canceled"     # 已取消
    REJECTED = "rejected"     # 被拒绝
    EXPIRED = "expired"       # 已过期


class TimeInForce(Enum):
    """订单有效期枚举"""
    DAY = "day"               # 当日有效
    GTC = "gtc"               # 一直有效直到取消
    IOC = "ioc"               # 立即成交或取消
    FOK = "fok"               # 全部成交或取消


class Order:
    """订单类"""
    
    def __init__(self, 
                 symbol: str,
                 order_type: OrderType,
                 direction: OrderDirection,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 trail_percent: Optional[float] = None,
                 time_in_force: TimeInForce = TimeInForce.DAY,
                 client_order_id: Optional[str] = None,
                 expiration: Optional[datetime] = None):
        """
        初始化订单
        
        Args:
            symbol: 交易标的代码
            order_type: 订单类型
            direction: 买入或卖出
            quantity: 数量
            price: 价格（对于限价单）
            stop_price: 触发价格（对于止损单或止损限价单）
            trail_percent: 追踪百分比（对于追踪止损单）
            time_in_force: 订单有效期
            client_order_id: 客户端订单ID
            expiration: 过期时间
        """
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.trail_percent = trail_percent
        self.time_in_force = time_in_force
        self.client_order_id = client_order_id or f"order_{uuid.uuid4().hex[:8]}"
        self.order_id = None  # 交易所返回的订单ID
        self.status = OrderStatus.PENDING
        self.submitted_time = None
        self.executed_time = None
        self.canceled_time = None
        self.expiration = expiration or (datetime.now() + timedelta(days=1) if time_in_force == TimeInForce.DAY else None)
        self.filled_quantity = 0.0
        self.remaining_quantity = quantity
        self.average_fill_price = 0.0
        self.last_fill_price = 0.0
        self.last_fill_time = None
        self.fill_history = []
        self.commission = 0.0
        self.rejection_reason = None
        self.tags = {}  # 用户自定义标签
        
        # 验证订单参数
        self._validate()
    
    def _validate(self):
        """验证订单参数"""
        # 订单类型验证
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"{self.order_type.value} 订单必须指定价格")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} 订单必须指定止损价格")
            
        if self.order_type == OrderType.TRAIL_STOP and self.trail_percent is None:
            raise ValueError("追踪止损单必须指定追踪百分比")
        
        # 数量验证
        if self.quantity <= 0:
            raise ValueError("订单数量必须大于零")
            
        # 价格验证（如果有）
        if self.price is not None and self.price <= 0:
            raise ValueError("价格必须大于零")
            
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("止损价格必须大于零")
        
    def update_status(self, status: OrderStatus, message: str = ""):
        """
        更新订单状态
        
        Args:
            status: 新状态
            message: 状态更新消息
        """
        old_status = self.status
        self.status = status
        
        timestamp = datetime.now()
        if status == OrderStatus.SUBMITTED:
            self.submitted_time = timestamp
        elif status == OrderStatus.FILLED:
            self.executed_time = timestamp
        elif status == OrderStatus.CANCELED:
            self.canceled_time = timestamp
            
        logger.info(f"订单 {self.client_order_id} 状态更新: {old_status.value} -> {status.value} {message}")
    
    def add_fill(self, fill_quantity: float, fill_price: float, timestamp: Optional[datetime] = None):
        """
        添加成交记录
        
        Args:
            fill_quantity: 成交数量
            fill_price: 成交价格
            timestamp: 成交时间
        
        Returns:
            bool: 是否完全成交
        """
        if fill_quantity <= 0:
            logger.warning(f"尝试添加非正数量的成交记录: {fill_quantity}")
            return False
            
        if fill_quantity > self.remaining_quantity:
            logger.warning(f"成交数量超过剩余数量: 成交 {fill_quantity}, 剩余 {self.remaining_quantity}, 已调整")
            fill_quantity = self.remaining_quantity
            
        fill_time = timestamp or datetime.now()
        
        # 更新成交记录
        self.fill_history.append({
            "timestamp": fill_time,
            "quantity": fill_quantity,
            "price": fill_price
        })
        
        # 更新订单信息
        old_filled = self.filled_quantity
        self.filled_quantity += fill_quantity
        self.remaining_quantity -= fill_quantity
        self.last_fill_price = fill_price
        self.last_fill_time = fill_time
        
        # 计算新的平均成交价格
        self.average_fill_price = (old_filled * self.average_fill_price + fill_quantity * fill_price) / self.filled_quantity
        
        # 更新订单状态
        if self.remaining_quantity <= 0:
            self.update_status(OrderStatus.FILLED, f"完全成交，平均价格: {self.average_fill_price:.4f}")
            return True
        else:
            self.update_status(OrderStatus.PARTIAL, f"部分成交: {self.filled_quantity}/{self.quantity}, 价格: {fill_price:.4f}")
            return False
    
    def cancel(self, reason: str = ""):
        """
        取消订单
        
        Args:
            reason: 取消原因
            
        Returns:
            bool: 是否成功取消
        """
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            logger.warning(f"无法取消状态为 {self.status.value} 的订单")
            return False
            
        self.update_status(OrderStatus.CANCELED, f"取消原因: {reason}")
        return True
    
    def reject(self, reason: str):
        """
        拒绝订单
        
        Args:
            reason: 拒绝原因
        """
        self.rejection_reason = reason
        self.update_status(OrderStatus.REJECTED, f"拒绝原因: {reason}")
    
    def is_active(self) -> bool:
        """
        订单是否活跃
        
        Returns:
            bool: 是否活跃
        """
        active_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        return self.status in active_statuses
    
    def is_complete(self) -> bool:
        """
        订单是否已完成处理（无论成功或失败）
        
        Returns:
            bool: 是否完成
        """
        return not self.is_active()
    
    def is_expired(self) -> bool:
        """
        订单是否已过期
        
        Returns:
            bool: 是否过期
        """
        if self.status == OrderStatus.EXPIRED:
            return True
            
        if self.expiration and datetime.now() > self.expiration:
            return True
            
        return False
    
    def get_value(self) -> float:
        """
        获取订单价值
        
        Returns:
            float: 订单价值
        """
        if self.average_fill_price > 0:
            return self.filled_quantity * self.average_fill_price
            
        if self.price:
            return self.quantity * self.price
            
        # 对于市价单，无法准确估计价值
        return 0.0
    
    def get_remaining_value(self) -> float:
        """
        获取剩余订单价值
        
        Returns:
            float: 剩余订单价值
        """
        if self.price:
            return self.remaining_quantity * self.price
            
        # 对于市价单，无法准确估计价值
        return 0.0
    
    def add_tag(self, key: str, value: Any):
        """
        添加标签
        
        Args:
            key: 标签键
            value: 标签值
        """
        self.tags[key] = value
    
    def to_dict(self) -> Dict:
        """
        转换为字典
        
        Returns:
            Dict: 订单字典
        """
        return {
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "order_type": self.order_type.value,
            "direction": self.direction.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "trail_percent": self.trail_percent,
            "time_in_force": self.time_in_force.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "submitted_time": self.submitted_time.strftime("%Y-%m-%d %H:%M:%S.%f") if self.submitted_time else None,
            "executed_time": self.executed_time.strftime("%Y-%m-%d %H:%M:%S.%f") if self.executed_time else None,
            "canceled_time": self.canceled_time.strftime("%Y-%m-%d %H:%M:%S.%f") if self.canceled_time else None,
            "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S.%f") if self.expiration else None,
            "rejection_reason": self.rejection_reason,
            "value": self.get_value(),
            "remaining_value": self.get_remaining_value(),
            "tags": self.tags
        }


class OrderManager:
    """订单管理器类"""
    
    def __init__(self, commission_rate: float = 0.0003, min_commission: float = 5.0):
        """
        初始化订单管理器
        
        Args:
            commission_rate: 佣金率
            min_commission: 最低佣金
        """
        self.orders: Dict[str, Order] = {}  # 所有订单
        self.active_orders: Dict[str, Order] = {}  # 活跃订单
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.order_history = []  # 订单历史
        
        logger.info(f"初始化订单管理器，佣金率: {commission_rate}, 最低佣金: {min_commission}")
    
    def create_order(self, 
                    symbol: str,
                    order_type: OrderType,
                    direction: OrderDirection,
                    quantity: float,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    time_in_force: TimeInForce = TimeInForce.DAY,
                    client_order_id: Optional[str] = None,
                    tags: Optional[Dict[str, Any]] = None) -> Order:
        """
        创建新订单
        
        Args:
            symbol: 交易标的代码
            order_type: 订单类型
            direction: 买入或卖出
            quantity: 数量
            price: 价格（对于限价单）
            stop_price: 触发价格（对于止损单或止损限价单）
            trail_percent: 追踪百分比（对于追踪止损单）
            time_in_force: 订单有效期
            client_order_id: 客户端订单ID
            tags: 订单标签
            
        Returns:
            Order: 创建的订单
        """
        try:
            order = Order(
                symbol=symbol,
                order_type=order_type,
                direction=direction,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                trail_percent=trail_percent,
                time_in_force=time_in_force,
                client_order_id=client_order_id
            )
            
            # 添加标签
            if tags:
                for key, value in tags.items():
                    order.add_tag(key, value)
            
            # 存储订单
            self.orders[order.client_order_id] = order
            self.active_orders[order.client_order_id] = order
            
            logger.info(f"创建订单: {order.client_order_id}, {symbol} {direction.value} {quantity} @ {price if price else 'market'}")
            return order
            
        except ValueError as e:
            logger.error(f"创建订单失败: {str(e)}")
            raise
    
    def submit_order(self, order: Union[Order, str]) -> Tuple[bool, str]:
        """
        提交订单
        
        Args:
            order: 订单对象或订单ID
            
        Returns:
            Tuple[bool, str]: 成功标志和消息
        """
        # 获取订单对象
        if isinstance(order, str):
            if order not in self.orders:
                return False, f"订单不存在: {order}"
            order = self.orders[order]
        
        # 检查订单状态
        if order.status != OrderStatus.PENDING:
            return False, f"订单状态不是待处理: {order.status.value}"
        
        # 更新订单状态
        order.update_status(OrderStatus.SUBMITTED)
        
        # 记录订单提交
        self._record_order_action(order, "SUBMIT")
        
        return True, f"订单 {order.client_order_id} 已提交"
    
    def cancel_order(self, order: Union[Order, str], reason: str = "") -> Tuple[bool, str]:
        """
        取消订单
        
        Args:
            order: 订单对象或订单ID
            reason: 取消原因
            
        Returns:
            Tuple[bool, str]: 成功标志和消息
        """
        # 获取订单对象
        if isinstance(order, str):
            if order not in self.orders:
                return False, f"订单不存在: {order}"
            order = self.orders[order]
        
        # 尝试取消订单
        if not order.cancel(reason):
            return False, f"无法取消订单 {order.client_order_id}: {order.status.value}"
        
        # 从活跃订单中移除
        if order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        # 记录订单取消
        self._record_order_action(order, "CANCEL", reason)
        
        return True, f"订单 {order.client_order_id} 已取消"
    
    def reject_order(self, order: Union[Order, str], reason: str) -> Tuple[bool, str]:
        """
        拒绝订单
        
        Args:
            order: 订单对象或订单ID
            reason: 拒绝原因
            
        Returns:
            Tuple[bool, str]: 成功标志和消息
        """
        # 获取订单对象
        if isinstance(order, str):
            if order not in self.orders:
                return False, f"订单不存在: {order}"
            order = self.orders[order]
        
        # 检查订单状态
        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            return False, f"订单状态不适合拒绝: {order.status.value}"
        
        # 拒绝订单
        order.reject(reason)
        
        # 从活跃订单中移除
        if order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        # 记录订单拒绝
        self._record_order_action(order, "REJECT", reason)
        
        return True, f"订单 {order.client_order_id} 已拒绝"
    
    def fill_order(self, 
                  order: Union[Order, str], 
                  fill_quantity: float, 
                  fill_price: float, 
                  timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        成交订单
        
        Args:
            order: 订单对象或订单ID
            fill_quantity: 成交数量
            fill_price: 成交价格
            timestamp: 成交时间
            
        Returns:
            Tuple[bool, str]: 成功标志和消息
        """
        # 获取订单对象
        if isinstance(order, str):
            if order not in self.orders:
                return False, f"订单不存在: {order}"
            order = self.orders[order]
        
        # 检查订单状态
        if not order.is_active():
            return False, f"订单不活跃: {order.status.value}"
        
        # 添加成交记录
        is_filled = order.add_fill(fill_quantity, fill_price, timestamp)
        
        # 计算佣金
        fill_value = fill_quantity * fill_price
        commission = max(fill_value * self.commission_rate, self.min_commission)
        order.commission += commission
        
        # 如果订单完全成交，从活跃订单中移除
        if is_filled and order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        # 记录订单成交
        self._record_order_action(
            order, 
            "FILL", 
            f"数量: {fill_quantity}, 价格: {fill_price}, 佣金: {commission:.2f}"
        )
        
        return True, f"订单 {order.client_order_id} 成交: {fill_quantity} @ {fill_price}"
    
    def expire_orders(self) -> int:
        """
        过期超时订单
        
        Returns:
            int: 过期的订单数量
        """
        expired_count = 0
        now = datetime.now()
        
        # 检查所有活跃订单
        for order_id in list(self.active_orders.keys()):
            order = self.active_orders[order_id]
            
            if order.expiration and now > order.expiration:
                # 更新状态为过期
                order.update_status(OrderStatus.EXPIRED)
                
                # 从活跃订单中移除
                del self.active_orders[order_id]
                
                # 记录订单过期
                self._record_order_action(order, "EXPIRE")
                
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"过期了 {expired_count} 个订单")
            
        return expired_count
    
    def _record_order_action(self, order: Order, action: str, message: str = ""):
        """
        记录订单行为
        
        Args:
            order: 订单对象
            action: 行为
            message: 消息
        """
        self.order_history.append({
            "timestamp": datetime.now(),
            "action": action,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "direction": order.direction.value,
            "status": order.status.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "message": message
        })
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        获取订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Optional[Order]: 订单对象
        """
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """
        获取所有活跃订单
        
        Returns:
            List[Order]: 活跃订单列表
        """
        return list(self.active_orders.values())
    
    def get_active_orders_for_symbol(self, symbol: str) -> List[Order]:
        """
        获取指定标的的活跃订单
        
        Args:
            symbol: 交易标的代码
            
        Returns:
            List[Order]: 活跃订单列表
        """
        return [order for order in self.active_orders.values() if order.symbol == symbol]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        获取指定状态的订单
        
        Args:
            status: 订单状态
            
        Returns:
            List[Order]: 订单列表
        """
        return [order for order in self.orders.values() if order.status == status]
    
    def get_filled_orders(self) -> List[Order]:
        """
        获取所有已成交订单
        
        Returns:
            List[Order]: 已成交订单列表
        """
        return self.get_orders_by_status(OrderStatus.FILLED)
    
    def get_order_summary(self) -> Dict:
        """
        获取订单摘要
        
        Returns:
            Dict: 订单摘要
        """
        # 按状态统计订单
        status_counts = {status.value: 0 for status in OrderStatus}
        for order in self.orders.values():
            status_counts[order.status.value] += 1
        
        # 按类型统计订单
        type_counts = {order_type.value: 0 for order_type in OrderType}
        for order in self.orders.values():
            type_counts[order.order_type.value] += 1
        
        # 计算总数值
        total_commission = sum(order.commission for order in self.orders.values())
        
        # 计算购买力使用情况（估计值）
        buying_power_used = sum(order.get_remaining_value() for order in self.active_orders.values() 
                               if order.direction == OrderDirection.BUY)
        
        return {
            "total_orders": len(self.orders),
            "active_orders": len(self.active_orders),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "total_commission": total_commission,
            "buying_power_used": buying_power_used
        }


def test_order_manager():
    """测试订单管理器"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 初始化订单管理器
    om = OrderManager(commission_rate=0.0003, min_commission=5.0)
    
    # 创建订单
    order1 = om.create_order(
        symbol="000001.SZ",
        order_type=OrderType.LIMIT,
        direction=OrderDirection.BUY,
        quantity=1000,
        price=15.50,
        time_in_force=TimeInForce.DAY,
        tags={"strategy": "momentum", "signal_strength": 0.8}
    )
    
    order2 = om.create_order(
        symbol="600000.SH",
        order_type=OrderType.MARKET,
        direction=OrderDirection.SELL,
        quantity=500,
        time_in_force=TimeInForce.IOC
    )
    
    order3 = om.create_order(
        symbol="000001.SZ",
        order_type=OrderType.STOP_LIMIT,
        direction=OrderDirection.SELL,
        quantity=800,
        price=14.80,
        stop_price=15.00,
        time_in_force=TimeInForce.GTC
    )
    
    # 提交订单
    print("\n提交订单:")
    for order in [order1, order2, order3]:
        success, msg = om.submit_order(order)
        print(f"{msg}")
    
    # 成交订单
    print("\n处理订单成交:")
    om.fill_order(order1, 500, 15.45)  # 部分成交
    om.fill_order(order2, 500, 10.20)  # 完全成交
    
    # 取消订单
    print("\n取消订单:")
    success, msg = om.cancel_order(order3, "策略调整")
    print(f"{msg}")
    
    # 再次成交剩余的订单1
    om.fill_order(order1, 500, 15.55)  # 完全成交
    
    # 获取订单统计
    summary = om.get_order_summary()
    
    print("\n订单统计:")
    print(f"总订单数: {summary['total_orders']}")
    print(f"活跃订单数: {summary['active_orders']}")
    print(f"总佣金: {summary['total_commission']:.2f}")
    
    print("\n按状态统计:")
    for status, count in summary['status_counts'].items():
        if count > 0:
            print(f"  {status}: {count}")
    
    print("\n按类型统计:")
    for type_name, count in summary['type_counts'].items():
        if count > 0:
            print(f"  {type_name}: {count}")
    
    # 获取成交订单
    print("\n已成交订单详情:")
    for order in om.get_filled_orders():
        order_dict = order.to_dict()
        print(f"订单ID: {order_dict['client_order_id']}")
        print(f"标的: {order_dict['symbol']}")
        print(f"方向: {order_dict['direction']}")
        print(f"数量: {order_dict['quantity']}")
        print(f"成交数量: {order_dict['filled_quantity']}")
        print(f"平均成交价格: {order_dict['average_fill_price']:.4f}")
        print(f"佣金: {order_dict['commission']:.2f}")
        print(f"订单价值: {order_dict['value']:.2f}")
        print(f"标签: {order_dict['tags']}")
        print("-----")


if __name__ == "__main__":
    test_order_manager() 