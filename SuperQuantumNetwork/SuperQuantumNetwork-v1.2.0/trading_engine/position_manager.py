#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Position Manager - 超神量子共生系统仓位管理器
负责管理所有持仓，包括仓位计算、风险控制和资金分配。
"""

import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# 设置日志
logger = logging.getLogger("PositionManager")

class PositionType(Enum):
    """持仓类型枚举"""
    LONG = "long"       # 多头仓位
    SHORT = "short"     # 空头仓位
    CASH = "cash"       # 现金


class Position:
    """单个持仓类"""
    
    def __init__(self, 
                 symbol: str, 
                 position_type: PositionType,
                 quantity: float = 0,
                 avg_price: float = 0.0,
                 current_price: float = 0.0,
                 timestamp: Optional[datetime] = None):
        """
        初始化持仓
        
        Args:
            symbol: 股票代码
            position_type: 持仓类型
            quantity: 持仓数量
            avg_price: 平均持仓价格
            current_price: 当前价格
            timestamp: 更新时间戳
        """
        self.symbol = symbol
        self.position_type = position_type
        self.quantity = quantity
        self.avg_price = avg_price
        self.current_price = current_price
        self.timestamp = timestamp or datetime.now()
        self.history = []  # 持仓变化历史
        self.realized_pnl = 0.0  # 已实现盈亏
        self.cost_basis = avg_price * quantity  # 持仓成本
        
        # 记录初始持仓
        self._record_history("INIT", quantity, avg_price)
    
    def _record_history(self, action: str, quantity: float, price: float):
        """记录持仓变动历史"""
        self.history.append({
            "timestamp": datetime.now(),
            "action": action,
            "quantity": quantity,
            "price": price,
            "resulting_position": self.quantity,
            "avg_price": self.avg_price
        })
    
    def update_price(self, current_price: float):
        """更新当前价格"""
        self.current_price = current_price
        self.timestamp = datetime.now()
    
    def add(self, quantity: float, price: float):
        """
        增加持仓
        
        Args:
            quantity: 增加的数量
            price: 交易价格
        """
        if quantity <= 0:
            logger.warning(f"试图用负数或零增加持仓: {quantity}")
            return
        
        # 计算新的平均价格
        new_cost = self.cost_basis + (quantity * price)
        new_quantity = self.quantity + quantity
        
        # 更新平均价格和总数量
        self.avg_price = new_cost / new_quantity if new_quantity > 0 else 0
        self.quantity = new_quantity
        self.cost_basis = new_cost
        
        # 记录历史
        self._record_history("ADD", quantity, price)
        logger.info(f"增加持仓 {self.symbol}: +{quantity} @ {price}, 新持仓: {self.quantity}, 平均价: {self.avg_price:.2f}")
    
    def reduce(self, quantity: float, price: float):
        """
        减少持仓
        
        Args:
            quantity: 减少的数量
            price: 交易价格
        
        Returns:
            float: 实现的盈亏
        """
        if quantity <= 0:
            logger.warning(f"试图用负数或零减少持仓: {quantity}")
            return 0.0
        
        if quantity > self.quantity:
            logger.warning(f"试图减少超过当前持仓的数量: 当前 {self.quantity}, 减少 {quantity}")
            quantity = self.quantity
        
        # 计算实现盈亏
        pnl = 0.0
        if self.position_type == PositionType.LONG:
            pnl = (price - self.avg_price) * quantity
        elif self.position_type == PositionType.SHORT:
            pnl = (self.avg_price - price) * quantity
        
        # 更新持仓
        self.quantity -= quantity
        
        # 如果持仓为零，重置平均价格
        if self.quantity <= 0:
            self.avg_price = 0.0
            self.cost_basis = 0.0
            self.quantity = 0
        else:
            # 成本基础减少按比例
            reduction_ratio = quantity / (self.quantity + quantity)
            self.cost_basis = self.cost_basis * (1 - reduction_ratio)
        
        # 更新已实现盈亏
        self.realized_pnl += pnl
        
        # 记录历史
        self._record_history("REDUCE", quantity, price)
        logger.info(f"减少持仓 {self.symbol}: -{quantity} @ {price}, 新持仓: {self.quantity}, 实现盈亏: {pnl:.2f}")
        
        return pnl
    
    def get_market_value(self) -> float:
        """获取当前市场价值"""
        if self.position_type == PositionType.CASH:
            return self.quantity
        
        return self.quantity * self.current_price
    
    def get_unrealized_pnl(self) -> float:
        """获取未实现盈亏"""
        if self.quantity == 0 or self.position_type == PositionType.CASH:
            return 0.0
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.avg_price) * self.quantity
        else:  # SHORT
            return (self.avg_price - self.current_price) * self.quantity
    
    def get_pnl_percentage(self) -> float:
        """获取盈亏百分比"""
        if self.cost_basis == 0:
            return 0.0
        
        unrealized = self.get_unrealized_pnl()
        return (unrealized / self.cost_basis) * 100.0
    
    def to_dict(self) -> Dict:
        """将持仓转换为字典"""
        return {
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "market_value": self.get_market_value(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "realized_pnl": self.realized_pnl,
            "pnl_percentage": self.get_pnl_percentage(),
            "cost_basis": self.cost_basis,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }


class PositionManager:
    """仓位管理器类"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        """
        初始化仓位管理器
        
        Args:
            initial_cash: 初始现金
        """
        self.positions: Dict[str, Position] = {}
        self.initial_cash = initial_cash
        
        # 初始化现金仓位
        self.cash_position = Position("CASH", PositionType.CASH, 
                                      quantity=initial_cash, 
                                      avg_price=1.0, 
                                      current_price=1.0)
        
        self.positions["CASH"] = self.cash_position
        self.history = []
        self.max_position_value = 0.0  # 记录最大仓位价值
        self.max_account_value = initial_cash  # 记录最大账户价值
        
        logger.info(f"初始化仓位管理器，初始资金: {initial_cash:.2f}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定证券的持仓"""
        return self.positions.get(symbol)
    
    def add_position(self, symbol: str, position_type: PositionType, 
                    quantity: float, price: float) -> Tuple[bool, str]:
        """
        添加或增加持仓
        
        Args:
            symbol: 证券代码
            position_type: 持仓类型
            quantity: 数量
            price: 价格
        
        Returns:
            Tuple[bool, str]: (成功标志, 消息)
        """
        # 检查是否有足够的现金
        cost = quantity * price
        if position_type == PositionType.LONG and cost > self.cash_position.quantity:
            return False, f"现金不足: 需要 {cost:.2f}, 可用 {self.cash_position.quantity:.2f}"
        
        # 如果已存在持仓，增加数量
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # 验证持仓类型一致
            if position.position_type != position_type:
                return False, f"持仓类型不匹配: 当前 {position.position_type.value}, 请求 {position_type.value}"
            
            position.add(quantity, price)
        else:
            # 创建新持仓
            self.positions[symbol] = Position(
                symbol=symbol,
                position_type=position_type,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        
        # 更新现金
        if position_type == PositionType.LONG:
            self.cash_position.reduce(cost, 1.0)
        elif position_type == PositionType.SHORT:
            # 空头增加资金（暂存到资金账户）
            self.cash_position.add(cost, 1.0)
        
        self._record_history(f"ADD_{position_type.value}", symbol, quantity, price)
        return True, f"成功添加持仓 {symbol}: {quantity} @ {price}"
    
    def reduce_position(self, symbol: str, quantity: float, 
                       price: float) -> Tuple[bool, str, float]:
        """
        减少持仓
        
        Args:
            symbol: 证券代码
            quantity: 减少数量
            price: 价格
        
        Returns:
            Tuple[bool, str, float]: (成功标志, 消息, 实现盈亏)
        """
        if symbol not in self.positions or symbol == "CASH":
            return False, f"持仓不存在: {symbol}", 0.0
        
        position = self.positions[symbol]
        
        if quantity > position.quantity:
            quantity = position.quantity
            logger.warning(f"减少数量超过持仓，已调整为: {quantity}")
        
        # 实现盈亏
        pnl = position.reduce(quantity, price)
        
        # 更新现金
        transaction_value = quantity * price
        if position.position_type == PositionType.LONG:
            self.cash_position.add(transaction_value, 1.0)
        elif position.position_type == PositionType.SHORT:
            self.cash_position.reduce(transaction_value, 1.0)
        
        # 如果持仓为零，删除持仓
        if position.quantity <= 0:
            del self.positions[symbol]
        
        self._record_history(f"REDUCE_{position.position_type.value}", symbol, quantity, price)
        return True, f"成功减少持仓 {symbol}: {quantity} @ {price}, 实现盈亏: {pnl:.2f}", pnl
    
    def update_position_price(self, symbol: str, price: float) -> bool:
        """更新持仓价格"""
        if symbol not in self.positions:
            return False
        
        self.positions[symbol].update_price(price)
        return True
    
    def update_prices(self, price_dict: Dict[str, float]):
        """批量更新价格"""
        for symbol, price in price_dict.items():
            self.update_position_price(symbol, price)
    
    def _record_history(self, action: str, symbol: str, quantity: float, price: float):
        """记录仓位变化历史"""
        self.history.append({
            "timestamp": datetime.now(),
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "cash_balance": self.cash_position.quantity,
            "total_value": self.get_total_value()
        })
    
    def get_position_value(self) -> float:
        """获取所有持仓的市场价值（不包括现金）"""
        value = 0.0
        for symbol, position in self.positions.items():
            if symbol != "CASH":
                value += position.get_market_value()
        
        # 更新最大仓位价值
        if value > self.max_position_value:
            self.max_position_value = value
            
        return value
    
    def get_total_value(self) -> float:
        """获取账户总价值（包括现金）"""
        total = self.cash_position.quantity + self.get_position_value()
        
        # 更新最大账户价值
        if total > self.max_account_value:
            self.max_account_value = total
            
        return total
    
    def get_unrealized_pnl(self) -> float:
        """获取总未实现盈亏"""
        pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol != "CASH":
                pnl += position.get_unrealized_pnl()
        return pnl
    
    def get_realized_pnl(self) -> float:
        """获取总已实现盈亏"""
        pnl = 0.0
        for symbol, position in self.positions.items():
            pnl += position.realized_pnl
        return pnl
    
    def get_net_exposure(self) -> float:
        """获取净暴露度（多头市值 - 空头市值）/ 总资产"""
        long_value = 0.0
        short_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol == "CASH":
                continue
                
            market_value = position.get_market_value()
            if position.position_type == PositionType.LONG:
                long_value += market_value
            elif position.position_type == PositionType.SHORT:
                short_value += market_value
        
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
            
        return (long_value - short_value) / total_value
    
    def get_leverage(self) -> float:
        """获取杠杆率（总仓位价值/总资产）"""
        position_value = self.get_position_value()
        total_value = self.get_total_value()
        
        if total_value == 0:
            return 0.0
            
        return position_value / total_value
    
    def get_drawdown(self) -> float:
        """获取当前回撤百分比"""
        current_value = self.get_total_value()
        
        if self.max_account_value == 0:
            return 0.0
            
        return ((self.max_account_value - current_value) / self.max_account_value) * 100.0
    
    def get_cash_percentage(self) -> float:
        """获取现金占总资产的百分比"""
        total_value = self.get_total_value()
        
        if total_value == 0:
            return 100.0
            
        return (self.cash_position.quantity / total_value) * 100.0
    
    def get_position_summary(self) -> Dict:
        """获取仓位摘要"""
        positions = []
        for symbol, position in self.positions.items():
            if symbol != "CASH":
                positions.append(position.to_dict())
        
        return {
            "positions": positions,
            "cash": self.cash_position.quantity,
            "total_position_value": self.get_position_value(),
            "total_value": self.get_total_value(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "realized_pnl": self.get_realized_pnl(),
            "net_exposure": self.get_net_exposure(),
            "leverage": self.get_leverage(),
            "drawdown": self.get_drawdown(),
            "cash_percentage": self.get_cash_percentage(),
            "position_count": len(self.positions) - 1  # 不包括现金
        }
    
    def calculate_optimal_position_size(self, 
                                      symbol: str, 
                                      risk_per_trade: float = 0.02,
                                      stop_loss_percent: float = 0.05) -> Dict:
        """
        计算最优仓位大小
        
        Args:
            symbol: 证券代码
            risk_per_trade: 每笔交易风险占总资产比例
            stop_loss_percent: 止损百分比
        
        Returns:
            Dict: 最优仓位信息
        """
        # 获取当前价格
        current_price = 0.0
        if symbol in self.positions:
            current_price = self.positions[symbol].current_price
        
        if current_price <= 0:
            return {
                "error": "无法获取有效价格",
                "position_size": 0,
                "value": 0.0
            }
        
        total_value = self.get_total_value()
        risk_amount = total_value * risk_per_trade
        
        # 基于止损计算
        price_risk = current_price * stop_loss_percent
        if price_risk <= 0:
            return {
                "error": "止损价格无效",
                "position_size": 0,
                "value": 0.0
            }
        
        # 计算股数，并确保整数
        shares = int(risk_amount / price_risk)
        position_value = shares * current_price
        
        return {
            "symbol": symbol,
            "optimal_shares": shares,
            "current_price": current_price,
            "position_value": position_value,
            "risk_amount": risk_amount,
            "stop_loss_percent": stop_loss_percent,
            "stop_loss_price": current_price * (1 - stop_loss_percent),
            "percentage_of_account": (position_value / total_value) * 100 if total_value > 0 else 0
        }
    
    def get_allocation_by_sector(self) -> Dict[str, float]:
        """
        按行业获取资产配置
        
        注意：需要外部提供symbol到sector的映射
        """
        # 这里仅返回示例，实际需要与外部数据库集成
        return {
            "Technology": 25.5,
            "Finance": 15.3,
            "Consumer": 22.1,
            "Healthcare": 18.7,
            "Other": 18.4
        }
    
    def export_positions_to_dataframe(self) -> pd.DataFrame:
        """导出持仓到DataFrame"""
        positions = []
        for symbol, position in self.positions.items():
            positions.append(position.to_dict())
        
        return pd.DataFrame(positions)
    
    def export_history_to_dataframe(self) -> pd.DataFrame:
        """导出历史记录到DataFrame"""
        return pd.DataFrame(self.history)


def test_position_manager():
    """测试仓位管理器"""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 初始化仓位管理器
    pm = PositionManager(initial_cash=1000000.0)
    print(f"初始账户价值: {pm.get_total_value():.2f}")
    
    # 添加多头持仓
    pm.add_position("000001.SZ", PositionType.LONG, 10000, 15.5)
    pm.add_position("600000.SH", PositionType.LONG, 5000, 8.2)
    
    # 添加空头持仓
    pm.add_position("000002.SZ", PositionType.SHORT, 3000, 22.3)
    
    # 更新价格
    pm.update_prices({
        "000001.SZ": 16.2,
        "600000.SH": 7.9,
        "000002.SZ": 21.5
    })
    
    # 获取账户摘要
    summary = pm.get_position_summary()
    print("\n账户摘要:")
    print(f"总资产: {summary['total_value']:.2f}")
    print(f"现金: {summary['cash']:.2f} ({summary['cash_percentage']:.2f}%)")
    print(f"持仓价值: {summary['total_position_value']:.2f}")
    print(f"未实现盈亏: {summary['unrealized_pnl']:.2f}")
    print(f"已实现盈亏: {summary['realized_pnl']:.2f}")
    print(f"净暴露度: {summary['net_exposure']:.4f}")
    print(f"杠杆率: {summary['leverage']:.4f}")
    print(f"当前回撤: {summary['drawdown']:.2f}%")
    
    # 输出持仓明细
    print("\n持仓明细:")
    for position in summary['positions']:
        print(f"{position['symbol']} ({position['position_type']}): "
              f"{position['quantity']} 股 @ {position['avg_price']:.2f}, "
              f"当前价: {position['current_price']:.2f}, "
              f"市值: {position['market_value']:.2f}, "
              f"未实现盈亏: {position['unrealized_pnl']:.2f} ({position['pnl_percentage']:.2f}%)")
    
    # 减少持仓
    print("\n减少持仓:")
    success, msg, pnl = pm.reduce_position("000001.SZ", 5000, 16.5)
    print(msg)
    
    # 重新获取账户摘要
    summary = pm.get_position_summary()
    print("\n更新后账户摘要:")
    print(f"总资产: {summary['total_value']:.2f}")
    print(f"现金: {summary['cash']:.2f}")
    print(f"持仓价值: {summary['total_position_value']:.2f}")
    print(f"已实现盈亏: {summary['realized_pnl']:.2f}")
    
    # 计算最优仓位
    print("\n最优仓位计算:")
    optimal = pm.calculate_optimal_position_size("600036.SH", risk_per_trade=0.02, stop_loss_percent=0.05)
    print(f"证券: {optimal.get('symbol', 'N/A')}")
    print(f"最优股数: {optimal.get('optimal_shares', 0)}")
    print(f"仓位市值: {optimal.get('position_value', 0):.2f}")
    print(f"账户占比: {optimal.get('percentage_of_account', 0):.2f}%")
    print(f"止损价格: {optimal.get('stop_loss_price', 0):.2f}")


if __name__ == "__main__":
    test_position_manager() 