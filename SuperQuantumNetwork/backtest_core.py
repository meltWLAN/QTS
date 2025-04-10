#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量化回测引擎核心 - 处理回测主循环和结果输出
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import time
import uuid
import os
import json
from collections import deque
from typing import Dict, List, Optional, Union, Callable, Any, Tuple

# 导入基础组件
from backtest_engine import (
    Event, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent,
    EventType, OrderType, OrderDirection, OrderStatus,
    DataHandler, Strategy, Portfolio, ExecutionHandler, RiskManager, PerformanceAnalyzer
)

# 日志配置
logger = logging.getLogger('BacktestCore')

class Backtest:
    """回测核心类，负责协调各个组件的工作"""
    
    def __init__(
        self,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 1000000.0,
        heartbeat: float = 0.0,
        benchmark_symbol: Optional[str] = None
    ):
        """
        初始化回测引擎
        
        参数:
        data_handler (DataHandler): 数据处理器
        strategy (Strategy): 策略
        portfolio (Portfolio): 投资组合
        execution_handler (ExecutionHandler): 执行处理器
        risk_manager (RiskManager, optional): 风险管理器
        initial_capital (float): 初始资金
        heartbeat (float): 每次循环之间的等待时间（秒）
        benchmark_symbol (str, optional): 基准标的代码
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.benchmark_symbol = benchmark_symbol
        
        # 事件队列
        self.events = deque()
        
        # 回测结果
        self.all_holdings = []  # 每日持仓价值历史
        self.all_positions = []  # 每日持仓数量历史
        self.all_trades = []  # 所有交易记录
        
        # 绩效分析器
        benchmark_data = None
        if benchmark_symbol:
            try:
                benchmark_data = data_handler.get_latest_data(benchmark_symbol)
            except:
                logger.warning(f"无法获取基准 {benchmark_symbol} 的数据")
        
        self.analyzer = PerformanceAnalyzer(benchmark_data)
        
        # 运行状态
        self.continue_backtest = True
        
        # 回测ID和开始时间
        self.backtest_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.datetime.now()
        
        logger.info(f"回测引擎初始化完成，ID: {self.backtest_id}")
    
    def _process_event(self, event: Event) -> None:
        """
        处理事件
        
        参数:
        event (Event): 要处理的事件
        """
        if event.type == EventType.MARKET_DATA:
            # 处理市场数据事件
            market_data_event = event  # type: MarketDataEvent
            
            # 策略根据市场数据生成信号
            signal_events = self.strategy.calculate_signals(market_data_event)
            
            # 将信号事件加入队列
            for signal_event in signal_events:
                self.events.append(signal_event)
                
            # 更新投资组合价值
            portfolio_event = self.portfolio.update_portfolio_value()
            self.analyzer.update(portfolio_event)
            
        elif event.type == EventType.SIGNAL:
            # 处理信号事件
            signal_event = event  # type: SignalEvent
            
            # 投资组合根据信号生成订单
            order_events = self.portfolio.update_on_signal(signal_event)
            
            # 风险管理
            if self.risk_manager:
                order_events = self.risk_manager.process_orders(order_events)
            
            # 将订单事件加入队列
            for order_event in order_events:
                self.events.append(order_event)
                
        elif event.type == EventType.ORDER:
            # 处理订单事件
            order_event = event  # type: OrderEvent
            
            # 执行订单
            fill_events = self.execution_handler.execute_order(order_event)
            
            # 将成交事件加入队列
            for fill_event in fill_events:
                self.events.append(fill_event)
                
                # 记录交易
                self._record_trade(fill_event)
                
        elif event.type == EventType.FILL:
            # 处理成交事件
            fill_event = event  # type: FillEvent
            
            # 更新投资组合
            self.portfolio.update_on_fill(fill_event)
            
        elif event.type == EventType.PORTFOLIO:
            # 处理投资组合更新事件
            portfolio_event = event  # type: PortfolioEvent
            
            # 更新绩效分析器
            self.analyzer.update(portfolio_event)
    
    def _record_trade(self, fill_event: FillEvent) -> None:
        """
        记录交易
        
        参数:
        fill_event (FillEvent): 成交事件
        """
        trade = {
            'timestamp': fill_event.timestamp,
            'symbol': fill_event.symbol,
            'direction': fill_event.direction.value,
            'quantity': fill_event.quantity,
            'price': fill_event.price,
            'commission': fill_event.commission,
            'order_id': fill_event.order_id
        }
        
        self.all_trades.append(trade)
    
    def run(self) -> Dict:
        """
        运行回测
        
        返回:
        Dict: 回测结果
        """
        logger.info("开始回测...")
        
        # 记录回测开始时间
        backtest_start = time.time()
        
        # 主循环
        while self.continue_backtest:
            # 更新市场数据
            market_data_events = self.data_handler.update_bars()
            
            if market_data_events is None:
                self.continue_backtest = False
                break
                
            # 将市场数据事件加入队列
            for event in market_data_events:
                self.events.append(event)
            
            # 处理所有事件
            while len(self.events) > 0:
                event = self.events.popleft()
                self._process_event(event)
                
            # 等待（如果需要）
            if self.heartbeat > 0:
                time.sleep(self.heartbeat)
        
        # 计算回测耗时
        backtest_end = time.time()
        backtest_time = backtest_end - backtest_start
        
        logger.info(f"回测完成，耗时: {backtest_time:.2f} 秒")
        
        # 计算绩效指标
        metrics = self.analyzer.calculate_metrics()
        
        # 生成回测报告
        report = self.generate_report(backtest_time, metrics)
        
        return report
    
    def generate_report(self, backtest_time: float, metrics: Dict) -> Dict:
        """
        生成回测报告
        
        参数:
        backtest_time (float): 回测耗时（秒）
        metrics (Dict): 绩效指标
        
        返回:
        Dict: 回测报告
        """
        # 基本信息
        report = {
            'backtest_id': self.backtest_id,
            'start_date': self.data_handler.start_date.strftime('%Y-%m-%d'),
            'end_date': self.data_handler.end_date.strftime('%Y-%m-%d'),
            'backtest_time': backtest_time,
            'initial_capital': self.initial_capital,
            'metrics': metrics,
            'total_trades': len(self.all_trades)
        }
        
        # 添加其他信息
        if self.all_trades:
            # 计算交易统计信息
            trade_df = pd.DataFrame(self.all_trades)
            
            # 分类买入和卖出交易
            buy_trades = trade_df[trade_df['direction'] == 'buy']
            sell_trades = trade_df[trade_df['direction'] == 'sell']
            
            report['trade_stats'] = {
                'total_trades': len(trade_df),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'average_trade_value': (trade_df['price'] * trade_df['quantity']).mean()
            }
            
            # 最大交易
            max_trade_idx = (trade_df['price'] * trade_df['quantity']).idxmax()
            max_trade = trade_df.iloc[max_trade_idx]
            
            report['max_trade'] = {
                'symbol': max_trade['symbol'],
                'direction': max_trade['direction'],
                'quantity': int(max_trade['quantity']),
                'price': float(max_trade['price']),
                'value': float(max_trade['price'] * max_trade['quantity']),
                'timestamp': max_trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return report
    
    def plot_results(self) -> None:
        """绘制回测结果图表"""
        self.analyzer.plot_portfolio_performance()
        
    def print_report(self) -> None:
        """打印回测报告"""
        # 计算绩效指标
        metrics = self.analyzer.calculate_metrics()
        
        print("\n=========== 回测报告 ===========")
        print(f"回测ID: {self.backtest_id}")
        print(f"回测期间: {self.data_handler.start_date.strftime('%Y-%m-%d')} 至 {self.data_handler.end_date.strftime('%Y-%m-%d')}")
        print(f"初始资金: {self.initial_capital:,.2f}")
        
        # 打印绩效指标
        self.analyzer.print_metrics()
        
        # 打印交易统计
        if self.all_trades:
            print("\n=== 交易统计 ===")
            print(f"总交易次数: {len(self.all_trades)}")
            
            # 计算买入和卖出次数
            buy_count = sum(1 for trade in self.all_trades if trade['direction'] == 'buy')
            sell_count = sum(1 for trade in self.all_trades if trade['direction'] == 'sell')
            
            print(f"买入交易: {buy_count}")
            print(f"卖出交易: {sell_count}")
            
            # 计算平均交易规模
            total_value = sum(trade['price'] * trade['quantity'] for trade in self.all_trades)
            avg_value = total_value / len(self.all_trades) if self.all_trades else 0
            
            print(f"平均交易金额: {avg_value:,.2f}")
        
        print("================================")
    
    def save_results(self, filename: str) -> None:
        """
        保存回测结果到文件
        
        参数:
        filename (str): 文件名
        """
        # 计算绩效指标
        metrics = self.analyzer.calculate_metrics()
        
        # 生成报告
        report = self.generate_report(0, metrics)
        
        # 转换交易记录为可序列化格式
        serializable_trades = []
        for trade in self.all_trades:
            serializable_trade = trade.copy()
            serializable_trade['timestamp'] = serializable_trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            serializable_trades.append(serializable_trade)
            
        report['trades'] = serializable_trades
        
        # 保存到文件
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"回测结果已保存到 {filename}")

class SimpleExecutionHandler(ExecutionHandler):
    """简单订单执行处理器，假设所有订单都能够以当前价格立即成交"""
    
    def __init__(self, data_handler: DataHandler, commission: float = 0.0):
        """
        初始化执行处理器
        
        参数:
        data_handler (DataHandler): 数据处理器
        commission (float): 佣金率
        """
        self.data_handler = data_handler
        self.commission = commission
    
    def execute_order(self, event: OrderEvent) -> List[FillEvent]:
        """
        执行订单并生成成交事件
        
        参数:
        event (OrderEvent): 订单事件
        
        返回:
        List[FillEvent]: 成交事件列表
        """
        if event.status != OrderStatus.CREATED:
            logger.warning(f"订单状态不是CREATED: {event.order_id}")
            return []
            
        # 获取当前市场数据
        current_data = self.data_handler.get_latest_data(event.symbol)
        
        if current_data.empty:
            logger.warning(f"无法获取 {event.symbol} 的当前市场数据")
            return []
            
        # 使用收盘价作为成交价
        fill_price = current_data['close'].iloc[-1]
        
        # 计算佣金
        commission = self.calculate_commission(event.quantity, fill_price)
        
        # 创建成交事件
        fill_event = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            direction=event.direction,
            quantity=event.quantity,
            price=fill_price,
            commission=commission,
            order_id=event.order_id
        )
        
        return [fill_event]
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """
        计算佣金
        
        参数:
        quantity (int): 数量
        price (float): 价格
        
        返回:
        float: 佣金
        """
        return quantity * price * self.commission

class BasicRiskManager(RiskManager):
    """基本风险管理器，实现简单的风险控制规则"""
    
    def __init__(self, max_order_size: int = 10000, max_position_size: int = 100000):
        """
        初始化风险管理器
        
        参数:
        max_order_size (int): 最大订单规模
        max_position_size (int): 最大持仓规模
        """
        self.max_order_size = max_order_size
        self.max_position_size = max_position_size
        self.positions = {}  # 当前持仓
    
    def process_orders(self, orders: List[OrderEvent]) -> List[OrderEvent]:
        """
        处理订单，应用风险管理规则
        
        参数:
        orders (List[OrderEvent]): 原始订单列表
        
        返回:
        List[OrderEvent]: 经过风险管理处理后的订单列表
        """
        processed_orders = []
        
        for order in orders:
            # 检查订单规模
            if abs(order.quantity) > self.max_order_size:
                logger.warning(f"订单规模 {order.quantity} 超过最大限制 {self.max_order_size}")
                
                # 调整订单规模
                order.quantity = self.max_order_size if order.quantity > 0 else -self.max_order_size
            
            # 检查持仓规模
            symbol = order.symbol
            current_position = self.positions.get(symbol, 0)
            
            if order.direction == OrderDirection.BUY:
                new_position = current_position + order.quantity
            else:  # SELL
                new_position = current_position - order.quantity
                
            if abs(new_position) > self.max_position_size:
                logger.warning(f"持仓规模 {new_position} 超过最大限制 {self.max_position_size}")
                
                # 计算可接受的订单规模
                if order.direction == OrderDirection.BUY:
                    max_allowed = self.max_position_size - current_position
                    order.quantity = min(order.quantity, max_allowed)
                else:  # SELL
                    max_allowed = current_position + self.max_position_size
                    order.quantity = min(order.quantity, max_allowed)
            
            if order.quantity > 0:
                processed_orders.append(order)
                
                # 更新持仓
                if order.direction == OrderDirection.BUY:
                    self.positions[symbol] = current_position + order.quantity
                else:  # SELL
                    self.positions[symbol] = current_position - order.quantity
        
        return processed_orders
    
    def update_position(self, symbol: str, quantity: int, direction: OrderDirection) -> None:
        """
        更新持仓
        
        参数:
        symbol (str): 标的代码
        quantity (int): 数量
        direction (OrderDirection): 方向
        """
        current_position = self.positions.get(symbol, 0)
        
        if direction == OrderDirection.BUY:
            self.positions[symbol] = current_position + quantity
        else:  # SELL
            self.positions[symbol] = current_position - quantity 