from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from event import EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent
from data import DataHandler
from portfolio import SimplePortfolio

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, start_date: str, end_date: str, symbols: List[str], initial_capital: float = 1000000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.data_handler: Optional[DataHandler] = None
        self.strategy = None
        self.portfolio: Optional[SimplePortfolio] = None
        self.execution_handler = None
        self.risk_manager = None
        self.events: List[Any] = []
        self.metrics: Dict[str, float] = {}

    def run(self) -> Dict[str, Any]:
        """运行回测"""
        if not self.data_handler or not self.strategy or not self.portfolio:
            raise ValueError("Missing required components")

        logger.info("开始回测...")
        logger.info(f"回测区间: {self.start_date} - {self.end_date}")
        logger.info(f"回测标的: {', '.join(self.symbols)}")
        logger.info(f"初始资金: {self.initial_capital:,.2f}")

        # 重置数据处理器
        self.data_handler.reset()
        
        # 初始化投资组合
        self.portfolio = SimplePortfolio(initial_capital=self.initial_capital)
        self.portfolio.data_handler = self.data_handler

        # 主回测循环
        event_count = 0
        signal_count = 0
        order_count = 0
        fill_count = 0
        
        # 使用循环迭代日期
        for i, date in enumerate(self.data_handler.dates):
            # 生成市场数据事件
            market_data = self.data_handler.get_data_for_date(date)
            if not market_data:
                continue
                
            market_event = MarketDataEvent(
                timestamp=date,
                data=market_data
            )
            event_count += 1
            
            # 生成交易信号
            signals = self.strategy.generate_signals(market_event)
            signal_count += len(signals)
            
            # 处理交易信号
            for signal in signals:
                # 创建订单事件
                order = self._create_order_from_signal(signal)
                if order:
                    order_count += 1
                    
                    # 执行订单
                    fill = self._execute_order(order, market_data)
                    if fill:
                        fill_count += 1
                        
                        # 更新投资组合
                        self.portfolio.update(fill)
            
            # 更新投资组合每日状态
            self.portfolio.update_daily(date, market_data)
        
        # 计算回测指标
        self._calculate_metrics()
        
        logger.info(f"回测完成，共处理 {event_count} 个事件")
        logger.info(f"- 生成 {signal_count} 个信号")
        logger.info(f"- 生成 {order_count} 个订单")
        logger.info(f"- 成交 {fill_count} 笔交易")
        
        return {
            "events": self.events,
            "metrics": self.metrics
        }

    def _calculate_metrics(self):
        """计算回测指标"""
        if not self.portfolio.history:
            return

        # 获取初始和最终资金
        initial_equity = self.portfolio.history[0].equity
        final_equity = self.portfolio.history[-1].equity

        # 计算收益率
        total_return = (final_equity - initial_equity) / initial_equity
        annual_return = total_return * (252 / len(self.portfolio.history))

        # 计算最大回撤
        max_drawdown = 0.0
        peak = initial_equity
        for event in self.portfolio.history:
            if event.equity > peak:
                peak = event.equity
            drawdown = (peak - event.equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算收益率序列和波动率
        returns = []
        for i in range(1, len(self.portfolio.history)):
            prev_equity = self.portfolio.history[i-1].equity
            curr_equity = self.portfolio.history[i].equity
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)
        
        if returns:
            import numpy as np
            daily_returns_std = np.std(returns)
            sharpe_ratio = (annual_return - 0.03) / (daily_returns_std * (252 ** 0.5)) if daily_returns_std > 0 else 0
        else:
            sharpe_ratio = 0

        # 保存指标
        self.metrics = {
            "final_equity": final_equity,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }

    def _create_order_from_signal(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """根据交易信号创建订单事件"""
        try:
            if signal.signal_type == 'LONG':
                # 计算可用资金和仓位大小
                cash = self.portfolio.cash
                max_position = cash * 0.95 / signal.price  # 使用95%的可用资金
                
                # 默认仓位：每只股票分配15%的可用资金
                position_size = int(cash * 0.15 / signal.price)
                
                # 约束仓位大小
                position_size = min(position_size, max_position)
                position_size = max(position_size, 100)  # 最小100股
                
                # 创建买入订单
                return OrderEvent(
                    timestamp=signal.datetime,
                    symbol=signal.symbol,
                    order_type='MARKET',
                    quantity=position_size,
                    direction='BUY',
                    price=signal.price
                )
                
            elif signal.signal_type == 'EXIT':
                # 获取当前持仓
                position = self.portfolio.positions.get(signal.symbol, 0)
                
                if position > 0:
                    # 创建卖出订单
                    return OrderEvent(
                        timestamp=signal.datetime,
                        symbol=signal.symbol,
                        order_type='MARKET',
                        quantity=position,
                        direction='SELL',
                        price=signal.price
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"创建订单时发生错误: {str(e)}")
            return None

    def _execute_order(self, order: OrderEvent, market_data: Dict[str, Any]) -> Optional[FillEvent]:
        """执行订单，返回成交事件"""
        try:
            # 简单执行，假设所有订单都能以当前价格成交
            symbol_data = market_data.get(order.symbol)
            if not symbol_data:
                return None
                
            # 获取成交价格 (当前收盘价)
            fill_price = float(symbol_data['close'])
            
            # 计算手续费 (双边万分之五)
            commission = fill_price * order.quantity * 0.0005
            
            # 创建成交事件
            return FillEvent(
                timestamp=order.timestamp,
                symbol=order.symbol,
                quantity=order.quantity,
                direction=order.direction,
                price=fill_price,
                commission=commission
            )
            
        except Exception as e:
            logger.error(f"执行订单时发生错误: {str(e)}")
            return None 