#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量化回测引擎 - 基于事件驱动的回测框架
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import time

# 日志配置
logger = logging.getLogger('BacktestEngine')

# 定义事件类型
class EventType(Enum):
    """事件类型枚举"""
    MARKET_DATA = 'market_data'  # 市场数据更新
    SIGNAL = 'signal'            # 交易信号
    ORDER = 'order'              # 订单
    FILL = 'fill'                # 订单成交
    PORTFOLIO = 'portfolio'      # 组合更新
    
# 定义订单类型
class OrderType(Enum):
    """订单类型枚举"""
    MARKET = 'market'           # 市价单
    LIMIT = 'limit'             # 限价单
    STOP = 'stop'               # 止损单
    STOP_LIMIT = 'stop_limit'   # 止损限价单
    
# 定义订单方向
class OrderDirection(Enum):
    """订单方向枚举"""
    BUY = 'buy'
    SELL = 'sell'
    
# 定义订单状态
class OrderStatus(Enum):
    """订单状态枚举"""
    CREATED = 'created'
    SUBMITTED = 'submitted'
    PARTIAL = 'partial'
    FILLED = 'filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    
# 事件基类
@dataclass
class Event:
    """事件基类"""
    type: EventType
    timestamp: datetime.datetime
    
# 市场数据事件
@dataclass
class MarketDataEvent(Event):
    """市场数据事件"""
    symbol: str
    data: pd.DataFrame
    
    def __post_init__(self):
        self.type = EventType.MARKET_DATA
        
# 信号事件
@dataclass
class SignalEvent(Event):
    """交易信号事件"""
    symbol: str
    direction: OrderDirection
    strength: float = 1.0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def __post_init__(self):
        self.type = EventType.SIGNAL
        
# 订单事件
@dataclass
class OrderEvent(Event):
    """订单事件"""
    symbol: str
    order_type: OrderType
    direction: OrderDirection
    quantity: int
    order_id: Optional[str] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.CREATED
    
    def __post_init__(self):
        self.type = EventType.ORDER
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"
            
# 成交事件
@dataclass
class FillEvent(Event):
    """成交事件"""
    symbol: str
    direction: OrderDirection
    quantity: int
    price: float
    commission: float
    order_id: str
    
    def __post_init__(self):
        self.type = EventType.FILL
        
# 组合事件
@dataclass
class PortfolioEvent(Event):
    """组合更新事件"""
    portfolio_value: float
    cash: float
    holdings: Dict[str, Dict]
    
    def __post_init__(self):
        self.type = EventType.PORTFOLIO

# 数据处理器接口
class DataHandler(ABC):
    """市场数据处理器抽象基类"""
    
    @abstractmethod
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        获取最新的n条市场数据
        
        参数:
        symbol (str): 股票代码
        n (int): 获取的数据条数
        
        返回:
        DataFrame: 包含请求的市场数据
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @abstractmethod
    def update_bars(self) -> Optional[List[MarketDataEvent]]:
        """
        更新市场数据，并返回相应的市场数据事件
        
        返回:
        List[MarketDataEvent]: 市场数据事件列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @property
    @abstractmethod
    def symbols(self) -> List[str]:
        """
        获取所有交易标的列表
        
        返回:
        List[str]: 交易标的列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @property
    @abstractmethod
    def start_date(self) -> datetime.datetime:
        """获取回测开始日期"""
        raise NotImplementedError("子类必须实现此方法")
        
    @property
    @abstractmethod
    def end_date(self) -> datetime.datetime:
        """获取回测结束日期"""
        raise NotImplementedError("子类必须实现此方法")
    
    @abstractmethod
    def get_data_by_date(self, symbol: str, date: Union[str, datetime.datetime]) -> pd.DataFrame:
        """
        获取指定日期的市场数据
        
        参数:
        symbol (str): 股票代码
        date (Union[str, datetime.datetime]): 日期
        
        返回:
        DataFrame: 包含指定日期的市场数据
        """
        raise NotImplementedError("子类必须实现此方法")
    
# 回测用的历史数据处理器实现
class HistoricalDataHandler(DataHandler):
    """使用历史数据进行回测的数据处理器"""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], start_date: Optional[Union[str, datetime.datetime]] = None, 
                 end_date: Optional[Union[str, datetime.datetime]] = None):
        """
        初始化历史数据处理器
        
        参数:
        data_dict (Dict[str, pd.DataFrame]): 股票代码到历史数据的映射
        start_date (Union[str, datetime.datetime], optional): 回测开始日期
        end_date (Union[str, datetime.datetime], optional): 回测结束日期
        """
        self.data_dict = data_dict
        self._symbols = list(data_dict.keys())
        
        # 检查所有DataFrame的索引是否都是日期类型
        for symbol, df in data_dict.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"数据框 {symbol} 的索引不是DatetimeIndex类型")
        
        # 获取所有数据集中的最早和最晚日期
        all_dates = pd.DatetimeIndex([])
        for df in data_dict.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()
        
        # 设置回测日期范围
        self._start_date = pd.to_datetime(start_date) if start_date else all_dates[0]
        self._end_date = pd.to_datetime(end_date) if end_date else all_dates[-1]
        
        # 过滤数据为指定的日期范围
        self.filtered_data = {}
        for symbol, df in data_dict.items():
            self.filtered_data[symbol] = df.loc[(df.index >= self._start_date) & (df.index <= self._end_date)]
        
        # 获取所有交易日
        self.trading_days = all_dates[(all_dates >= self._start_date) & (all_dates <= self._end_date)]
        
        # 当前位置和最新数据
        self.current_idx = 0
        self.current_date = self.trading_days[0] if len(self.trading_days) > 0 else None
        self.latest_data = {symbol: [] for symbol in self._symbols}
        
        logger.info(f"初始化历史数据处理器, 回测期间: {self._start_date} 至 {self._end_date}, 共 {len(self.trading_days)} 个交易日")
    
    @property
    def symbols(self) -> List[str]:
        """获取所有交易标的列表"""
        return self._symbols
    
    @property
    def start_date(self) -> datetime.datetime:
        """获取回测开始日期"""
        return self._start_date
    
    @property
    def end_date(self) -> datetime.datetime:
        """获取回测结束日期"""
        return self._end_date
    
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        获取最新的n条市场数据
        
        参数:
        symbol (str): 股票代码
        n (int): 获取的数据条数
        
        返回:
        DataFrame: 包含请求的市场数据
        """
        try:
            latest_bars = self.latest_data[symbol]
            return pd.DataFrame(latest_bars[-n:])
        except (KeyError, IndexError):
            logger.warning(f"获取 {symbol} 的最新数据时出错，可能没有足够的数据")
            return pd.DataFrame()
    
    def update_bars(self) -> Optional[List[MarketDataEvent]]:
        """
        更新市场数据，并返回相应的市场数据事件
        
        返回:
        List[MarketDataEvent]: 市场数据事件列表
        """
        if self.current_idx >= len(self.trading_days):
            return None  # 回测结束
            
        current_date = self.trading_days[self.current_idx]
        events = []
        
        for symbol in self._symbols:
            if current_date in self.filtered_data[symbol].index:
                # 获取当前日期的市场数据
                bar = self.filtered_data[symbol].loc[current_date].to_dict()
                bar['timestamp'] = current_date
                
                # 将数据添加到最新数据列表
                self.latest_data[symbol].append(bar)
                
                # 创建市场数据事件
                df = pd.DataFrame([bar])
                df.set_index('timestamp', inplace=True)
                event = MarketDataEvent(
                    type=EventType.MARKET_DATA,  # 显式设置类型
                    timestamp=current_date,
                    symbol=symbol,
                    data=df
                )
                events.append(event)
        
        # 更新当前位置
        self.current_idx += 1
        self.current_date = current_date
        
        return events
    
    def get_data_by_date(self, symbol: str, date: Union[str, datetime.datetime]) -> pd.DataFrame:
        """
        获取指定日期的市场数据
        
        参数:
        symbol (str): 股票代码
        date (Union[str, datetime.datetime]): 日期
        
        返回:
        DataFrame: 包含指定日期的市场数据
        """
        date = pd.to_datetime(date)
        if symbol in self.filtered_data and date in self.filtered_data[symbol].index:
            data = self.filtered_data[symbol].loc[date].to_dict()
            data['timestamp'] = date
            df = pd.DataFrame([data])
            df.set_index('timestamp', inplace=True)
            return df
        else:
            logger.warning(f"无法获取 {symbol} 在 {date} 的数据")
            return pd.DataFrame()

# 策略接口
class Strategy(ABC):
    """交易策略抽象基类"""
    
    def __init__(self):
        """初始化策略"""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_signals(self, event: Event) -> List[SignalEvent]:
        """
        计算交易信号
        
        参数:
        event (Event): 触发信号计算的事件
        
        返回:
        List[SignalEvent]: 交易信号事件列表
        """
        raise NotImplementedError("子类必须实现此方法")

# 投资组合接口
class Portfolio(ABC):
    """投资组合抽象基类"""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        初始化投资组合
        
        参数:
        initial_capital (float): 初始资金
        """
        self.initial_capital = initial_capital
        self.current_holdings = {
            'cash': initial_capital,
            'total': initial_capital
        }
    
    @abstractmethod
    def update_on_signal(self, event: SignalEvent) -> List[OrderEvent]:
        """
        基于信号更新投资组合
        
        参数:
        event (SignalEvent): 信号事件
        
        返回:
        List[OrderEvent]: 订单事件列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @abstractmethod
    def update_on_fill(self, event: FillEvent) -> List[Event]:
        """
        基于成交更新投资组合
        
        参数:
        event (FillEvent): 成交事件
        
        返回:
        List[Event]: 后续事件列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    @abstractmethod
    def update_portfolio_value(self) -> List[PortfolioEvent]:
        """
        更新投资组合价值
        
        返回:
        List[PortfolioEvent]: 投资组合更新事件
        """
        raise NotImplementedError("子类必须实现此方法")

# 执行接口
class ExecutionHandler(ABC):
    """订单执行处理器抽象基类"""
    
    @abstractmethod
    def execute_order(self, event: OrderEvent) -> List[FillEvent]:
        """
        执行订单并生成成交事件
        
        参数:
        event (OrderEvent): 订单事件
        
        返回:
        List[FillEvent]: 成交事件列表
        """
        raise NotImplementedError("子类必须实现此方法")

# 风险管理接口
class RiskManager(ABC):
    """风险管理器抽象基类"""
    
    @abstractmethod
    def process_orders(self, orders: List[OrderEvent]) -> List[OrderEvent]:
        """
        处理订单，应用风险管理规则
        
        参数:
        orders (List[OrderEvent]): 原始订单列表
        
        返回:
        List[OrderEvent]: 经过风险管理处理后的订单列表
        """
        raise NotImplementedError("子类必须实现此方法")

# 绩效评估类
class PerformanceAnalyzer:
    """绩效分析器"""
    
    def __init__(self, benchmark_data: Optional[pd.DataFrame] = None):
        """
        初始化绩效分析器
        
        参数:
        benchmark_data (DataFrame, optional): 基准数据
        """
        self.portfolio_values = []
        self.benchmark_data = benchmark_data
        self.metrics = {}
    
    def update(self, event: PortfolioEvent) -> None:
        """
        更新投资组合价值历史
        
        参数:
        event (PortfolioEvent): 投资组合更新事件
        """
        self.portfolio_values.append({
            'timestamp': event.timestamp,
            'portfolio_value': event.portfolio_value,
            'cash': event.cash
        })
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        计算回报率
        
        返回:
        DataFrame: 包含投资组合价值和回报率的DataFrame
        """
        if not self.portfolio_values:
            return pd.DataFrame()
            
        # 创建投资组合价值的DataFrame
        df = pd.DataFrame(self.portfolio_values)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 计算每日回报率
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # 计算累积回报率
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        # 添加基准数据（如果有）
        if self.benchmark_data is not None:
            benchmark = self.benchmark_data.copy()
            if 'close' in benchmark.columns:
                benchmark['daily_return'] = benchmark['close'].pct_change()
                benchmark['cumulative_return'] = (1 + benchmark['daily_return']).cumprod() - 1
                
                # 合并到结果DataFrame
                df = df.join(benchmark[['close', 'daily_return', 'cumulative_return']], 
                             rsuffix='_benchmark')
        
        return df
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算绩效指标
        
        返回:
        Dict[str, float]: 绩效指标字典
        """
        returns_df = self.calculate_returns()
        
        if returns_df.empty or 'daily_return' not in returns_df.columns:
            return {}
        
        # 计算年化回报率
        total_days = (returns_df.index[-1] - returns_df.index[0]).days
        annual_factor = 252 / total_days if total_days > 0 else 0
        
        # 计算投资组合的年化回报率
        total_return = returns_df['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # 计算波动率
        volatility = returns_df['daily_return'].std() * np.sqrt(252)
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = returns_df['portfolio_value'] / returns_df['portfolio_value'].iloc[0] - 1
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        # 计算胜率
        winning_days = (returns_df['daily_return'] > 0).sum()
        total_days = len(returns_df)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # 计算基准相关指标（如果有）
        benchmark_metrics = {}
        if 'daily_return_benchmark' in returns_df.columns:
            # 计算基准年化回报率
            benchmark_total_return = returns_df['cumulative_return_benchmark'].iloc[-1]
            benchmark_annual_return = (1 + benchmark_total_return) ** annual_factor - 1
            
            # 计算基准波动率
            benchmark_volatility = returns_df['daily_return_benchmark'].std() * np.sqrt(252)
            
            # 计算相对指标
            alpha = annual_return - benchmark_annual_return
            
            # 计算贝塔
            covariance = returns_df['daily_return'].cov(returns_df['daily_return_benchmark'])
            benchmark_variance = returns_df['daily_return_benchmark'].var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # 计算信息比率
            tracking_error = (returns_df['daily_return'] - returns_df['daily_return_benchmark']).std() * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            benchmark_metrics = {
                'benchmark_annual_return': benchmark_annual_return,
                'benchmark_volatility': benchmark_volatility,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio
            }
        
        # 组合所有指标
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            **benchmark_metrics
        }
        
        return self.metrics
    
    def plot_portfolio_performance(self) -> None:
        """绘制投资组合绩效图表"""
        returns_df = self.calculate_returns()
        
        if returns_df.empty:
            logger.warning("没有足够的数据用于绘图")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 绘制投资组合价值
        axes[0].plot(returns_df.index, returns_df['portfolio_value'], label='Portfolio Value')
        axes[0].set_title('Portfolio Value')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制累积回报率
        axes[1].plot(returns_df.index, returns_df['cumulative_return'], label='Portfolio')
        
        if 'cumulative_return_benchmark' in returns_df.columns:
            axes[1].plot(returns_df.index, returns_df['cumulative_return_benchmark'], label='Benchmark')
            
        axes[1].set_title('Cumulative Return')
        axes[1].set_ylabel('Return')
        axes[1].legend()
        axes[1].grid(True)
        
        # 计算回撤
        portfolio_max = returns_df['portfolio_value'].cummax()
        drawdown = (returns_df['portfolio_value'] - portfolio_max) / portfolio_max
        
        # 绘制回撤
        axes[2].fill_between(returns_df.index, 0, drawdown, color='red', alpha=0.3)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def print_metrics(self) -> None:
        """打印绩效指标"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            logger.warning("没有足够的数据用于计算绩效指标")
            return
        
        print("\n=== 绩效指标 ===")
        print(f"总回报率: {metrics['total_return']:.2%}")
        print(f"年化回报率: {metrics['annual_return']:.2%}")
        print(f"波动率: {metrics['volatility']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        
        if 'benchmark_annual_return' in metrics:
            print("\n=== 相对基准指标 ===")
            print(f"基准年化回报率: {metrics['benchmark_annual_return']:.2%}")
            print(f"Alpha: {metrics['alpha']:.2%}")
            print(f"Beta: {metrics['beta']:.2f}")
            print(f"信息比率: {metrics['information_ratio']:.2f}")

# 回测引擎
class BacktestEngine:
    """基于事件驱动的回测引擎"""
    
    def __init__(self, start_date: Union[str, datetime.datetime], 
                end_date: Union[str, datetime.datetime], 
                symbols: List[str], 
                initial_capital: float = 100000.0,
                heartbeat: float = 0.0,
                benchmark_symbol: Optional[str] = None):
        """
        初始化回测引擎
        
        参数:
        start_date (Union[str, datetime.datetime]): 回测开始日期
        end_date (Union[str, datetime.datetime]): 回测结束日期
        symbols (List[str]): 回测标的代码列表
        initial_capital (float): 初始资金
        heartbeat (float): 心跳间隔（秒），用于控制回测速度
        benchmark_symbol (str, optional): 基准标的代码
        """
        self.start_date = start_date if isinstance(start_date, datetime.datetime) else pd.to_datetime(start_date)
        self.end_date = end_date if isinstance(end_date, datetime.datetime) else pd.to_datetime(end_date)
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.benchmark_symbol = benchmark_symbol
        
        # 组件
        self.data_handler = None
        self.strategy = None
        self.portfolio = None
        self.execution_handler = None
        self.risk_manager = None
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 事件队列
        self.events = []
        
        # 回测结果
        self.metrics = {}
        self.equity_curve = None
    
    def set_strategy(self, strategy: Strategy) -> None:
        """
        设置交易策略
        
        参数:
        strategy (Strategy): 交易策略实例
        """
        self.strategy = strategy
        logger.info(f"设置策略: {strategy.__class__.__name__}")
    
    def set_portfolio(self, portfolio: Portfolio) -> None:
        """
        设置投资组合
        
        参数:
        portfolio (Portfolio): 投资组合实例
        """
        self.portfolio = portfolio
        logger.info(f"设置投资组合: {portfolio.__class__.__name__}")
    
    def set_execution_handler(self, execution_handler: ExecutionHandler) -> None:
        """
        设置执行处理器
        
        参数:
        execution_handler (ExecutionHandler): 执行处理器实例
        """
        self.execution_handler = execution_handler
        logger.info(f"设置执行处理器: {execution_handler.__class__.__name__}")
    
    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        """
        设置风险管理器
        
        参数:
        risk_manager (RiskManager): 风险管理器实例
        """
        self.risk_manager = risk_manager
        logger.info(f"设置风险管理器: {risk_manager.__class__.__name__}")
    
    def run(self, data_source: DataHandler = None) -> Dict[str, Any]:
        """
        运行回测
        
        参数:
        data_source (DataHandler, optional): 数据源实例
        
        返回:
        Dict[str, Any]: 回测结果
        """
        # 设置数据处理器
        if data_source is not None:
            self.data_handler = data_source
        
        if self.data_handler is None:
            raise ValueError("必须先设置数据处理器")
        
        if self.strategy is None:
            raise ValueError("必须先设置交易策略")
        
        if self.portfolio is None:
            raise ValueError("必须先设置投资组合")
        
        # 设置默认的执行处理器
        if self.execution_handler is None:
            from backtest_engine import SimulatedExecutionHandler
            self.execution_handler = SimulatedExecutionHandler()
            logger.info("使用默认的模拟执行处理器")
        
        # 设置默认的风险管理器
        if self.risk_manager is None:
            from backtest_engine import BasicRiskManager
            self.risk_manager = BasicRiskManager()
            logger.info("使用默认的基本风险管理器")
        
        logger.info("开始回测...")
        logger.info(f"回测时间段: {self.start_date} - {self.end_date}")
        logger.info(f"交易标的: {', '.join(self.symbols)}")
        logger.info(f"初始资金: {self.initial_capital}")
        
        # 开始回测循环
        try:
            continue_backtest = True
            
            while continue_backtest:
                # 更新市场数据
                market_events = self.data_handler.update_bars()
                
                if market_events is None or len(market_events) == 0:
                    # 没有更多数据，结束回测
                    continue_backtest = False
                    break
                
                # 处理市场数据事件
                for market_event in market_events:
                    self.events.append(market_event)
                
                # 处理所有事件
                while len(self.events) > 0:
                    event = self.events.pop(0)
                    
                    # 根据事件类型分发处理
                    if event.type == EventType.MARKET_DATA:
                        # 计算交易信号
                        signals = self.strategy.calculate_signals(event)
                        for signal in signals:
                            self.events.append(signal)
                    
                    elif event.type == EventType.SIGNAL:
                        # 生成订单
                        orders = self.portfolio.update_on_signal(event)
                        
                        # 应用风险管理
                        if self.risk_manager is not None and orders:
                            orders = self.risk_manager.process_orders(orders)
                        
                        # 添加订单到事件队列
                        for order in orders:
                            self.events.append(order)
                    
                    elif event.type == EventType.ORDER:
                        # 执行订单
                        fills = self.execution_handler.execute_order(event)
                        for fill in fills:
                            self.events.append(fill)
                    
                    elif event.type == EventType.FILL:
                        # 更新投资组合
                        portfolio_events = self.portfolio.update_on_fill(event)
                        for port_event in portfolio_events:
                            self.events.append(port_event)
                    
                    elif event.type == EventType.PORTFOLIO:
                        # 更新绩效分析
                        self.performance_analyzer.update(event)
                
                # 控制回测速度
                if self.heartbeat > 0:
                    time.sleep(self.heartbeat)
            
            # 计算回测绩效指标
            self.metrics = self.performance_analyzer.calculate_metrics()
            self.equity_curve = self.performance_analyzer.calculate_returns()
            
            logger.info("回测完成")
            logger.info(f"最终投资组合价值: {self.metrics.get('final_equity', 'N/A')}")
            logger.info(f"总回报率: {self.metrics.get('total_return', 0):.2%}")
            logger.info(f"年化回报率: {self.metrics.get('annual_return', 0):.2%}")
            logger.info(f"最大回撤: {self.metrics.get('max_drawdown', 0):.2%}")
            
            # 返回回测结果
            return {
                'metrics': self.metrics,
                'equity_curve': self.equity_curve
            }
            
        except Exception as e:
            logger.error(f"回测过程中出错: {str(e)}", exc_info=True)
            return {
                'error': str(e)
            }
    
    def plot_results(self) -> None:
        """绘制回测结果图表"""
        if not hasattr(self, 'performance_analyzer') or self.performance_analyzer is None:
            logger.error("没有绩效分析器，无法绘图")
            return
        
        try:
            self.performance_analyzer.plot_portfolio_performance()
        except Exception as e:
            logger.error(f"绘制回测结果时出错: {str(e)}")
            raise

# 简化的模拟执行处理器
class SimulatedExecutionHandler(ExecutionHandler):
    """模拟执行处理器"""
    
    def execute_order(self, order: OrderEvent) -> List[FillEvent]:
        """
        执行订单，生成成交事件
        
        参数:
        order (OrderEvent): 订单事件
        
        返回:
        List[FillEvent]: 成交事件列表
        """
        # 模拟成交价格（使用当前收盘价）
        data = self.data_handler.get_latest_data(order.symbol, 1)
        if data is None or data.empty:
            logger.warning(f"无法获取 {order.symbol} 的数据用于成交价格")
            return []
        
        # 获取最新价格
        price = data.iloc[-1]['close']
        
        # 计算成交量（简化起见，全部成交）
        quantity = order.quantity
        
        # 计算手续费（简化为万分之2）
        commission = price * quantity * 0.0002
        
        # 创建成交事件
        fill_event = FillEvent(
            type=EventType.FILL,
            timestamp=order.timestamp,
            symbol=order.symbol,
            direction=order.direction,
            quantity=quantity,
            price=price,
            commission=commission,
            order_id=order.order_id
        )
        
        logger.info(f"订单执行成功: {order.direction} {quantity} {order.symbol} @ {price:.2f}")
        
        return [fill_event]

# 基本风险管理器
class BasicRiskManager(RiskManager):
    """基本风险管理器"""
    
    def __init__(self, max_order_size: int = 1000, max_position_size: int = 10000, 
                max_single_order_value: float = 50000.0):
        """
        初始化基本风险管理器
        
        参数:
        max_order_size (int): 最大订单数量
        max_position_size (int): 最大持仓数量
        max_single_order_value (float): 单个订单最大金额
        """
        self.max_order_size = max_order_size
        self.max_position_size = max_position_size
        self.max_single_order_value = max_single_order_value
        self.current_positions = {}
    
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
            # 检查订单数量是否超过限制
            if order.quantity > self.max_order_size:
                logger.warning(f"订单数量 {order.quantity} 超过限制 {self.max_order_size}，调整为最大值")
                order.quantity = self.max_order_size
            
            # 检查持仓数量是否超过限制
            symbol = order.symbol
            current_position = self.current_positions.get(symbol, 0)
            
            if order.direction == "BUY":
                new_position = current_position + order.quantity
                if new_position > self.max_position_size:
                    # 调整订单数量
                    allowed_increase = max(0, self.max_position_size - current_position)
                    if allowed_increase > 0:
                        logger.warning(f"持仓数量 {new_position} 超过限制 {self.max_position_size}，调整订单")
                        order.quantity = allowed_increase
                    else:
                        logger.warning(f"持仓数量 {current_position} 已达到限制 {self.max_position_size}，拒绝订单")
                        continue
            
            # 如果订单数量为0，跳过
            if order.quantity <= 0:
                continue
            
            # 添加处理后的订单
            processed_orders.append(order)
            
            # 更新当前持仓记录
            if order.direction == "BUY":
                self.current_positions[symbol] = current_position + order.quantity
            elif order.direction == "SELL":
                self.current_positions[symbol] = max(0, current_position - order.quantity)
        
        return processed_orders 