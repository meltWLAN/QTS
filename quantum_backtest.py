#!/usr/bin/env python3
"""
超神量子回测系统 - 高级回测脚本
集成数据连接器和量子爆发策略
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入SuperQuantumNetwork中的组件
try:
    from SuperQuantumNetwork.data import TushareDataHandler
    from SuperQuantumNetwork.quantum_burst_strategy_enhanced import QuantumBurstStrategyEnhanced
    from SuperQuantumNetwork.backtest_engine import BacktestEngine
    from SuperQuantumNetwork.backtest_engine import SimulatedExecutionHandler, BasicRiskManager
    from SuperQuantumNetwork.event import EventType, MarketDataEvent
    
    print("成功导入SuperQuantumNetwork组件")
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"无法导入SuperQuantumNetwork组件: {str(e)}")
    print("将使用基础回测模式")
    MODULES_AVAILABLE = False

# 导入日志系统
try:
    from src.utils.logger import setup_backtest_logger
    logger = setup_backtest_logger(strategy_name="量子爆发策略")
except ImportError:
    # 如果没有专用的日志系统，创建基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"quantum_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("QuantumBacktest")


class SimplePortfolio:
    """简化的投资组合类，用于回测"""
    
    def __init__(self, initial_capital=1000000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # 股票代码 -> 持仓数量
        self.history = []    # 投资组合历史记录
        self.data_handler = None
        
    def update(self, fill_event):
        """处理成交事件，更新投资组合"""
        symbol = fill_event.symbol
        direction = fill_event.direction
        quantity = fill_event.quantity
        price = fill_event.price
        commission = fill_event.commission
        
        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = 0
            
        if direction == "BUY":
            self.positions[symbol] += quantity
            self.capital -= (price * quantity + commission)
        elif direction == "SELL":
            self.positions[symbol] -= quantity
            self.capital += (price * quantity - commission)
            
        # 记录状态
        self.history.append({
            'timestamp': fill_event.timestamp,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'capital': self.capital,
            'positions': self.positions.copy()
        })
        
    def update_daily(self, timestamp, market_data):
        """更新每日投资组合状态"""
        portfolio_value = self.capital
        
        # 计算持仓价值
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in market_data:
                price = market_data[symbol]['close']
                portfolio_value += price * quantity
                
        # 记录每日状态
        self.history.append({
            'timestamp': timestamp,
            'equity': portfolio_value,
            'cash': self.capital
        })


def run_backtest():
    """运行回测"""
    # 回测参数设置
    start_date = "20230101"
    end_date = "20231231"
    initial_capital = 1000000.0
    
    # 回测标的
    symbols = [
        "000001.SZ",  # 平安银行
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
        "601318.SH",  # 中国平安
        "000333.SZ",  # 美的集团
        "600036.SH",  # 招商银行
        "601888.SH",  # 中国中免
    ]
    benchmark = "000300.SH"  # 沪深300作为基准
    
    logger.info("=============================================")
    logger.info("超神量子回测系统 - 高级回测")
    logger.info("量子爆发策略")
    logger.info(f"回测区间: {start_date} - {end_date}")
    logger.info(f"回测标的: {len(symbols)} 只股票")
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info("=============================================")
    
    # 初始化Tushare连接器
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    if MODULES_AVAILABLE:
        try:
            # 使用SuperQuantumNetwork的完整回测引擎
            logger.info("使用SuperQuantumNetwork的完整回测引擎")
            
            # 添加基准指数到回测标的中
            all_symbols = symbols.copy()
            if benchmark not in all_symbols:
                all_symbols.append(benchmark)
                
            # 初始化数据处理器
            logger.info(f"初始化TushareDataHandler，使用Token: {token[:8]}...")
            data_handler = TushareDataHandler(all_symbols, start_date, end_date, token=token)
            
            # 初始化回测引擎
            engine = BacktestEngine(
                start_date=start_date,
                end_date=end_date,
                symbols=all_symbols,
                initial_capital=initial_capital,
                heartbeat=0.0,
                benchmark_symbol=benchmark
            )
            
            # 设置数据源
            engine.data_handler = data_handler
            
            # 初始化策略
            strategy = QuantumBurstStrategyEnhanced(data_handler=data_handler)
            engine.set_strategy(strategy)
            
            # 初始化投资组合
            portfolio = SimplePortfolio(initial_capital=initial_capital)
            portfolio.data_handler = data_handler
            engine.set_portfolio(portfolio)
            
            # 设置执行处理器和风险管理器
            engine.set_execution_handler(SimulatedExecutionHandler())
            engine.set_risk_manager(BasicRiskManager())
            
            # 运行回测
            logger.info("开始运行回测...")
            results = engine.run()
            
            # 显示回测结果
            logger.info("回测完成，显示结果:")
            for key, value in results['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.2%}" if 'return' in key.lower() or 'drawdown' in key.lower() else f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
            
            # 绘制回测结果
            engine.plot_results()
            
        except Exception as e:
            logger.error(f"运行SuperQuantumNetwork回测引擎时出错: {str(e)}")
            logger.error("切换到简化回测模式")
            run_simple_backtest(symbols, start_date, end_date, initial_capital, token)
    else:
        # 如果完整的回测引擎不可用，使用简化的回测
        run_simple_backtest(symbols, start_date, end_date, initial_capital, token)


def run_simple_backtest(symbols, start_date, end_date, initial_capital, token):
    """运行简化版回测"""
    logger.info("使用简化版回测引擎")
    
    # 这里可以实现一个简化的回测逻辑
    # 使用之前示例中的模拟数据源和简单MA策略
    
    # 模拟Tushare数据源
    class SimpleTushareSource:
        def __init__(self, token):
            self.token = token
            
        def get_daily_data(self, code, start_date, end_date):
            """生成模拟日线数据"""
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')
            date_range = [start + timedelta(days=x) for x in range((end-start).days + 1)]
            date_range = [d for d in date_range if d.weekday() < 5]  # 只保留工作日
            
            data = []
            price = 100.0  # 起始价格
            
            for date in date_range:
                change = np.random.normal(0, 1) / 100
                price = price * (1 + change)
                
                open_price = price * (1 + np.random.normal(0, 0.005))
                high_price = price * (1 + abs(np.random.normal(0, 0.01)))
                low_price = price * (1 - abs(np.random.normal(0, 0.01)))
                close_price = price
                volume = np.random.randint(1000, 10000)
                
                data.append({
                    'date': date.strftime('%Y%m%d'),
                    'code': code,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'amount': volume * close_price
                })
            
            return pd.DataFrame(data)
    
    # 初始化数据源
    data_source = SimpleTushareSource(token)
    
    # 回测循环
    portfolio = {symbol: 0 for symbol in symbols}  # 持仓
    cash = initial_capital  # 现金
    trades = []  # 交易记录
    daily_values = []  # 每日投资组合价值
    
    # 遍历每个股票
    for symbol in symbols:
        logger.info(f"正在获取 {symbol} 的历史数据...")
        
        try:
            # 获取股票数据
            df = data_source.get_daily_data(symbol, start_date=start_date, end_date=end_date)
            
            if df is None or len(df) == 0:
                logger.warning(f"获取 {symbol} 数据失败或数据为空，跳过该股票")
                continue
                
            logger.info(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            
            # 计算技术指标 - 简单的移动平均线策略
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # 生成交易信号
            df['signal'] = 0
            df.loc[df['MA5'] > df['MA20'], 'signal'] = 1  # 买入信号
            df.loc[df['MA5'] < df['MA20'], 'signal'] = -1  # 卖出信号
            
            # 移除NaN值
            df = df.dropna()
            
            # 回测
            position = 0
            for i in range(1, len(df)):
                prev_signal = df.iloc[i-1]['signal']
                curr_signal = df.iloc[i]['signal']
                price = df.iloc[i]['close']
                date = df.iloc[i]['date']
                
                # 信号变化时交易
                if prev_signal != curr_signal:
                    # 买入信号
                    if curr_signal == 1 and position == 0:
                        # 计算买入股数（使用20%资金买入）
                        shares = int(cash * 0.2 / price)
                        
                        if shares > 0:
                            cost = shares * price
                            cash -= cost
                            portfolio[symbol] += shares
                            position = 1
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'cost': cost,
                                'cash': cash
                            })
                            
                            logger.info(f"买入 {symbol}: 日期={date}, 价格={price:.2f}, 数量={shares}, 花费={cost:.2f}, 剩余现金={cash:.2f}")
                    
                    # 卖出信号
                    elif curr_signal == -1 and position == 1:
                        shares = portfolio[symbol]
                        
                        if shares > 0:
                            revenue = shares * price
                            cash += revenue
                            portfolio[symbol] = 0
                            position = 0
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'price': price,
                                'shares': shares,
                                'revenue': revenue,
                                'cash': cash
                            })
                            
                            logger.info(f"卖出 {symbol}: 日期={date}, 价格={price:.2f}, 数量={shares}, 收入={revenue:.2f}, 现金={cash:.2f}")
                
                # 记录每日投资组合价值
                portfolio_value = cash
                for s, shares in portfolio.items():
                    if shares > 0:
                        # 如果是当前股票，使用当前价格
                        if s == symbol:
                            stock_value = shares * price
                        else:
                            # 对于其他股票，尝试获取当天价格，如果没有则使用最后已知价格
                            # 这里简化处理，使用上次交易价格
                            last_trade = None
                            for trade in reversed(trades):
                                if trade['symbol'] == s:
                                    last_trade = trade
                                    break
                            
                            if last_trade:
                                stock_value = shares * last_trade['price']
                            else:
                                stock_value = 0
                        
                        portfolio_value += stock_value
                
                daily_values.append({
                    'date': date,
                    'symbol': symbol,
                    'close': price,
                    'portfolio_value': portfolio_value
                })
        
        except Exception as e:
            logger.error(f"处理 {symbol} 时发生错误: {str(e)}")
    
    # 计算最终投资组合价值
    final_portfolio_value = cash
    for symbol, shares in portfolio.items():
        if shares > 0:
            # 获取最后交易价格
            last_trade = None
            for trade in reversed(trades):
                if trade['symbol'] == symbol:
                    last_trade = trade
                    break
            
            if last_trade:
                stock_value = shares * last_trade['price']
                final_portfolio_value += stock_value
    
    # 计算收益率
    total_return = (final_portfolio_value - initial_capital) / initial_capital
    
    # 显示回测结果
    logger.info("=============================================")
    logger.info("回测完成，结果如下:")
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info(f"最终资金: {final_portfolio_value:,.2f}")
    logger.info(f"总收益: {final_portfolio_value - initial_capital:,.2f}")
    logger.info(f"收益率: {total_return:.2%}")
    logger.info(f"交易次数: {len(trades)}")
    logger.info("=============================================")
    
    # 返回结果
    return {
        'initial_capital': initial_capital,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'trade_count': len(trades),
        'trades': trades,
        'daily_values': daily_values
    }


if __name__ == "__main__":
    try:
        run_backtest()
        print("\n回测完成！查看日志文件获取详细信息。")
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}", exc_info=True)
        print(f"\n回测出错: {str(e)}") 