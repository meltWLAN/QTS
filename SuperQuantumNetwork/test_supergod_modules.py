#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量化系统模块测试脚本
测试各个核心模块的功能并修复发现的问题
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_modules_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodTest")

def test_data_connector():
    """测试数据连接器"""
    logger.info("测试数据连接器")
    
    # 首先尝试Tushare
    try:
        from tushare_data_connector import TushareDataConnector
        
        # 创建数据源实例
        ts_source = TushareDataConnector()
        
        # 获取股票数据
        df = ts_source.get_market_data("000001.SH", 
                                      start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is not None and not df.empty:
            logger.info(f"成功获取Tushare股票数据，共{len(df)}条记录")
            logger.info(f"数据预览:\n{df.head()}")
            
            # 获取板块数据
            sector_data = ts_source.get_sector_data()
            if sector_data is not None:
                logger.info(f"成功获取Tushare板块数据，领先板块: {len(sector_data['leading_sectors'])}")
                
            logger.info("✅ Tushare数据连接器测试通过")
            return True
            
    except Exception as e:
        logger.warning(f"Tushare数据连接测试失败: {str(e)}")
    
    # 如果Tushare失败，尝试AKShare
    try:
        from akshare_data_connector import AKShareDataConnector
        
        # 创建数据源实例
        ak_source = AKShareDataConnector()
        
        # 获取股票数据
        df = ak_source.get_market_data("000001.SH", 
                                      start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is not None and not df.empty:
            logger.info(f"成功获取AKShare股票数据，共{len(df)}条记录")
            logger.info(f"数据预览:\n{df.head()}")
            
            # 获取板块数据
            sector_data = ak_source.get_sector_data()
            if sector_data is not None:
                logger.info(f"成功获取AKShare板块数据，领先板块: {len(sector_data['leading_sectors'])}")
                
            logger.info("✅ AKShare数据连接器测试通过")
            return True
            
    except Exception as e:
        logger.error(f"AKShare数据连接测试失败: {str(e)}")
        logger.error("❌ 数据连接器测试失败 - 请确保安装了tushare或akshare库")
        return False

def test_technical_indicators():
    """测试技术指标计算模块"""
    logger.info("测试技术指标计算模块")
    try:
        from technical_indicators import TechnicalIndicator
        
        # 获取测试数据
        # 首先尝试Tushare
        try:
            from tushare_data_connector import TushareDataConnector
            data_source = TushareDataConnector()
        except Exception:
            # 如果Tushare失败，尝试AKShare
            try:
                from akshare_data_connector import AKShareDataConnector
                data_source = AKShareDataConnector()
            except Exception as e:
                logger.error(f"无法加载数据源: {str(e)}")
                return False
        
        # 获取上证指数数据
        df = data_source.get_market_data("000001.SH", 
                                      start_date=(datetime.now() - timedelta(days=90)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is None or df.empty:
            logger.error("获取测试数据失败")
            return False
        
        # 计算各种技术指标
        # 移动平均线
        ma5 = TechnicalIndicator.MA(df, period=5)
        if ma5 is None:
            logger.error("计算MA5失败")
            return False
        
        # 指数移动平均线
        ema12 = TechnicalIndicator.EMA(df, period=12)
        if ema12 is None:
            logger.error("计算EMA12失败")
            return False
        
        # MACD
        macd = TechnicalIndicator.MACD(df)
        if macd is None:
            logger.error("计算MACD失败")
            return False
        
        # RSI
        rsi = TechnicalIndicator.RSI(df)
        if rsi is None:
            logger.error("计算RSI失败")
            return False
        
        # 布林带
        boll = TechnicalIndicator.BOLL(df)
        if boll is None:
            logger.error("计算布林带失败")
            return False
        
        # KDJ
        kdj = TechnicalIndicator.KDJ(df)
        if kdj is None:
            logger.error("计算KDJ失败")
            return False
        
        logger.info("成功计算各项技术指标")
        logger.info(f"MA5最新值: {ma5.iloc[-1]:.2f}")
        logger.info(f"EMA12最新值: {ema12.iloc[-1]:.2f}")
        logger.info(f"MACD最新值: {macd['macd'].iloc[-1]:.2f}, 信号线: {macd['signal'].iloc[-1]:.2f}")
        logger.info(f"RSI最新值: {rsi.iloc[-1]:.2f}")
        logger.info(f"布林带上轨: {boll['upper'].iloc[-1]:.2f}, 中轨: {boll['middle'].iloc[-1]:.2f}, 下轨: {boll['lower'].iloc[-1]:.2f}")
        logger.info(f"KDJ指标: K={kdj['K'].iloc[-1]:.2f}, D={kdj['D'].iloc[-1]:.2f}, J={kdj['J'].iloc[-1]:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"测试技术指标计算模块时出错: {str(e)}", exc_info=True)
        return False

def test_market_visualizer():
    """测试市场可视化模块"""
    logger.info("测试市场可视化模块")
    try:
        from market_visualizer import MarketVisualizer
        
        # 获取测试数据
        # 首先尝试Tushare
        try:
            from tushare_data_connector import TushareDataConnector
            data_source = TushareDataConnector()
        except Exception:
            # 如果Tushare失败，尝试AKShare
            try:
                from akshare_data_connector import AKShareDataConnector
                data_source = AKShareDataConnector()
            except Exception as e:
                logger.error(f"无法加载数据源: {str(e)}")
                return False
        
        # 获取股票数据
        df = data_source.get_market_data("000001.SZ", 
                                      start_date=(datetime.now() - timedelta(days=60)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is None or df.empty:
            logger.error("获取测试数据失败")
            return False
        
        # 创建可视化器
        visualizer = MarketVisualizer()
        
        # 计算技术指标
        from technical_indicators import TechnicalIndicator
        macd = TechnicalIndicator.MACD(df)
        rsi = TechnicalIndicator.RSI(df)
        boll = TechnicalIndicator.BOLL(df)
        
        # 准备指标字典
        indicators = {
            'MACD': macd,
            'RSI': rsi,
            'Bollinger': boll
        }
        
        # 绘制K线图
        try:
            visualizer.plot_candlestick(df, title='平安银行K线图')
            logger.info("成功绘制K线图")
        except Exception as e:
            logger.error(f"绘制K线图失败: {str(e)}")
            return False
        
        # 绘制技术指标
        try:
            visualizer.plot_technical_indicators(df, indicators, title='平安银行技术指标')
            logger.info("成功绘制技术指标图")
        except Exception as e:
            logger.error(f"绘制技术指标图失败: {str(e)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"测试市场可视化模块时出错: {str(e)}", exc_info=True)
        return False

def test_trading_signals():
    """测试交易信号生成模块"""
    logger.info("测试交易信号生成模块")
    try:
        from trading_signals import MACrossSignalGenerator, RSISignalGenerator, SignalManager
        
        # 获取测试数据
        # 首先尝试Tushare
        try:
            from tushare_data_connector import TushareDataConnector
            data_source = TushareDataConnector()
        except Exception:
            # 如果Tushare失败，尝试AKShare
            try:
                from akshare_data_connector import AKShareDataConnector
                data_source = AKShareDataConnector()
            except Exception as e:
                logger.error(f"无法加载数据源: {str(e)}")
                return False
        
        # 获取股票数据
        df = data_source.get_market_data("000001.SZ", 
                                      start_date=(datetime.now() - timedelta(days=90)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is None or df.empty:
            logger.error("获取测试数据失败")
            return False
        
        # 创建信号生成器
        ma_cross = MACrossSignalGenerator(fast_period=5, slow_period=20)
        rsi_signal = RSISignalGenerator(period=14, overbought=70, oversold=30)
        
        # 创建信号管理器
        signal_manager = SignalManager()
        signal_manager.add_generator(ma_cross)
        signal_manager.add_generator(rsi_signal)
        
        # 生成信号
        ma_signals = ma_cross.generate(df)
        if ma_signals is None:
            logger.error("MA交叉信号生成失败")
            return False
            
        rsi_signals = rsi_signal.generate(df)
        if rsi_signals is None:
            logger.error("RSI信号生成失败")
            return False
        
        # 使用信号管理器生成综合信号
        combined_signals = signal_manager.generate_signals(df)
        if combined_signals is None:
            logger.error("综合信号生成失败")
            return False
        
        logger.info(f"MA交叉信号数量: {len(ma_signals)}")
        logger.info(f"RSI信号数量: {len(rsi_signals)}")
        logger.info(f"综合信号数量: {len(combined_signals)}")
        
        return True
    except Exception as e:
        logger.error(f"测试交易信号生成模块时出错: {str(e)}", exc_info=True)
        return False

def test_backtest_engine():
    """测试回测引擎"""
    logger.info("测试回测引擎...")
    
    try:
        from backtest_engine import BacktestEngine, Strategy, Portfolio, EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, PortfolioEvent, OrderDirection
        
        # 实现一个简单的策略类
        class SimpleMAStrategy(Strategy):
            def __init__(self, fast_period=5, slow_period=20):
                super().__init__()
                self.fast_period = fast_period
                self.slow_period = slow_period
            
            def calculate_signals(self, event):
                if event.type == EventType.MARKET_DATA:
                    # 获取市场数据
                    data = event.data
                    if data is None or len(data) < self.slow_period:
                        return []
                        
                    # 计算快速和慢速移动平均线
                    data_df = data.copy()
                    data_df['ma_fast'] = data_df['close'].rolling(window=self.fast_period).mean()
                    data_df['ma_slow'] = data_df['close'].rolling(window=self.slow_period).mean()
                    
                    # 生成信号
                    signals = []
                    if len(data_df) >= self.slow_period + 1:
                        # 获取倒数第二行，因为最后一行是最新数据
                        current = data_df.iloc[-1]
                        prev = data_df.iloc[-2]
                        
                        # 金叉信号（快线从下方穿过慢线）
                        if prev['ma_fast'] <= prev['ma_slow'] and current['ma_fast'] > current['ma_slow']:
                            signals.append(
                                SignalEvent(
                                    type=EventType.SIGNAL,
                                    timestamp=current.name,
                                    symbol=event.symbol,
                                    direction=OrderDirection.BUY,
                                    strength=1.0
                                )
                            )
                        # 死叉信号（快线从上方穿过慢线）
                        elif prev['ma_fast'] >= prev['ma_slow'] and current['ma_fast'] < current['ma_slow']:
                            signals.append(
                                SignalEvent(
                                    type=EventType.SIGNAL,
                                    timestamp=current.name,
                                    symbol=event.symbol,
                                    direction=OrderDirection.SELL,
                                    strength=1.0
                                )
                            )
                    
                    return signals
                return []
        
        # 实现一个简单的投资组合类
        class SimplePortfolio(Portfolio):
            def __init__(self, initial_capital=100000.0):
                super().__init__(initial_capital)
                self.current_positions = {}  # 当前持仓
                self.current_holdings = {
                    'cash': initial_capital,
                    'total': initial_capital
                }
            
            def update_on_signal(self, event):
                if event.type == EventType.SIGNAL:
                    # 简单订单：信号买入/卖出100股
                    symbol = event.symbol
                    quantity = 100
                    
                    if event.direction == OrderDirection.BUY:
                        order = OrderEvent(
                            type=EventType.ORDER,
                            timestamp=event.timestamp,
                            symbol=symbol,
                            order_type="MARKET",
                            direction=OrderDirection.BUY,
                            quantity=quantity
                        )
                        return [order]
                    elif event.direction == OrderDirection.SELL:
                        # 检查是否有足够的持仓
                        current_position = self.current_positions.get(symbol, 0)
                        if current_position >= quantity:
                            order = OrderEvent(
                                type=EventType.ORDER,
                                timestamp=event.timestamp,
                                symbol=symbol,
                                order_type="MARKET",
                                direction=OrderDirection.SELL,
                                quantity=quantity
                            )
                            return [order]
                
                return []
            
            def update_on_fill(self, event):
                if event.type == EventType.FILL:
                    # 更新持仓和资金
                    symbol = event.symbol
                    quantity = event.quantity
                    price = event.price
                    cost = price * quantity
                    
                    # 更新持仓
                    if event.direction == OrderDirection.BUY:
                        self.current_positions[symbol] = self.current_positions.get(symbol, 0) + quantity
                        self.current_holdings['cash'] -= cost
                    elif event.direction == OrderDirection.SELL:
                        self.current_positions[symbol] = self.current_positions.get(symbol, 0) - quantity
                        self.current_holdings['cash'] += cost
                    
                    # 更新投资组合价值
                    return self.update_portfolio_value()
                
                return []
            
            def update_portfolio_value(self):
                # 计算当前组合价值
                total_value = self.current_holdings['cash']
                
                # 仅简单示例，实际应该基于最新市场价格
                for symbol, quantity in self.current_positions.items():
                    # 假设价格为100
                    price = 100.0
                    total_value += price * quantity
                
                self.current_holdings['total'] = total_value
                
                # 创建投资组合事件
                portfolio_event = PortfolioEvent(
                    type=EventType.PORTFOLIO,
                    timestamp=datetime.now(),
                    portfolio_value=self.current_holdings['total'],
                    cash=self.current_holdings['cash'],
                    holdings=self.current_positions.copy()
                )
                
                return [portfolio_event]
        
        # 获取测试数据
        # 首先尝试Tushare
        try:
            from tushare_data_connector import TushareDataConnector
            data_source = TushareDataConnector()
        except Exception:
            # 如果Tushare失败，尝试AKShare
            try:
                from akshare_data_connector import AKShareDataConnector
                data_source = AKShareDataConnector()
            except Exception as e:
                logger.error(f"无法加载数据源: {str(e)}")
                return False
        
        # 获取股票数据
        df = data_source.get_market_data("000001.SZ", 
                                      start_date=(datetime.now() - timedelta(days=180)).strftime('%Y%m%d'),
                                      end_date=datetime.now().strftime('%Y%m%d'))
        
        if df is None or df.empty:
            logger.error("获取测试数据失败")
            return False
        
        # 创建回测引擎
        backtest = BacktestEngine(
            start_date=datetime.now() - timedelta(days=180),
            end_date=datetime.now(),
            symbols=["000001.SZ"],
            initial_capital=100000.0
        )
        
        # 添加策略和投资组合
        backtest.set_strategy(SimpleMAStrategy(fast_period=5, slow_period=20))
        backtest.set_portfolio(SimplePortfolio(initial_capital=100000.0))
        
        # 运行回测
        results = backtest.run(data_source=data_source)
        
        # 处理回测结果
        logger.info("回测完成!")
        logger.info(f"初始资金: {backtest.initial_capital:.2f}")
        
        # 检查结果包含指标
        if 'metrics' in results:
            metrics = results['metrics']
            logger.info("可用的度量标准:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info("可用的度量标准:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        
        # 绘制回测结果
        try:
            backtest.plot_results()
            logger.info("成功绘制回测结果图表")
        except Exception as e:
            logger.error(f"绘制回测结果图表时出错: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"测试回测引擎时出错: {str(e)}", exc_info=True)
        return False

def main():
    """主函数"""
    logger.info("开始测试超神系统模块...")
    
    # 测试数据连接器
    if test_data_connector():
        logger.info("✅ 数据连接器测试通过")
    else:
        logger.error("❌ 数据连接器测试失败")
    
    # 测试技术指标计算模块
    if test_technical_indicators():
        logger.info("✅ 技术指标计算模块测试通过")
    else:
        logger.error("❌ 技术指标计算模块测试失败")
    
    # 测试市场可视化模块
    if test_market_visualizer():
        logger.info("✅ 市场可视化模块测试通过")
    else:
        logger.error("❌ 市场可视化模块测试失败")
    
    # 测试交易信号生成模块
    if test_trading_signals():
        logger.info("✅ 交易信号生成模块测试通过")
    else:
        logger.error("❌ 交易信号生成模块测试失败")
    
    # 测试回测引擎
    if test_backtest_engine():
        logger.info("✅ 回测引擎测试通过")
    else:
        logger.error("❌ 回测引擎测试失败")
    
    logger.info("超神系统模块测试完成")

if __name__ == "__main__":
    main() 