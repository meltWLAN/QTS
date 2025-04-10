#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计套利策略模型实现
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from technical_indicators import TechnicalIndicator
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 日志配置
logger = logging.getLogger('StatisticalArbitrage')

class PairsTradingStrategy:
    """配对交易统计套利策略"""
    
    def __init__(self, lookback_period=60, entry_threshold=2.0, exit_threshold=0.5):
        """
        初始化配对交易策略
        
        参数:
        lookback_period (int): 用于计算平均和标准差的历史数据长度
        entry_threshold (float): 进场阈值（以标准差倍数表示）
        exit_threshold (float): 出场阈值（以标准差倍数表示）
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.pairs = []
        self.current_positions = {}
        self.trade_history = []
        self.pair_performance = {}
    
    def find_cointegrated_pairs(self, price_df, significance=0.05):
        """
        寻找协整关系的股票对
        
        参数:
        price_df (DataFrame): 包含多只股票价格数据的DataFrame，每一列是一只股票的收盘价
        significance (float): 协整检验的显著性水平
        
        返回:
        list: 协整对列表，每个元素是(stock1, stock2, p_value)的元组
        """
        try:
            n = len(price_df.columns)
            pvalue_matrix = np.ones((n, n))
            keys = price_df.columns
            pairs = []
            
            # 对所有可能的股票对进行协整检验
            for i in range(n):
                for j in range(i+1, n):
                    # 检查数据是否有效
                    stock1 = price_df.iloc[:, i]
                    stock2 = price_df.iloc[:, j]
                    
                    if stock1.isnull().any() or stock2.isnull().any():
                        logger.warning(f"存在缺失数据: {keys[i]} 或 {keys[j]}")
                        continue
                    
                    # 进行Engle-Granger两步法协整检验
                    result = coint(stock1, stock2)
                    pvalue = result[1]
                    pvalue_matrix[i, j] = pvalue
                    
                    if pvalue < significance:
                        # 发现协整对
                        pairs.append((keys[i], keys[j], pvalue))
                        logger.info(f"发现协整对: {keys[i]} - {keys[j]}, p值: {pvalue:.4f}")
            
            # 按照p值升序排序
            return sorted(pairs, key=lambda x: x[2])
        except Exception as e:
            logger.error(f"寻找协整对时出错: {str(e)}")
            return []
    
    def calculate_hedge_ratio(self, stock1_prices, stock2_prices):
        """
        计算对冲比率（使用线性回归）
        
        参数:
        stock1_prices (Series): 第一只股票的价格序列
        stock2_prices (Series): 第二只股票的价格序列
        
        返回:
        float: 对冲比率
        """
        try:
            # 使用OLS回归计算对冲比率
            model = sm.OLS(stock1_prices, sm.add_constant(stock2_prices))
            result = model.fit()
            hedge_ratio = result.params[1]
            return hedge_ratio
        except Exception as e:
            logger.error(f"计算对冲比率时出错: {str(e)}")
            return None
    
    def calculate_spread(self, stock1_prices, stock2_prices, hedge_ratio):
        """
        计算股票对的价差序列
        
        参数:
        stock1_prices (Series): 第一只股票的价格序列
        stock2_prices (Series): 第二只股票的价格序列
        hedge_ratio (float): 对冲比率
        
        返回:
        Series: 价差序列
        """
        try:
            # 计算价差 = stock1 - hedge_ratio * stock2
            spread = stock1_prices - hedge_ratio * stock2_prices
            return spread
        except Exception as e:
            logger.error(f"计算价差时出错: {str(e)}")
            return None
    
    def is_stationary(self, spread, critical_value=0.05):
        """
        检验价差序列是否平稳（使用ADF检验）
        
        参数:
        spread (Series): 价差序列
        critical_value (float): 临界值

        返回:
        bool: 是否平稳
        """
        try:
            # 进行ADF检验
            result = adfuller(spread)
            p_value = result[1]
            return p_value < critical_value
        except Exception as e:
            logger.error(f"ADF检验时出错: {str(e)}")
            return False
    
    def add_pair(self, stock1, stock2):
        """
        添加交易对
        
        参数:
        stock1 (str): 第一只股票代码
        stock2 (str): 第二只股票代码
        """
        pair = (stock1, stock2)
        if pair not in self.pairs:
            self.pairs.append(pair)
            self.current_positions[pair] = {'status': 'neutral', 'units_stock1': 0, 'units_stock2': 0}
            self.pair_performance[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
            logger.info(f"添加交易对: {stock1} - {stock2}")
    
    def generate_signals(self, price_df):
        """
        为所有交易对生成交易信号
        
        参数:
        price_df (DataFrame): 包含股票价格数据的DataFrame
        
        返回:
        dict: 交易信号，格式为 {pair: signal}
        """
        signals = {}
        
        for pair in self.pairs:
            stock1, stock2 = pair
            
            # 检查是否有足够的数据
            if stock1 not in price_df.columns or stock2 not in price_df.columns:
                logger.warning(f"数据缺失: {stock1} 或 {stock2}")
                continue
            
            # 获取价格数据
            stock1_prices = price_df[stock1].values
            stock2_prices = price_df[stock2].values
            
            # 计算对冲比率和价差
            hedge_ratio = self.calculate_hedge_ratio(stock1_prices[-self.lookback_period:], stock2_prices[-self.lookback_period:])
            if hedge_ratio is None:
                continue
                
            spread = self.calculate_spread(stock1_prices, stock2_prices, hedge_ratio)
            if spread is None:
                continue
            
            # 计算价差的均值和标准差
            spread_mean = np.mean(spread[-self.lookback_period:])
            spread_std = np.std(spread[-self.lookback_period:])
            
            # 计算当前z-score
            current_spread = spread[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            # 生成交易信号
            current_position = self.current_positions[pair]['status']
            
            if current_position == 'neutral':
                if z_score > self.entry_threshold:
                    # 价差过高，做空价差（做空stock1，做多stock2）
                    signals[pair] = {'action': 'entry', 'type': 'short_spread', 'z_score': z_score, 'hedge_ratio': hedge_ratio}
                    logger.info(f"信号: 做空价差 {stock1}-{stock2}, z-score: {z_score:.2f}")
                elif z_score < -self.entry_threshold:
                    # 价差过低，做多价差（做多stock1，做空stock2）
                    signals[pair] = {'action': 'entry', 'type': 'long_spread', 'z_score': z_score, 'hedge_ratio': hedge_ratio}
                    logger.info(f"信号: 做多价差 {stock1}-{stock2}, z-score: {z_score:.2f}")
            
            elif current_position == 'long_spread':
                if z_score > -self.exit_threshold:
                    # 价差回归，平仓
                    signals[pair] = {'action': 'exit', 'type': 'long_spread', 'z_score': z_score, 'hedge_ratio': hedge_ratio}
                    logger.info(f"信号: 平仓多价差 {stock1}-{stock2}, z-score: {z_score:.2f}")
            
            elif current_position == 'short_spread':
                if z_score < self.exit_threshold:
                    # 价差回归，平仓
                    signals[pair] = {'action': 'exit', 'type': 'short_spread', 'z_score': z_score, 'hedge_ratio': hedge_ratio}
                    logger.info(f"信号: 平仓空价差 {stock1}-{stock2}, z-score: {z_score:.2f}")
        
        return signals
    
    def execute_trades(self, signals, price_df, capital_per_pair=100000):
        """
        执行交易信号
        
        参数:
        signals (dict): 交易信号
        price_df (DataFrame): 价格数据
        capital_per_pair (float): 每对分配的资金
        """
        current_prices = price_df.iloc[-1]
        current_time = datetime.now()
        
        for pair, signal in signals.items():
            stock1, stock2 = pair
            action = signal['action']
            signal_type = signal['type']
            hedge_ratio = signal['hedge_ratio']
            
            # 获取当前价格
            stock1_price = current_prices[stock1]
            stock2_price = current_prices[stock2]
            
            if action == 'entry':
                # 计算每只股票的头寸规模
                # 假设我们用一半资金买stock1，一半资金按对冲比例买stock2
                capital_for_stock1 = capital_per_pair / 2
                units_stock1 = int(capital_for_stock1 / stock1_price)
                units_stock2 = int((capital_per_pair - (units_stock1 * stock1_price)) / stock2_price)
                
                # 根据信号类型调整头寸方向
                if signal_type == 'long_spread':
                    # 做多价差 = 做多stock1，做空stock2
                    units_stock2 = -int(abs(units_stock1 * hedge_ratio))
                else:  # short_spread
                    # 做空价差 = 做空stock1，做多stock2
                    units_stock1 = -units_stock1
                    units_stock2 = int(abs(units_stock1 * hedge_ratio))
                
                # 记录交易
                trade = {
                    'timestamp': current_time,
                    'pair': pair,
                    'action': 'entry',
                    'type': signal_type,
                    'stock1': stock1,
                    'stock1_units': units_stock1,
                    'stock1_price': stock1_price,
                    'stock2': stock2,
                    'stock2_units': units_stock2,
                    'stock2_price': stock2_price,
                    'z_score': signal['z_score'],
                    'hedge_ratio': hedge_ratio
                }
                
                # 更新当前头寸
                self.current_positions[pair] = {
                    'status': signal_type,
                    'entry_time': current_time,
                    'units_stock1': units_stock1,
                    'stock1_entry_price': stock1_price,
                    'units_stock2': units_stock2,
                    'stock2_entry_price': stock2_price,
                    'entry_z_score': signal['z_score'],
                    'hedge_ratio': hedge_ratio
                }
                
                logger.info(f"执行交易: {action} {signal_type} {stock1}({units_stock1}@{stock1_price}) - {stock2}({units_stock2}@{stock2_price})")
                self.trade_history.append(trade)
                
            elif action == 'exit':
                # 检查是否有持仓
                position = self.current_positions[pair]
                if position['status'] == 'neutral':
                    logger.warning(f"收到平仓信号但无持仓: {pair}")
                    continue
                
                # 平仓计算
                units_stock1 = position['units_stock1']
                units_stock2 = position['units_stock2']
                entry_stock1_price = position['stock1_entry_price']
                entry_stock2_price = position['stock2_entry_price']
                
                # 计算盈亏
                stock1_pnl = units_stock1 * (stock1_price - entry_stock1_price)
                stock2_pnl = units_stock2 * (stock2_price - entry_stock2_price)
                total_pnl = stock1_pnl + stock2_pnl
                
                # 记录交易
                trade = {
                    'timestamp': current_time,
                    'pair': pair,
                    'action': 'exit',
                    'type': position['status'],
                    'stock1': stock1,
                    'stock1_units': -units_stock1,  # 平仓
                    'stock1_price': stock1_price,
                    'stock2': stock2,
                    'stock2_units': -units_stock2,  # 平仓
                    'stock2_price': stock2_price,
                    'pnl': total_pnl,
                    'z_score': signal['z_score'],
                    'days_held': (current_time - position['entry_time']).days
                }
                
                # 更新绩效统计
                self.pair_performance[pair]['trades'] += 1
                if total_pnl > 0:
                    self.pair_performance[pair]['wins'] += 1
                else:
                    self.pair_performance[pair]['losses'] += 1
                self.pair_performance[pair]['total_pnl'] += total_pnl
                
                # 重置头寸
                self.current_positions[pair] = {'status': 'neutral', 'units_stock1': 0, 'units_stock2': 0}
                
                logger.info(f"执行交易: {action} {position['status']} {stock1}({-units_stock1}@{stock1_price}) - {stock2}({-units_stock2}@{stock2_price}), PnL: {total_pnl:.2f}")
                self.trade_history.append(trade)
    
    def backtest(self, price_df, start_date=None, end_date=None, capital_per_pair=100000):
        """
        回测策略
        
        参数:
        price_df (DataFrame): 价格数据，索引为日期
        start_date (str): 回测开始日期，格式 'YYYY-MM-DD'
        end_date (str): 回测结束日期，格式 'YYYY-MM-DD'
        capital_per_pair (float): 每对分配的资金
        
        返回:
        DataFrame: 回测结果
        """
        # 设置回测日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            price_df = price_df[price_df.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            price_df = price_df[price_df.index <= end_date]
        
        # 初始化回测
        self.trade_history = []
        for pair in self.pairs:
            self.current_positions[pair] = {'status': 'neutral', 'units_stock1': 0, 'units_stock2': 0}
            self.pair_performance[pair] = {'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0}
        
        # 每日运行回测
        for i in range(self.lookback_period, len(price_df)):
            date = price_df.index[i]
            historical_data = price_df.iloc[:i+1]
            
            # 生成信号
            signals = self.generate_signals(historical_data)
            
            # 执行交易
            if signals:
                self.execute_trades(signals, historical_data, capital_per_pair)
        
        # 平掉所有剩余头寸
        final_signals = {}
        for pair, position in self.current_positions.items():
            if position['status'] != 'neutral':
                final_signals[pair] = {'action': 'exit', 'type': position['status'], 'z_score': 0, 'hedge_ratio': position['hedge_ratio']}
        
        if final_signals:
            self.execute_trades(final_signals, price_df)
        
        # 生成回测报告
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            
            # 计算累积盈亏和绩效指标
            if 'pnl' in trades_df.columns:
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                # 计算胜率
                win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df[trades_df['action'] == 'exit']) if len(trades_df[trades_df['action'] == 'exit']) > 0 else 0
                
                # 计算盈亏比
                avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
                avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
                profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
                
                # 打印绩效概况
                total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
                logger.info(f"回测完成. 总PnL: {total_pnl:.2f}, 胜率: {win_rate:.2f}, 盈亏比: {profit_loss_ratio:.2f}")
                
                for pair, performance in self.pair_performance.items():
                    trades = performance['trades']
                    if trades > 0:
                        win_rate = performance['wins'] / trades
                        logger.info(f"交易对 {pair}: 交易次数 {trades}, 胜率 {win_rate:.2f}, 总PnL {performance['total_pnl']:.2f}")
                
                return trades_df
            else:
                logger.warning("回测期间没有平仓交易")
                return pd.DataFrame()
        else:
            logger.warning("回测期间没有产生交易")
            return pd.DataFrame()
    
    def plot_pair_spread(self, stock1_prices, stock2_prices, hedge_ratio, lookback_period=None):
        """
        绘制股票对价差走势图
        
        参数:
        stock1_prices (Series): 第一只股票的价格序列
        stock2_prices (Series): 第二只股票的价格序列
        hedge_ratio (float): 对冲比率
        lookback_period (int): 回看周期，默认使用策略的lookback_period
        """
        if lookback_period is None:
            lookback_period = self.lookback_period
        
        # 计算价差
        spread = self.calculate_spread(stock1_prices, stock2_prices, hedge_ratio)
        
        # 计算移动均值和标准差
        spread_mean = spread.rolling(window=lookback_period).mean()
        spread_std = spread.rolling(window=lookback_period).std()
        
        # 计算z-score
        z_score = (spread - spread_mean) / spread_std
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # 绘制股票价格
        ax1.plot(stock1_prices.index, stock1_prices, label='Stock 1')
        ax1.plot(stock2_prices.index, stock2_prices * hedge_ratio, label='Stock 2 (Adjusted)')
        ax1.set_title('Stock Prices')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制价差
        ax2.plot(spread.index, spread, label='Spread')
        ax2.plot(spread_mean.index, spread_mean, label='Mean', linestyle='--')
        ax2.plot(spread_mean.index, spread_mean + spread_std * self.entry_threshold, label=f'+{self.entry_threshold} Std', linestyle=':')
        ax2.plot(spread_mean.index, spread_mean - spread_std * self.entry_threshold, label=f'-{self.entry_threshold} Std', linestyle=':')
        ax2.set_title('Price Spread')
        ax2.legend()
        ax2.grid(True)
        
        # 绘制z-score
        ax3.plot(z_score.index, z_score, label='Z-Score')
        ax3.axhline(y=self.entry_threshold, color='r', linestyle='--', label=f'Entry (+{self.entry_threshold})')
        ax3.axhline(y=-self.entry_threshold, color='r', linestyle='--', label=f'Entry (-{self.entry_threshold})')
        ax3.axhline(y=self.exit_threshold, color='g', linestyle='--', label=f'Exit (+{self.exit_threshold})')
        ax3.axhline(y=-self.exit_threshold, color='g', linestyle='--', label=f'Exit (-{self.exit_threshold})')
        ax3.axhline(y=0, color='k', linestyle='-')
        ax3.set_title('Z-Score')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_backtest_results(self, trades_df):
        """
        绘制回测结果
        
        参数:
        trades_df (DataFrame): 交易历史DataFrame
        """
        if trades_df.empty or 'pnl' not in trades_df.columns:
            logger.warning("没有足够的交易数据用于绘图")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 绘制累积盈亏曲线
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        exit_trades = trades_df[trades_df['action'] == 'exit']
        ax1.plot(exit_trades['timestamp'], exit_trades['cumulative_pnl'])
        ax1.set_title('Cumulative PnL')
        ax1.grid(True)
        
        # 绘制每笔交易盈亏
        ax2.bar(exit_trades['timestamp'], exit_trades['pnl'])
        ax2.set_title('Individual Trade PnL')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    # 在实际应用中，你需要从数据源加载真实股票数据
    date_range = pd.date_range(start='2022-01-01', periods=252, freq='B')
    
    # 模拟两支股票价格，它们具有协整关系
    np.random.seed(42)
    stock1 = np.random.normal(0, 1, 252).cumsum() + 100
    stock2 = stock1 * 0.7 + np.random.normal(0, 2, 252).cumsum() + 50
    
    # 创建DataFrame
    price_df = pd.DataFrame({
        'Stock_A': stock1,
        'Stock_B': stock2
    }, index=date_range)
    
    # 创建策略实例
    strategy = PairsTradingStrategy(lookback_period=50, entry_threshold=2.0, exit_threshold=0.5)
    
    # 添加交易对
    strategy.add_pair('Stock_A', 'Stock_B')
    
    # 回测策略
    trades_df = strategy.backtest(price_df, capital_per_pair=100000)
    
    # 绘制结果
    if not trades_df.empty:
        strategy.plot_backtest_results(trades_df)
        
    # 可视化价差
    hedge_ratio = strategy.calculate_hedge_ratio(price_df['Stock_A'], price_df['Stock_B'])
    strategy.plot_pair_spread(price_df['Stock_A'], price_df['Stock_B'], hedge_ratio) 