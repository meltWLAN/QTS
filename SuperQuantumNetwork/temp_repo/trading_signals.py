#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交易信号生成模块 - 处理各种类型的交易信号生成
"""

import numpy as np
import pandas as pd
from technical_indicators import TechnicalIndicator
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 日志配置
logger = logging.getLogger('TradingSignals')

class SignalGenerator:
    """交易信号生成器基类"""
    
    def __init__(self, name):
        """
        初始化信号生成器
        
        参数:
        name (str): 信号生成器名称
        """
        self.name = name
        self.signals = []
        
    def generate(self, data):
        """
        生成信号的抽象方法
        
        参数:
        data (DataFrame): 市场数据
        
        返回:
        dict: 信号字典
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def add_signal(self, timestamp, symbol, signal_type, direction, strength=1.0, **kwargs):
        """
        添加一个信号
        
        参数:
        timestamp (datetime): 信号时间戳
        symbol (str): 标的代码
        signal_type (str): 信号类型
        direction (str): 方向，'buy'或'sell'
        strength (float): 信号强度，0-1
        **kwargs: 附加信息
        """
        signal = {
            'timestamp': timestamp,
            'symbol': symbol,
            'type': signal_type,
            'direction': direction,
            'strength': strength,
            'generator': self.name
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            signal[key] = value
            
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_time=None, end_time=None, symbols=None):
        """
        获取指定条件的信号
        
        参数:
        start_time (datetime): 开始时间
        end_time (datetime): 结束时间
        symbols (list): 标的代码列表
        
        返回:
        list: 符合条件的信号列表
        """
        filtered_signals = self.signals
        
        if start_time:
            filtered_signals = [s for s in filtered_signals if s['timestamp'] >= start_time]
        if end_time:
            filtered_signals = [s for s in filtered_signals if s['timestamp'] <= end_time]
        if symbols:
            filtered_signals = [s for s in filtered_signals if s['symbol'] in symbols]
            
        return filtered_signals
    
    def clear_signals(self):
        """清除所有信号"""
        self.signals = []


class MACrossSignalGenerator(SignalGenerator):
    """移动平均线交叉信号生成器"""
    
    def __init__(self, fast_period=5, slow_period=20):
        """
        初始化MA交叉信号生成器
        
        参数:
        fast_period (int): 快线周期
        slow_period (int): 慢线周期
        """
        super().__init__(f"MA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate(self, data):
        """
        生成MA交叉信号
        
        参数:
        data (DataFrame): 价格数据，包含'close'列和日期索引
        
        返回:
        dict: 信号字典
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return {}
                
            if len(data) < self.slow_period:
                logger.warning(f"数据长度({len(data)})小于慢线周期({self.slow_period})")
                return {}
                
            # 计算快线和慢线
            fast_ma = TechnicalIndicator.MA(data, period=self.fast_period)
            slow_ma = TechnicalIndicator.MA(data, period=self.slow_period)
            
            # 获取交叉信号
            crossover = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)
            crossunder = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)
            
            # 生成信号
            signal_dict = {}
            
            for i in range(1, len(data)):
                timestamp = data.index[i]
                symbol = data.get('symbol', [''])[0] if 'symbol' in data.columns else ''
                
                if crossover.iloc[i]:
                    # 生成多头信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MA_Cross',
                        direction='buy',
                        fast_ma=fast_ma.iloc[i],
                        slow_ma=slow_ma.iloc[i],
                        strength=abs(fast_ma.iloc[i] - slow_ma.iloc[i]) / data['close'].iloc[i]
                    )
                    signal_dict[timestamp] = signal
                    
                elif crossunder.iloc[i]:
                    # 生成空头信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MA_Cross',
                        direction='sell',
                        fast_ma=fast_ma.iloc[i],
                        slow_ma=slow_ma.iloc[i],
                        strength=abs(fast_ma.iloc[i] - slow_ma.iloc[i]) / data['close'].iloc[i]
                    )
                    signal_dict[timestamp] = signal
                    
            return signal_dict
        except Exception as e:
            logger.error(f"生成MA交叉信号时发生错误: {str(e)}")
            return {}


class RSISignalGenerator(SignalGenerator):
    """RSI指标信号生成器"""
    
    def __init__(self, period=14, overbought=70, oversold=30):
        """
        初始化RSI信号生成器
        
        参数:
        period (int): RSI周期
        overbought (float): 超买阈值
        oversold (float): 超卖阈值
        """
        super().__init__(f"RSI_{period}")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def generate(self, data):
        """
        生成RSI信号
        
        参数:
        data (DataFrame): 价格数据，包含'close'列和日期索引
        
        返回:
        dict: 信号字典
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return {}
                
            if len(data) < self.period + 1:
                logger.warning(f"数据长度({len(data)})不足")
                return {}
                
            # 计算RSI
            rsi = TechnicalIndicator.RSI(data, period=self.period)
            
            # 生成信号
            signal_dict = {}
            
            for i in range(1, len(data)):
                if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i-1]):
                    continue
                    
                timestamp = data.index[i]
                symbol = data.get('symbol', [''])[0] if 'symbol' in data.columns else ''
                
                # 从超买区域回落
                if rsi.iloc[i-1] > self.overbought and rsi.iloc[i] < self.overbought:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='RSI_Overbought',
                        direction='sell',
                        rsi=rsi.iloc[i],
                        strength=(rsi.iloc[i-1] - self.overbought) / (100 - self.overbought)
                    )
                    signal_dict[timestamp] = signal
                    
                # 从超卖区域反弹
                elif rsi.iloc[i-1] < self.oversold and rsi.iloc[i] > self.oversold:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='RSI_Oversold',
                        direction='buy',
                        rsi=rsi.iloc[i],
                        strength=(self.oversold - rsi.iloc[i-1]) / self.oversold
                    )
                    signal_dict[timestamp] = signal
                    
            return signal_dict
        except Exception as e:
            logger.error(f"生成RSI信号时发生错误: {str(e)}")
            return {}


class MACDSignalGenerator(SignalGenerator):
    """MACD指标信号生成器"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        初始化MACD信号生成器
        
        参数:
        fast_period (int): 快线EMA周期
        slow_period (int): 慢线EMA周期
        signal_period (int): 信号线周期
        """
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def generate(self, data):
        """
        生成MACD信号
        
        参数:
        data (DataFrame): 价格数据，包含'close'列和日期索引
        
        返回:
        dict: 信号字典
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return {}
                
            if len(data) < self.slow_period + self.signal_period:
                logger.warning(f"数据长度({len(data)})不足")
                return {}
                
            # 计算MACD
            macd_result = TechnicalIndicator.MACD(
                data, 
                fast_period=self.fast_period, 
                slow_period=self.slow_period, 
                signal_period=self.signal_period
            )
            
            if macd_result is None:
                logger.error("MACD计算失败")
                return {}
                
            macd_line = macd_result['macd']
            signal_line = macd_result['signal']
            histogram = macd_result['histogram']
            
            # 生成信号
            signal_dict = {}
            
            for i in range(1, len(data)):
                if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]) or pd.isna(histogram.iloc[i]):
                    continue
                    
                timestamp = data.index[i]
                symbol = data.get('symbol', [''])[0] if 'symbol' in data.columns else ''
                
                # MACD线上穿信号线
                if macd_line.iloc[i-1] < signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                    # 生成多头信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MACD_Cross',
                        direction='buy',
                        macd=macd_line.iloc[i],
                        signal=signal_line.iloc[i],
                        histogram=histogram.iloc[i],
                        strength=abs(macd_line.iloc[i] - signal_line.iloc[i]) / data['close'].iloc[i] * 100
                    )
                    signal_dict[timestamp] = signal
                    
                # MACD线下穿信号线
                elif macd_line.iloc[i-1] > signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                    # 生成空头信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MACD_Cross',
                        direction='sell',
                        macd=macd_line.iloc[i],
                        signal=signal_line.iloc[i],
                        histogram=histogram.iloc[i],
                        strength=abs(macd_line.iloc[i] - signal_line.iloc[i]) / data['close'].iloc[i] * 100
                    )
                    signal_dict[timestamp] = signal
                    
                # 零轴上穿
                elif macd_line.iloc[i-1] < 0 and macd_line.iloc[i] > 0:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MACD_ZeroCross',
                        direction='buy',
                        macd=macd_line.iloc[i],
                        signal=signal_line.iloc[i],
                        histogram=histogram.iloc[i],
                        strength=abs(macd_line.iloc[i]) / data['close'].iloc[i] * 100
                    )
                    signal_dict[timestamp] = signal
                    
                # 零轴下穿
                elif macd_line.iloc[i-1] > 0 and macd_line.iloc[i] < 0:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='MACD_ZeroCross',
                        direction='sell',
                        macd=macd_line.iloc[i],
                        signal=signal_line.iloc[i],
                        histogram=histogram.iloc[i],
                        strength=abs(macd_line.iloc[i]) / data['close'].iloc[i] * 100
                    )
                    signal_dict[timestamp] = signal
                    
            return signal_dict
        except Exception as e:
            logger.error(f"生成MACD信号时发生错误: {str(e)}")
            return {}


class BollingerBandsSignalGenerator(SignalGenerator):
    """布林带信号生成器"""
    
    def __init__(self, period=20, std_dev=2):
        """
        初始化布林带信号生成器
        
        参数:
        period (int): 周期
        std_dev (float): 标准差倍数
        """
        super().__init__(f"BollingerBands_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
        
    def generate(self, data):
        """
        生成布林带信号
        
        参数:
        data (DataFrame): 价格数据，包含'close'列和日期索引
        
        返回:
        dict: 信号字典
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return {}
                
            if len(data) < self.period:
                logger.warning(f"数据长度({len(data)})小于周期({self.period})")
                return {}
                
            # 计算布林带
            bb = TechnicalIndicator.BOLL(data, period=self.period, std_dev=self.std_dev)
            
            if bb is None:
                logger.error("布林带计算失败")
                return {}
                
            upper = bb['upper']
            middle = bb['middle']
            lower = bb['lower']
            
            # 生成信号
            signal_dict = {}
            
            for i in range(1, len(data)):
                if pd.isna(upper.iloc[i]) or pd.isna(middle.iloc[i]) or pd.isna(lower.iloc[i]):
                    continue
                    
                timestamp = data.index[i]
                symbol = data.get('symbol', [''])[0] if 'symbol' in data.columns else ''
                close = data['close'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                
                # 突破上轨
                if prev_close < upper.iloc[i-1] and close > upper.iloc[i]:
                    # 生成突破上轨信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='BB_UpperBreakout',
                        direction='sell',  # 上轨突破后通常反转向下
                        price=close,
                        upper=upper.iloc[i],
                        middle=middle.iloc[i],
                        lower=lower.iloc[i],
                        strength=(close - upper.iloc[i]) / (upper.iloc[i] - middle.iloc[i])
                    )
                    signal_dict[timestamp] = signal
                    
                # 突破下轨
                elif prev_close > lower.iloc[i-1] and close < lower.iloc[i]:
                    # 生成突破下轨信号
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='BB_LowerBreakout',
                        direction='buy',  # 下轨突破后通常反转向上
                        price=close,
                        upper=upper.iloc[i],
                        middle=middle.iloc[i],
                        lower=lower.iloc[i],
                        strength=(lower.iloc[i] - close) / (middle.iloc[i] - lower.iloc[i])
                    )
                    signal_dict[timestamp] = signal
                    
                # 从上轨回落
                elif prev_close > upper.iloc[i-1] and close < upper.iloc[i]:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='BB_UpperPullback',
                        direction='sell',
                        price=close,
                        upper=upper.iloc[i],
                        middle=middle.iloc[i],
                        lower=lower.iloc[i],
                        strength=(prev_close - upper.iloc[i-1]) / (upper.iloc[i-1] - middle.iloc[i-1])
                    )
                    signal_dict[timestamp] = signal
                    
                # 从下轨反弹
                elif prev_close < lower.iloc[i-1] and close > lower.iloc[i]:
                    signal = self.add_signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='BB_LowerPullback',
                        direction='buy',
                        price=close,
                        upper=upper.iloc[i],
                        middle=middle.iloc[i],
                        lower=lower.iloc[i],
                        strength=(lower.iloc[i-1] - prev_close) / (middle.iloc[i-1] - lower.iloc[i-1])
                    )
                    signal_dict[timestamp] = signal
                    
            return signal_dict
        except Exception as e:
            logger.error(f"生成布林带信号时发生错误: {str(e)}")
            return {}


class SignalManager:
    """信号管理器 - 管理多个信号生成器"""
    
    def __init__(self):
        """初始化信号管理器"""
        self.generators = {}  # 信号生成器字典: {生成器:权重}
        
    def add_generator(self, generator, weight=1.0):
        """
        添加信号生成器
        
        参数:
        generator (SignalGenerator): 信号生成器实例
        weight (float): 权重, 0-1之间
        """
        self.generators[generator] = weight
        logger.info(f"添加信号生成器: {generator.name}, 权重: {weight}")
        
    def generate_signals(self, data):
        """
        生成所有信号并返回组合后的信号列表
        
        参数:
        data (DataFrame): 市场数据
        
        返回:
        list: 组合后的信号列表
        """
        all_signals = []
        
        # 使用每个生成器生成信号
        for generator, weight in self.generators.items():
            try:
                # 生成信号
                signals_dict = generator.generate(data)
                
                if signals_dict:
                    logger.info(f"信号生成器 {generator.name} 生成了 {len(signals_dict)} 个信号")
                    
                    # 提取信号列表并添加权重
                    signals = list(signals_dict.values())
                    for signal in signals:
                        signal['weight'] = weight
                        all_signals.append(signal)
            except Exception as e:
                logger.error(f"信号生成器 {generator.name} 生成信号时出错: {str(e)}")
        
        # 按时间排序
        all_signals.sort(key=lambda x: x['timestamp'])
        
        return all_signals
        
    def generate_all_signals(self, data):
        """
        生成所有信号
        
        参数:
        data (DataFrame): 市场数据
        
        返回:
        dict: 合并的信号字典, 键为时间戳, 值为信号列表
        """
        all_signals = self.generate_signals(data)
        return {signal['timestamp']: signal for signal in all_signals}
    
    def get_signals_by_date(self, date, direction=None):
        """
        获取指定日期的信号
        
        参数:
        date (datetime): 日期
        direction (str): 方向，'buy'或'sell'，为None表示两个方向都获取
        
        返回:
        list: 符合条件的信号列表
        """
        # 将日期转换为datetime
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # 获取当天信号
        day_signals = [s for s in self.generate_all_signals(data) if s['timestamp'].date() == date.date()]
        
        # 按方向过滤
        if direction:
            day_signals = [s for s in day_signals if s['direction'] == direction]
            
        return day_signals
    
    def get_signals_by_symbol(self, symbol, start_date=None, end_date=None, direction=None):
        """
        获取指定标的的信号
        
        参数:
        symbol (str): 标的代码
        start_date (datetime): 开始日期
        end_date (datetime): 结束日期
        direction (str): 方向，'buy'或'sell'，为None表示两个方向都获取
        
        返回:
        list: 符合条件的信号列表
        """
        # 筛选标的
        symbol_signals = [s for s in self.generate_all_signals(data) if s['symbol'] == symbol]
        
        # 按日期范围过滤
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            symbol_signals = [s for s in symbol_signals if s['timestamp'] >= start_date]
            
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            symbol_signals = [s for s in symbol_signals if s['timestamp'] <= end_date]
            
        # 按方向过滤
        if direction:
            symbol_signals = [s for s in symbol_signals if s['direction'] == direction]
            
        return symbol_signals
    
    def get_composite_signal(self, signals):
        """
        计算综合信号强度
        
        参数:
        signals (list): 信号列表
        
        返回:
        float: 综合信号强度，正值表示多头，负值表示空头
        """
        if not signals:
            return 0
            
        buy_strength = sum(s['strength'] * s['weight'] for s in signals if s['direction'] == 'buy')
        sell_strength = sum(s['strength'] * s['weight'] for s in signals if s['direction'] == 'sell')
        
        return buy_strength - sell_strength
    
    def plot_signals(self, data, symbol=None, start_date=None, end_date=None):
        """
        绘制信号
        
        参数:
        data (DataFrame): 市场数据，包含'close'列和日期索引
        symbol (str): 标的代码，为None表示所有标的
        start_date (datetime): 开始日期
        end_date (datetime): 结束日期
        """
        # 过滤数据
        plot_data = data.copy()
        
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            plot_data = plot_data[plot_data.index >= start_date]
            
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            plot_data = plot_data[plot_data.index <= end_date]
            
        # 获取信号
        if symbol:
            signals = self.get_signals_by_symbol(symbol, start_date, end_date)
        else:
            signals = self.generate_all_signals(data)
            
            # 过滤日期范围
            if start_date:
                signals = [s for s in signals if s['timestamp'] >= start_date]
            if end_date:
                signals = [s for s in signals if s['timestamp'] <= end_date]
        
        # 绘制价格和信号
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制价格
        ax.plot(plot_data.index, plot_data['close'], label='Price')
        
        # 绘制买入信号
        buy_signals = [s for s in signals if s['direction'] == 'buy']
        if buy_signals:
            buy_x = [s['timestamp'] for s in buy_signals]
            buy_y = [plot_data.loc[s['timestamp'], 'close'] if s['timestamp'] in plot_data.index else None for s in buy_signals]
            buy_y = [y for y in buy_y if y is not None]
            buy_x = [x for i, x in enumerate(buy_x) if buy_y[i] is not None]
            
            ax.scatter(buy_x, buy_y, color='green', marker='^', s=100, label='Buy Signal')
            
        # 绘制卖出信号
        sell_signals = [s for s in signals if s['direction'] == 'sell']
        if sell_signals:
            sell_x = [s['timestamp'] for s in sell_signals]
            sell_y = [plot_data.loc[s['timestamp'], 'close'] if s['timestamp'] in plot_data.index else None for s in sell_signals]
            sell_y = [y for y in sell_y if y is not None]
            sell_x = [x for i, x in enumerate(sell_x) if sell_y[i] is not None]
            
            ax.scatter(sell_x, sell_y, color='red', marker='v', s=100, label='Sell Signal')
        
        # 设置图表
        ax.set_title(f'Price and Trading Signals for {symbol if symbol else "All Symbols"}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close = np.random.normal(0, 1, 100).cumsum() + 100
    
    data = pd.DataFrame({
        'open': close * np.random.uniform(0.98, 1.0, 100),
        'high': close * np.random.uniform(1.0, 1.05, 100),
        'low': close * np.random.uniform(0.95, 1.0, 100),
        'close': close,
        'volume': np.random.normal(1000000, 500000, 100)
    }, index=date_range)
    
    # 添加标的列
    data['symbol'] = 'TEST'
    
    # 创建信号管理器
    manager = SignalManager()
    
    # 添加各种信号生成器
    manager.add_generator(MACrossSignalGenerator(fast_period=5, slow_period=20), weight=1.0)
    manager.add_generator(RSISignalGenerator(period=14, overbought=70, oversold=30), weight=0.8)
    manager.add_generator(MACDSignalGenerator(), weight=1.2)
    manager.add_generator(BollingerBandsSignalGenerator(period=20, std_dev=2), weight=1.0)
    
    # 生成信号
    all_signals = manager.generate_all_signals(data)
    
    # 输出信号概况
    print(f"总共生成 {len(all_signals)} 个信号")
    buy_signals = [s for s in all_signals if s['direction'] == 'buy']
    sell_signals = [s for s in all_signals if s['direction'] == 'sell']
    print(f"买入信号: {len(buy_signals)}, 卖出信号: {len(sell_signals)}")
    
    # 绘制信号
    manager.plot_signals(data, symbol='TEST') 