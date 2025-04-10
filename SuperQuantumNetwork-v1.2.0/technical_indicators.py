#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标计算模块 - 提供各种常用技术指标的计算实现
"""

import numpy as np
import pandas as pd
import talib as ta
import logging

logger = logging.getLogger('TechnicalIndicators')

class TechnicalIndicator:
    """技术指标计算基类"""
    
    @staticmethod
    def MA(df, period=5, price='close'):
        """
        计算移动平均线 Moving Average
        
        参数:
        df (DataFrame): 包含价格数据的DataFrame
        period (int): 周期
        price (str): 使用的价格列名，默认为'close'
        
        返回:
        Series: 移动平均线值
        """
        try:
            if price not in df.columns:
                logger.error(f"列 {price} 不存在于DataFrame中")
                return None
                
            return df[price].rolling(window=period).mean()
        except Exception as e:
            logger.error(f"计算MA时出错: {str(e)}")
            return None
    
    @staticmethod
    def EMA(df, period=12, price='close'):
        """
        计算指数移动平均线 Exponential Moving Average
        
        参数:
        df (DataFrame): 包含价格数据的DataFrame
        period (int): 周期
        price (str): 使用的价格列名，默认为'close'
        
        返回:
        Series: 指数移动平均线值
        """
        try:
            if price not in df.columns:
                logger.error(f"列 {price} 不存在于DataFrame中")
                return None
                
            return df[price].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"计算EMA时出错: {str(e)}")
            return None
    
    @staticmethod
    def MACD(df, fast_period=12, slow_period=26, signal_period=9, price='close'):
        """
        计算MACD (Moving Average Convergence Divergence)
        
        参数:
        df (DataFrame): 包含价格数据的DataFrame
        fast_period (int): 快线周期
        slow_period (int): 慢线周期
        signal_period (int): 信号线周期
        price (str): 使用的价格列名，默认为'close'
        
        返回:
        DataFrame: 包含MACD, Signal和Histogram的DataFrame
        """
        try:
            if price not in df.columns:
                logger.error(f"列 {price} 不存在于DataFrame中")
                return None
                
            # 计算快线和慢线
            ema_fast = TechnicalIndicator.EMA(df, period=fast_period, price=price)
            ema_slow = TechnicalIndicator.EMA(df, period=slow_period, price=price)
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # 计算柱状图
            histogram = macd_line - signal_line
            
            # 创建结果DataFrame
            result = pd.DataFrame({
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }, index=df.index)
            
            return result
        except Exception as e:
            logger.error(f"计算MACD时出错: {str(e)}")
            return None
    
    @staticmethod
    def RSI(df, period=14, price='close'):
        """
        计算RSI (Relative Strength Index)
        
        参数:
        df (DataFrame): 包含价格数据的DataFrame
        period (int): 周期
        price (str): 使用的价格列名，默认为'close'
        
        返回:
        Series: RSI值
        """
        try:
            if price not in df.columns:
                logger.error(f"列 {price} 不存在于DataFrame中")
                return None
                
            # 计算价格变化
            delta = df[price].diff()
            
            # 区分上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 计算移动平均
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # 计算RS和RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"计算RSI时出错: {str(e)}")
            return None
    
    @staticmethod
    def BOLL(df, period=20, std_dev=2, price='close'):
        """
        计算布林带 (Bollinger Bands)
        
        参数:
        df (DataFrame): 包含价格数据的DataFrame
        period (int): 周期
        std_dev (float): 标准差倍数
        price (str): 使用的价格列名，默认为'close'
        
        返回:
        DataFrame: 包含上轨、中轨和下轨的DataFrame
        """
        try:
            if price not in df.columns:
                logger.error(f"列 {price} 不存在于DataFrame中")
                return None
                
            # 计算中轨（SMA）
            middle_band = df[price].rolling(window=period).mean()
            
            # 计算标准差
            std = df[price].rolling(window=period).std()
            
            # 计算上轨和下轨
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            # 创建结果DataFrame
            result = pd.DataFrame({
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band
            }, index=df.index)
            
            return result
        except Exception as e:
            logger.error(f"计算Bollinger Bands时出错: {str(e)}")
            return None
            
    @staticmethod
    def KDJ(df, n=9, m1=3, m2=3):
        """
        计算KDJ指标
        
        参数:
        df (DataFrame): 包含OHLC数据的DataFrame
        n (int): RSV周期
        m1 (int): K值平滑因子
        m2 (int): D值平滑因子
        
        返回:
        DataFrame: 包含K,D,J值的DataFrame
        """
        try:
            # 检查必要的列
            required_cols = ['high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"列 {col} 不存在于DataFrame中")
                    return None
            
            # 计算RSV
            rsv = 100 * ((df['close'] - df['low'].rolling(n).min()) / 
                         (df['high'].rolling(n).max() - df['low'].rolling(n).min()))
            
            # 计算K,D,J
            k = rsv.ewm(alpha=1/m1, adjust=False).mean()
            d = k.ewm(alpha=1/m2, adjust=False).mean()
            j = 3 * k - 2 * d
            
            # 创建结果DataFrame
            result = pd.DataFrame({
                'K': k,
                'D': d,
                'J': j
            }, index=df.index)
            
            return result
        except Exception as e:
            logger.error(f"计算KDJ时出错: {str(e)}")
            return None
    
    @staticmethod
    def CCI(df, period=14):
        """
        计算CCI (Commodity Channel Index)
        
        参数:
        df (DataFrame): 包含OHLC数据的DataFrame
        period (int): 周期
        
        返回:
        Series: CCI值
        """
        try:
            # 检查必要的列
            required_cols = ['high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"列 {col} 不存在于DataFrame中")
                    return None
            
            # 计算典型价格
            tp = (df['high'] + df['low'] + df['close']) / 3
            
            # 计算典型价格的SMA
            tp_sma = tp.rolling(window=period).mean()
            
            # 计算均差
            md = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            
            # 计算CCI
            cci = (tp - tp_sma) / (0.015 * md)
            
            return cci
        except Exception as e:
            logger.error(f"计算CCI时出错: {str(e)}")
            return None

    @staticmethod
    def ATR(df, period=14):
        """
        计算ATR (Average True Range)
        
        参数:
        df (DataFrame): 包含OHLC数据的DataFrame
        period (int): 周期
        
        返回:
        Series: ATR值
        """
        try:
            # 检查必要的列
            required_cols = ['high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"列 {col} 不存在于DataFrame中")
                    return None
            
            # 计算True Range
            df_tr = pd.DataFrame()
            df_tr['h-l'] = df['high'] - df['low']
            df_tr['h-pc'] = np.abs(df['high'] - df['close'].shift(1))
            df_tr['l-pc'] = np.abs(df['low'] - df['close'].shift(1))
            
            df_tr['tr'] = df_tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            
            # 计算ATR
            atr = df_tr['tr'].rolling(window=period).mean()
            
            return atr
        except Exception as e:
            logger.error(f"计算ATR时出错: {str(e)}")
            return None
    
    @staticmethod
    def VWAP(df):
        """
        计算VWAP (Volume Weighted Average Price)
        
        参数:
        df (DataFrame): 包含OHLC和交易量数据的DataFrame
        
        返回:
        Series: VWAP值
        """
        try:
            # 检查必要的列
            required_cols = ['high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"列 {col} 不存在于DataFrame中")
                    return None
            
            # 计算典型价格
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3
            
            # 计算交易量乘以典型价格
            df['pv'] = df['tp'] * df['volume']
            
            # 计算累计交易量和累计价格交易量乘积
            df['cum_vol'] = df['volume'].cumsum()
            df['cum_pv'] = df['pv'].cumsum()
            
            # 计算VWAP
            vwap = df['cum_pv'] / df['cum_vol']
            
            # 删除临时列
            df.drop(['tp', 'pv', 'cum_vol', 'cum_pv'], axis=1, inplace=True)
            
            return vwap
        except Exception as e:
            logger.error(f"计算VWAP时出错: {str(e)}")
            return None

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000000, 500000, 100)
    }, index=date_range)
    
    # 修正高低价
    for i in range(len(df)):
        max_val = max(df.iloc[i]['open'], df.iloc[i]['close'])
        min_val = min(df.iloc[i]['open'], df.iloc[i]['close'])
        df.loc[df.index[i], 'high'] = max(df.iloc[i]['high'], max_val)
        df.loc[df.index[i], 'low'] = min(df.iloc[i]['low'], min_val)
    
    # 测试各种指标
    ma = TechnicalIndicator.MA(df, period=5)
    print(f"MA5: {ma.iloc[-1]:.2f}")
    
    ema = TechnicalIndicator.EMA(df, period=12)
    print(f"EMA12: {ema.iloc[-1]:.2f}")
    
    macd = TechnicalIndicator.MACD(df)
    print(f"MACD: {macd['macd'].iloc[-1]:.2f}, Signal: {macd['signal'].iloc[-1]:.2f}")
    
    rsi = TechnicalIndicator.RSI(df)
    print(f"RSI: {rsi.iloc[-1]:.2f}")
    
    boll = TechnicalIndicator.BOLL(df)
    print(f"BOLL: Upper={boll['upper'].iloc[-1]:.2f}, Middle={boll['middle'].iloc[-1]:.2f}, Lower={boll['lower'].iloc[-1]:.2f}")
    
    kdj = TechnicalIndicator.KDJ(df)
    print(f"KDJ: K={kdj['K'].iloc[-1]:.2f}, D={kdj['D'].iloc[-1]:.2f}, J={kdj['J'].iloc[-1]:.2f}") 