#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
市场数据可视化模块 - 提供市场走势和交易信号的可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import mplfinance as mpf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

# 日志配置
logger = logging.getLogger('MarketVisualizer')

class MarketVisualizer:
    """市场数据可视化类"""
    
    def __init__(self, style='default'):
        """
        初始化市场可视化器
        
        参数:
        style (str): 图表样式
        """
        # 使用内置风格而不是依赖seaborn
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"无法使用指定样式 '{style}': {str(e)}，使用默认样式")
            try:
                plt.style.use('default')
            except:
                # 如果默认样式也不可用，不设置样式
                pass
            
        self.colors = {
            'up': '#ff4d4d',      # 上涨颜色 (红色)
            'down': '#00cc66',    # 下跌颜色 (绿色)
            'volume_up': '#ff9999',  # 成交量上涨颜色
            'volume_down': '#99e699',  # 成交量下跌颜色
            'ma5': '#f1c40f',     # 5日均线 (黄色)
            'ma10': '#3498db',    # 10日均线 (蓝色)
            'ma20': '#9b59b6',    # 20日均线 (紫色)
            'ma30': '#34495e',    # 30日均线 (深灰)
            'ma60': '#e74c3c',    # 60日均线 (红色)
            'ma120': '#2ecc71',   # 120日均线 (绿色)
            'ma250': '#1abc9c',   # 250日均线 (青色)
            'macd': '#3498db',    # MACD线 (蓝色)
            'signal': '#e74c3c',  # 信号线 (红色)
            'histogram_positive': '#00cc66',  # 柱状图正值 (绿色)
            'histogram_negative': '#ff4d4d',  # 柱状图负值 (红色)
            'buy_signal': '#2ecc71',  # 买入信号 (绿色)
            'sell_signal': '#e74c3c',  # 卖出信号 (红色)
            'background': '#f8f9fa',  # 背景色 (浅灰)
            'grid': '#dee2e6',    # 网格线 (灰色)
            'text': '#212529'     # 文本 (深灰)
        }
    
    def plot_candlestick(self, data: pd.DataFrame, title: str = 'Price Chart',
                         ma_periods: List[int] = [5, 10, 20, 60], volume: bool = True,
                         signals: Optional[pd.DataFrame] = None) -> None:
        """
        绘制K线图
        
        参数:
        data (DataFrame): OHLCV数据，包含'open', 'high', 'low', 'close', 'volume'列
        title (str): 图表标题
        ma_periods (List[int]): 移动平均周期列表
        volume (bool): 是否绘制成交量
        signals (DataFrame, optional): 交易信号数据
        """
        try:
            # 检查必要的列
            required_cols = ['open', 'high', 'low', 'close']
            if volume:
                required_cols.append('volume')
                
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"数据中缺少'{col}'列")
                    return
            
            # 确保数据不为空
            if len(data) == 0:
                logger.error("数据为空，无法绘制K线图")
                return
                
            # 准备mplfinance所需格式的数据
            ohlc_data = data[required_cols].copy()
            
            # 添加移动平均线
            for ma in ma_periods:
                if ma < len(data):  # 确保周期小于数据长度
                    ohlc_data[f'MA{ma}'] = data['close'].rolling(window=ma).mean()
                else:
                    logger.warning(f"MA{ma}周期大于数据长度，跳过")
            
            # 设置mplfinance样式
            mc = mpf.make_marketcolors(
                up=self.colors['up'],
                down=self.colors['down'],
                edge='inherit',
                wick='inherit',
                volume={'up': self.colors['volume_up'], 'down': self.colors['volume_down']}
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor=self.colors['grid'],
                facecolor=self.colors['background'],
                figcolor=self.colors['background'],
                y_on_right=False
            )
            
            # 创建均线叠加
            ma_dict = {}
            for ma in ma_periods:
                if f'MA{ma}' in ohlc_data.columns:
                    ma_dict[f'MA{ma}'] = dict(
                        color=self.colors.get(f'ma{ma}', '#000000'),
                        linestyle='-',
                        linewidth=1.0
                    )
            
            # 准备交易信号
            addplots = []
            if signals is not None and not signals.empty:
                try:
                    buy_signals = signals[signals['direction'] == 'buy']
                    sell_signals = signals[signals['direction'] == 'sell']
                    
                    # 检查信号是否与数据有重叠的索引
                    valid_buy_indices = buy_signals.index.intersection(ohlc_data.index)
                    valid_sell_indices = sell_signals.index.intersection(ohlc_data.index)
                    
                    if not valid_buy_indices.empty:
                        buy_markers = [
                            mpf.make_addplot(
                                pd.Series(
                                    buy_signals.loc[idx, 'price'] if idx in buy_signals.index else np.nan,
                                    index=[idx]
                                ),
                                type='scatter',
                                markersize=100,
                                marker='^',
                                color=self.colors['buy_signal']
                            )
                            for idx in valid_buy_indices
                        ]
                        addplots.extend(buy_markers)
                    
                    if not valid_sell_indices.empty:
                        sell_markers = [
                            mpf.make_addplot(
                                pd.Series(
                                    sell_signals.loc[idx, 'price'] if idx in sell_signals.index else np.nan,
                                    index=[idx]
                                ),
                                type='scatter',
                                markersize=100,
                                marker='v',
                                color=self.colors['sell_signal']
                            )
                            for idx in valid_sell_indices
                        ]
                        addplots.extend(sell_markers)
                except Exception as e:
                    logger.warning(f"处理交易信号时出错: {str(e)}")
                
            # 添加均线图
            for ma in ma_periods:
                if f'MA{ma}' in ohlc_data.columns and not ohlc_data[f'MA{ma}'].isna().all():
                    addplots.append(
                        mpf.make_addplot(
                            ohlc_data[f'MA{ma}'],
                            color=self.colors.get(f'ma{ma}', '#000000'),
                            width=1
                        )
                    )
            
            # 绘制图表
            fig, axes = mpf.plot(
                ohlc_data,
                type='candle',
                style=s,
                title=title,
                ylabel='Price',
                ylabel_lower='Volume' if volume else '',
                volume=volume,
                figsize=(12, 8),
                addplot=addplots if addplots else None,
                returnfig=True
            )
            
            # 显示图表
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制K线图时出错: {str(e)}")
    
    def plot_technical_indicators(self, data: pd.DataFrame, indicators: Dict[str, pd.DataFrame],
                                title: str = 'Technical Indicators', signals: Optional[pd.DataFrame] = None) -> None:
        """
        绘制技术指标图表
        
        参数:
        data (DataFrame): 价格数据，包含'close'列
        indicators (Dict[str, DataFrame]): 技术指标字典，键为指标名称，值为指标数据
        title (str): 图表标题
        signals (DataFrame, optional): 交易信号数据
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return
                
            # 计算需要的子图数量
            n_indicators = len(indicators)
            
            # 创建图表
            fig = plt.figure(figsize=(12, 8 + 2 * n_indicators))
            gs = GridSpec(2 + n_indicators, 1, height_ratios=[3] + [1] * n_indicators + [1])
            
            # 绘制价格图
            ax_price = fig.add_subplot(gs[0])
            ax_price.plot(data.index, data['close'], label='Price', color=self.colors['macd'])
            ax_price.set_title(title)
            ax_price.set_ylabel('Price')
            ax_price.grid(True)
            
            # 添加交易信号
            if signals is not None and not signals.empty:
                buy_signals = signals[signals['direction'] == 'buy']
                sell_signals = signals[signals['direction'] == 'sell']
                
                for idx, signal in buy_signals.iterrows():
                    if idx in data.index:
                        ax_price.scatter(idx, data.loc[idx, 'close'], marker='^', color=self.colors['buy_signal'], s=100)
                        
                for idx, signal in sell_signals.iterrows():
                    if idx in data.index:
                        ax_price.scatter(idx, data.loc[idx, 'close'], marker='v', color=self.colors['sell_signal'], s=100)
            
            # 绘制各个技术指标
            for i, (indicator_name, indicator_data) in enumerate(indicators.items(), 1):
                ax = fig.add_subplot(gs[i], sharex=ax_price)
                
                # 根据指标类型进行不同的绘制
                if indicator_name.lower() == 'macd':
                    # MACD指标通常包含MACD线、信号线和柱状图
                    if 'macd' in indicator_data.columns and 'signal' in indicator_data.columns:
                        ax.plot(indicator_data.index, indicator_data['macd'], label='MACD', color=self.colors['macd'])
                        ax.plot(indicator_data.index, indicator_data['signal'], label='Signal', color=self.colors['signal'])
                        
                        # 绘制柱状图（区分正负值）
                        if 'histogram' in indicator_data.columns:
                            for idx, value in indicator_data['histogram'].items():
                                if value >= 0:
                                    ax.bar(idx, value, color=self.colors['histogram_positive'], width=0.5)
                                else:
                                    ax.bar(idx, value, color=self.colors['histogram_negative'], width=0.5)
                        
                        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
                    else:
                        ax.plot(indicator_data.index, indicator_data, label=indicator_name)
                        
                elif indicator_name.lower() == 'rsi':
                    # RSI指标
                    ax.plot(indicator_data.index, indicator_data, label='RSI', color=self.colors['macd'])
                    ax.axhline(70, color=self.colors['sell_signal'], linestyle='--', alpha=0.5)
                    ax.axhline(30, color=self.colors['buy_signal'], linestyle='--', alpha=0.5)
                    ax.axhline(50, color='black', linestyle='--', alpha=0.3)
                    ax.set_ylim(0, 100)
                    
                elif indicator_name.lower() in ['boll', 'bollinger']:
                    # 布林带指标
                    if all(x in indicator_data.columns for x in ['upper', 'middle', 'lower']):
                        ax.plot(indicator_data.index, indicator_data['upper'], label='Upper', color=self.colors['sell_signal'], alpha=0.7)
                        ax.plot(indicator_data.index, indicator_data['middle'], label='Middle', color=self.colors['macd'])
                        ax.plot(indicator_data.index, indicator_data['lower'], label='Lower', color=self.colors['buy_signal'], alpha=0.7)
                        
                        # 添加填充区域
                        ax.fill_between(
                            indicator_data.index,
                            indicator_data['upper'],
                            indicator_data['lower'],
                            color=self.colors['macd'],
                            alpha=0.1
                        )
                    else:
                        ax.plot(indicator_data.index, indicator_data, label=indicator_name)
                        
                elif indicator_name.lower() == 'kdj':
                    # KDJ指标
                    if all(x in indicator_data.columns for x in ['K', 'D', 'J']):
                        ax.plot(indicator_data.index, indicator_data['K'], label='K', color=self.colors['macd'])
                        ax.plot(indicator_data.index, indicator_data['D'], label='D', color=self.colors['signal'])
                        ax.plot(indicator_data.index, indicator_data['J'], label='J', color='green', alpha=0.7)
                        ax.axhline(80, color=self.colors['sell_signal'], linestyle='--', alpha=0.5)
                        ax.axhline(20, color=self.colors['buy_signal'], linestyle='--', alpha=0.5)
                        ax.set_ylim(0, 100)
                    else:
                        ax.plot(indicator_data.index, indicator_data, label=indicator_name)
                        
                else:
                    # 其他指标
                    if isinstance(indicator_data, pd.DataFrame) and len(indicator_data.columns) > 1:
                        for col in indicator_data.columns:
                            ax.plot(indicator_data.index, indicator_data[col], label=col)
                    else:
                        ax.plot(indicator_data.index, indicator_data, label=indicator_name, color=self.colors['macd'])
                
                ax.set_ylabel(indicator_name)
                ax.grid(True)
                ax.legend(loc='upper left')
            
            # 设置x轴日期格式
            date_format = mdates.DateFormatter('%Y-%m-%d')
            for ax in fig.axes:
                ax.xaxis.set_major_formatter(date_format)
                ax.xaxis.set_tick_params(rotation=45)
            
            # 调整布局
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制技术指标时出错: {str(e)}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame, method: str = 'pearson',
                              title: str = 'Correlation Matrix') -> None:
        """
        绘制相关性矩阵热图
        
        参数:
        data (DataFrame): 价格数据，包含多个资产的收盘价
        method (str): 计算相关性的方法，'pearson'或'spearman'
        title (str): 图表标题
        """
        try:
            # 计算相关性矩阵
            correlation = data.corr(method=method)
            
            # 设置图表大小
            plt.figure(figsize=(10, 8))
            
            # 使用seaborn绘制热图
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(
                correlation,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.75}
            )
            
            plt.title(title)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制相关性矩阵时出错: {str(e)}")
    
    def plot_returns_distribution(self, data: pd.DataFrame, window: int = 252,
                                title: str = 'Returns Distribution') -> None:
        """
        绘制收益率分布图
        
        参数:
        data (DataFrame): 价格数据，包含'close'列
        window (int): 计算窗口大小
        title (str): 图表标题
        """
        try:
            if 'close' not in data.columns:
                logger.error("数据中缺少'close'列")
                return
                
            # 计算日收益率和年化收益率
            daily_returns = data['close'].pct_change().dropna()
            rolling_annual_returns = (daily_returns + 1).rolling(window=window).apply(lambda x: np.prod(x) - 1)
            
            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 绘制日收益率分布
            sns.histplot(daily_returns, kde=True, ax=axes[0], color=self.colors['macd'])
            axes[0].axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.4f}')
            axes[0].axvline(0, color='black', linestyle='-')
            axes[0].set_title(f'Daily Returns Distribution (Mean: {daily_returns.mean():.4f}, Std: {daily_returns.std():.4f})')
            axes[0].set_xlabel('Daily Return')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            
            # 绘制年化收益率分布
            sns.histplot(rolling_annual_returns.dropna(), kde=True, ax=axes[1], color=self.colors['ma60'])
            axes[1].axvline(rolling_annual_returns.mean(), color='red', linestyle='--', label=f'Mean: {rolling_annual_returns.mean():.4f}')
            axes[1].axvline(0, color='black', linestyle='-')
            axes[1].set_title(f'Annual Returns Distribution (Mean: {rolling_annual_returns.mean():.4f}, Std: {rolling_annual_returns.std():.4f})')
            axes[1].set_xlabel(f'{window}-Day Rolling Annual Return')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制收益率分布时出错: {str(e)}")
    
    def plot_sector_heatmap(self, sector_data: pd.DataFrame, date: Optional[Union[str, datetime]] = None,
                          title: str = 'Sector Performance Heatmap') -> None:
        """
        绘制板块热力图
        
        参数:
        sector_data (DataFrame): 板块数据，包含'name'和'change_pct'列
        date (Union[str, datetime], optional): 日期
        title (str): 图表标题
        """
        try:
            if 'name' not in sector_data.columns or 'change_pct' not in sector_data.columns:
                logger.error("数据中缺少'name'或'change_pct'列")
                return
                
            # 按涨跌幅排序
            sorted_data = sector_data.sort_values(by='change_pct', ascending=False)
            
            # 设置图表大小
            plt.figure(figsize=(12, 8))
            
            # 创建色块数据
            names = sorted_data['name'].values
            changes = sorted_data['change_pct'].values
            
            # 计算每个板块的颜色
            colors = []
            for change in changes:
                if change > 0:
                    # 使用红色表示上涨
                    intensity = min(change / 5, 1.0)  # 限制最大值为1
                    colors.append((1, 1 - intensity, 1 - intensity))
                else:
                    # 使用绿色表示下跌
                    intensity = min(abs(change) / 5, 1.0)
                    colors.append((1 - intensity, 1, 1 - intensity))
            
            # 绘制热力图
            plt.bar(names, changes, color=colors)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, (name, change) in enumerate(zip(names, changes)):
                plt.text(i, change + (0.2 if change >= 0 else -0.5),
                        f'{change:.2%}',
                        ha='center', va='center', fontsize=8,
                        color='black' if abs(change) < 3 else 'white')
            
            # 设置标题和标签
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
            plt.title(f"{title} - {date_str}" if date_str else title)
            plt.ylabel('Change %')
            plt.xticks(rotation=90)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制板块热力图时出错: {str(e)}")

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.normal(100, 2, 100),
        'high': np.random.normal(102, 2, 100),
        'low': np.random.normal(98, 2, 100),
        'close': np.random.normal(100, 2, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }, index=date_range)
    
    # 修正价格
    for i in range(len(data)):
        row = data.iloc[i]
        high = max(row['open'], row['close']) + abs(np.random.normal(0, 0.5))
        low = min(row['open'], row['close']) - abs(np.random.normal(0, 0.5))
        data.loc[data.index[i], 'high'] = high
        data.loc[data.index[i], 'low'] = low
    
    # 创建可视化器
    visualizer = MarketVisualizer()
    
    # 绘制K线图
    visualizer.plot_candlestick(data, title='Sample Price Chart')
    
    # 创建技术指标
    from technical_indicators import TechnicalIndicator
    
    # 计算MACD
    macd = TechnicalIndicator.MACD(data)
    
    # 计算RSI
    rsi = TechnicalIndicator.RSI(data)
    
    # 计算布林带
    boll = TechnicalIndicator.BOLL(data)
    
    # 绘制技术指标
    indicators = {
        'MACD': macd,
        'RSI': rsi,
        'Bollinger': boll
    }
    
    visualizer.plot_technical_indicators(data, indicators, title='Technical Indicators') 