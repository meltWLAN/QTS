#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子爆发策略 - 增强版
集成了高维量子感知和多维度市场预测
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
import math

from event import EventType, MarketDataEvent, SignalEvent
from data import DataHandler

logger = logging.getLogger(__name__)

class QuantumBurstStrategyEnhanced:
    """
    增强版量子爆发策略 - 集成高维度量子共振预测
    
    特点:
    1. 多维度量子特征: 整合动量、波动率、趋势强度、市场情绪等维度
    2. 自适应参数: 根据市场环境动态调整策略参数
    3. 波动预测: 使用非线性模型预测未来价格波动
    4. 市场状态识别: 识别牛市、熊市、震荡市等不同市场状态
    5. 风险管理: 动态止损和仓位管理
    """
    
    def __init__(self, data_handler=None):
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化超神量子爆发策略增强版")
        
        # 初始化数据处理器
        self.data_handler = data_handler
        
        # 策略基本配置 - 调整为更低的门槛
        self.max_positions = 10         # 最大持仓数量
        self.lookback_period = 60       # 历史数据回溯期
        self.min_days_required = 20     # 最小所需历史数据天数
        self.price_threshold = 0.08     # 价格变化阈值 (降低以捕捉更多信号)
        self.volume_threshold = 1.5     # 成交量变化阈值 (降低以捕捉更多信号)
        self.signal_threshold = 0.55    # 信号强度阈值 (降低以捕捉更多信号)
        
        # 风险管理参数
        self.base_stop_loss = 0.05      # 基础止损比例 (5%)
        self.trailing_stop = 0.4        # 追踪止损系数 (回撤超过盈利的40%)
        self.take_profit = 0.20         # 止盈比例 (20%)
        self.max_hold_days = 20         # 最大持仓天数
        
        # 市场状态和参数自适应
        self.market_state = "neutral"   # 市场状态: bullish, bearish, neutral, volatile
        self.market_index_code = '000001.SH'  # 默认使用上证指数
        self.market_index_data = None    # 市场指数数据
        self.market_trend = 0.0          # 市场趋势强度
        self.market_volatility = 0.0     # 市场波动率
        self.param_history = []          # 参数调整历史
        
        # 增强型量子特征
        self.quantum_weights = {
            'trend_strength': 0.20,    # 趋势强度
            'momentum': 0.20,          # 动量
            'volatility': 0.15,        # 波动率
            'volume_structure': 0.15,  # 量价结构
            'reversal_pattern': 0.10,  # 反转模式
            'support_resistance': 0.10,# 支撑阻力
            'quantum_oscillator': 0.10 # 量子震荡器
        }
        
        # 持仓管理
        self.positions = {}              # 当前持仓
        self.position_hold_days = {}     # 持仓天数
        self.position_entry_prices = {}  # 入场价格
        self.position_high_prices = {}   # 持仓期间最高价格
        self.position_scores = {}        # 持仓评分
        
        # 初始化技术指标缓存
        self.indicator_cache = {}

    def update_market_state(self, event):
        """更新市场状态"""
        current_date = pd.Timestamp(event.timestamp).date()
        self.current_date = current_date
        self.logger.debug(f"更新市场状态，当前日期：{current_date}")

        # 1. 获取当前可用的股票列表
        if hasattr(self.data_handler, 'get_all_securities'):
            available_stocks = self.data_handler.get_all_securities()
        else:
            # 如果数据处理器没有get_all_securities方法，则使用symbols属性
            available_stocks = self.data_handler.symbols if hasattr(self.data_handler, 'symbols') else []
            
        self.logger.info(f"可用股票代码数量: {len(available_stocks)}")
        if available_stocks:
            self.logger.info(f"可用股票代码示例: {available_stocks[:3]}...")
            
            # 获取第一个股票的历史数据来调试
            try:
                first_stock = available_stocks[0]
                # 检查是否有特定的方法来获取历史数据
                if hasattr(self.data_handler, 'get_history_bars'):
                    history_data = self.data_handler.get_history_bars(
                        first_stock, 
                        self.lookback_period, 
                        frequency='1d',
                        fields=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                        end_date=current_date
                    )
                else:
                    # 尝试使用get_historical_data方法
                    history_data = self.data_handler.get_historical_data(
                        first_stock, 
                        self.lookback_period
                    )
                    # 将列表格式转换为DataFrame格式
                    if history_data and isinstance(history_data, list):
                        history_data = pd.DataFrame(history_data)
                
                # 打印历史数据的基本信息
                self.logger.info(f"获取到的历史数据点数: {len(history_data) if history_data is not None else 0}")
                
                if history_data is not None and len(history_data) > 0:
                    if isinstance(history_data, pd.DataFrame):
                        # 打印历史数据的列名
                        self.logger.info(f"历史数据列名: {list(history_data.columns)}")
                        
                        # 打印第一条和最后一条数据
                        self.logger.info(f"第一条历史数据: {history_data.iloc[0].to_dict()}")
                        self.logger.info(f"最后一条历史数据: {history_data.iloc[-1].to_dict()}")
                    else:
                        # 如果不是DataFrame格式，打印简要信息
                        self.logger.info(f"第一条历史数据: {history_data[0] if len(history_data) > 0 else 'N/A'}")
                        self.logger.info(f"最后一条历史数据: {history_data[-1] if len(history_data) > 0 else 'N/A'}")
                else:
                    self.logger.warning(f"无法获取股票 {first_stock} 的历史数据")
            except Exception as e:
                self.logger.error(f"获取历史数据时出错: {str(e)}")
        
        # 2. 尝试获取市场指数数据
        try:
            # 尝试多个主要指数，直到找到可用的
            market_index_found = False
            for index_code in [self.market_index_code, '000001.SH', '399001.SZ', '000300.SH']:
                if index_code in available_stocks:
                    self.logger.info(f"尝试使用 {index_code} 作为市场指数")
                    try:
                        # 根据数据处理器的API选择适当的方法获取历史数据
                        if hasattr(self.data_handler, 'get_history_bars'):
                            market_index_data = self.data_handler.get_history_bars(
                                index_code, 
                                self.lookback_period, 
                                frequency='1d',
                                fields=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                                end_date=current_date
                            )
                        else:
                            # 尝试使用get_historical_data方法
                            market_index_data = self.data_handler.get_historical_data(
                                index_code, 
                                self.lookback_period
                            )
                            # 检查获取到的数据
                            if market_index_data is None:
                                self.logger.warning(f"指数 {index_code} 的历史数据为None")
                                continue
                                
                            if len(market_index_data) == 0:
                                self.logger.warning(f"指数 {index_code} 的历史数据为空列表")
                                continue
                                
                            self.logger.info(f"成功获取指数 {index_code} 数据，条数: {len(market_index_data)}")
                            
                            # 将列表格式转换为DataFrame格式
                            if isinstance(market_index_data, list):
                                self.logger.info(f"将列表格式转换为DataFrame，第一条数据: {market_index_data[0]}")
                                market_index_data = pd.DataFrame(market_index_data)
                        
                        # 验证数据有效性
                        if market_index_data is not None and len(market_index_data) > 0:
                            self.logger.info(f"成功获取市场指数 {index_code} 数据，数据点数: {len(market_index_data)}")
                            if isinstance(market_index_data, pd.DataFrame):
                                self.logger.info(f"指数数据列: {list(market_index_data.columns)}")
                                self.logger.info(f"指数数据第一行: {market_index_data.iloc[0].to_dict()}")
                            else:
                                self.logger.info(f"指数数据类型: {type(market_index_data)}")
                                
                            self.market_index_data = market_index_data
                            self.market_index_code = index_code  # 更新成功获取数据的指数代码
                            market_index_found = True
                            break  # 找到有效指数数据后退出循环
                        else:
                            self.logger.warning(f"市场指数 {index_code} 数据为空，尝试下一个指数")
                    except Exception as e:
                        self.logger.warning(f"获取指数 {index_code} 数据时出错: {str(e)}")
                else:
                    self.logger.warning(f"指数 {index_code} 不在可用列表中")
            
            if not market_index_found:
                raise ValueError("所有尝试的市场指数都无法获取数据")
        except Exception as e:
            self.logger.warning(f"无法获取市场指数数据: {str(e)}，使用所有股票平均表")
            
            # 3. 如果获取市场指数失败，创建一个综合指数
            try:
                composite_data = None
                date_field = None
                stocks_processed = 0
                
                for stock_code in available_stocks:
                    if self._is_index_symbol(stock_code):
                        continue  # 跳过指数类型的代码
                        
                    try:
                        # 根据数据处理器的API选择适当的方法获取历史数据
                        if hasattr(self.data_handler, 'get_history_bars'):
                            stock_data = self.data_handler.get_history_bars(
                                stock_code, 
                                self.lookback_period, 
                                frequency='1d',
                                fields=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                                end_date=current_date
                            )
                        else:
                            # 尝试使用get_historical_data方法
                            stock_data = self.data_handler.get_historical_data(
                                stock_code, 
                                self.lookback_period
                            )
                            # 将列表格式转换为DataFrame格式
                            if stock_data and isinstance(stock_data, list):
                                stock_data = pd.DataFrame(stock_data)
                        
                        if stock_data is None or len(stock_data) == 0:
                            self.logger.debug(f"股票 {stock_code} 没有可用的历史数据")
                            continue
                            
                        # 确定日期字段
                        if date_field is None:
                            if isinstance(stock_data, pd.DataFrame):
                                if 'date' in stock_data.columns:
                                    date_field = 'date'
                                    self.logger.info(f"使用 'date' 作为日期字段")
                                elif 'trade_date' in stock_data.columns:
                                    date_field = 'trade_date'
                                    self.logger.info(f"使用 'trade_date' 作为日期字段")
                                else:
                                    self.logger.warning(f"找不到日期字段，可用列: {list(stock_data.columns)}")
                                    continue
                            else:
                                # 如果是列表格式，假设每个元素都是字典且包含'date'字段
                                if 'date' in stock_data[0]:
                                    date_field = 'date'
                                    self.logger.info(f"使用 'date' 作为日期字段")
                                else:
                                    self.logger.warning(f"找不到日期字段，数据格式: {type(stock_data[0])}")
                                    continue
                        
                        # 初始化复合数据
                        if composite_data is None:
                            if isinstance(stock_data, pd.DataFrame):
                                composite_data = pd.DataFrame({date_field: stock_data[date_field]})
                                composite_data['close_sum'] = 0
                                composite_data['volume_sum'] = 0
                                composite_data['stocks_count'] = 0
                            else:
                                # 对于列表格式，先创建日期列表
                                dates = [item[date_field] for item in stock_data]
                                composite_data = pd.DataFrame({
                                    date_field: dates,
                                    'close_sum': 0,
                                    'volume_sum': 0,
                                    'stocks_count': 0
                                })
                            self.logger.info(f"初始化复合数据结构，使用日期字段: {date_field}")
                        
                        # 添加数据到复合指数
                        if isinstance(stock_data, pd.DataFrame):
                            # 确保日期匹配
                            merged_data = pd.merge(composite_data, stock_data[[date_field, 'close']], on=date_field, how='inner')
                            
                            if 'close' not in stock_data.columns:
                                self.logger.warning(f"股票 {stock_code} 数据中没有 'close' 字段")
                                continue
                                
                            # 更新复合数据
                            for idx, row in merged_data.iterrows():
                                date_val = row[date_field]
                                close_val = stock_data.loc[stock_data[date_field] == date_val, 'close'].values[0]
                                volume_val = stock_data.loc[stock_data[date_field] == date_val, 'volume'].values[0] if 'volume' in stock_data.columns else 0
                                
                                composite_data.loc[composite_data[date_field] == date_val, 'close_sum'] += close_val
                                composite_data.loc[composite_data[date_field] == date_val, 'volume_sum'] += volume_val
                                composite_data.loc[composite_data[date_field] == date_val, 'stocks_count'] += 1
                        else:
                            # 对于列表格式，遍历股票数据中的每个项目
                            for item in stock_data:
                                date_val = item[date_field]
                                close_val = item['close']
                                volume_val = item.get('volume', 0)
                                
                                # 找到对应日期的行
                                row_idx = composite_data[composite_data[date_field] == date_val].index
                                if len(row_idx) > 0:
                                    composite_data.loc[row_idx, 'close_sum'] += close_val
                                    composite_data.loc[row_idx, 'volume_sum'] += volume_val
                                    composite_data.loc[row_idx, 'stocks_count'] += 1
                        
                        stocks_processed += 1
                        if stocks_processed % 10 == 0:
                            self.logger.debug(f"已处理 {stocks_processed} 只股票来创建综合指数")
                        
                        # 限制处理的股票数量，避免过度消耗资源
                        if stocks_processed >= 30:  # 使用前30只股票创建综合指数
                            self.logger.info("已达到股票处理上限，停止收集更多股票数据")
                            break
                            
                    except Exception as stock_err:
                        self.logger.debug(f"处理股票 {stock_code} 数据时出错: {str(stock_err)}")
                
                if composite_data is not None and len(composite_data) > 0:
                    # 计算平均值
                    composite_data['close'] = composite_data['close_sum'] / composite_data['stocks_count']
                    composite_data['volume'] = composite_data['volume_sum']
                    
                    # 添加其他必要字段
                    composite_data['open'] = composite_data['close']  # 简化处理
                    composite_data['high'] = composite_data['close']
                    composite_data['low'] = composite_data['close']
                    
                    # 保存综合指数数据
                    self.market_index_data = composite_data[[date_field, 'open', 'high', 'low', 'close', 'volume']].rename(columns={date_field: 'date'})
                    self.logger.info(f"成功创建综合指数，包含 {stocks_processed} 只股票的数据，数据点数: {len(self.market_index_data)}")
                    
                    if len(self.market_index_data) > 0:
                        self.logger.info(f"综合指数第一条数据: {self.market_index_data.iloc[0].to_dict()}")
                        self.logger.info(f"综合指数最后一条数据: {self.market_index_data.iloc[-1].to_dict()}")
                else:
                    self.logger.error("无法创建有效的综合指数数据")
            except Exception as composite_err:
                self.logger.error(f"创建综合指数时出错: {str(composite_err)}")
        
        # 4. 如果我们有市场数据，计算市场指标
        if hasattr(self, 'market_index_data') and self.market_index_data is not None:
            self._calculate_market_indicators()
        else:
            self.logger.error("没有可用的市场数据来计算指标")

        # 5. 更新量子相关特征
        self._update_quantum_features()

        # 6. 更新市场状态和模式
        self._update_market_mode()

    def _adapt_parameters(self):
        """
        根据市场状态自适应调整策略参数
        """
        try:
            # 保存当前参数
            current_params = {
                'price_threshold': self.price_threshold,
                'volume_threshold': self.volume_threshold,
                'signal_threshold': self.signal_threshold,
                'stop_loss': self.base_stop_loss,
                'max_positions': self.max_positions
            }
            self.param_history.append(current_params)
            
            # 根据市场状态调整参数
            if self.market_state == "bullish":
                # 看涨市场 - 更积极的入场, 更宽松的止损
                self.price_threshold = 0.08  # 降低价格门槛
                self.volume_threshold = 1.3  # 降低成交量门槛
                self.signal_threshold = 0.55  # 降低信号强度门槛
                self.base_stop_loss = 0.045  # 适度放宽止损
                self.max_positions = 10  # 增加持仓上限
                self.take_profit_ratio = 0.25
                self.position_sizing_factor = 1.2
                
                # 调整特征权重 - 偏向趋势和动量
                self.quantum_weights['trend_strength'] = 0.30
                self.quantum_weights['momentum'] = 0.25
                self.quantum_weights['volatility'] = 0.10
                self.quantum_weights['volume_structure'] = 0.15
                self.quantum_weights['reversal_pattern'] = 0.05
                self.quantum_weights['support_resistance'] = 0.05
                self.quantum_weights['quantum_oscillator'] = 0.10
                
                self.logger.info("牛市模式：采用更激进的交易设置")
                
            elif self.market_state == "bearish":
                # 看跌市场 - 更严格的入场, 更紧的止损
                self.price_threshold = 0.12  # 提高价格门槛
                self.volume_threshold = 1.8  # 提高成交量门槛
                self.signal_threshold = 0.65  # 提高信号强度门槛
                self.base_stop_loss = 0.035  # 收紧止损
                self.max_positions = 5  # 减少持仓上限
                self.take_profit_ratio = 0.15
                self.position_sizing_factor = 0.8
                
                # 调整特征权重 - 偏向波动率和防御因素
                self.quantum_weights['volatility'] = 0.25
                self.quantum_weights['trend_strength'] = 0.15
                self.quantum_weights['momentum'] = 0.10
                self.quantum_weights['volume_structure'] = 0.15
                self.quantum_weights['reversal_pattern'] = 0.15
                self.quantum_weights['support_resistance'] = 0.15
                self.quantum_weights['quantum_oscillator'] = 0.05
                
                self.logger.info("熊市模式：采用更保守的交易设置")
                
            elif self.market_state == "volatile":
                # 震荡市场 - 注重波动特征, 适中止损
                self.price_threshold = 0.10  # 适中价格门槛
                self.volume_threshold = 1.5  # 适中成交量门槛
                self.signal_threshold = 0.60  # 适中信号强度门槛
                self.base_stop_loss = 0.04  # 适中止损
                self.max_positions = 7  # 适中持仓上限
                self.take_profit_ratio = 0.18
                self.position_sizing_factor = 1.0
                
                # 调整特征权重 - 偏向波动性和反转
                self.quantum_weights['volatility'] = 0.20
                self.quantum_weights['reversal_pattern'] = 0.20
                self.quantum_weights['volume_structure'] = 0.20
                self.quantum_weights['trend_strength'] = 0.10
                self.quantum_weights['momentum'] = 0.10
                self.quantum_weights['support_resistance'] = 0.10
                self.quantum_weights['quantum_oscillator'] = 0.10
                
                self.logger.info("震荡市场模式：采用波动交易设置")
                
            else:  # neutral市场
                # 中性市场 - 均衡配置
                self.price_threshold = 0.09  # 均衡价格门槛
                self.volume_threshold = 1.5  # 均衡成交量门槛
                self.signal_threshold = 0.58  # 均衡信号强度门槛
                self.base_stop_loss = 0.04  # 均衡止损
                self.max_positions = 8  # 均衡持仓上限
                self.take_profit_ratio = 0.20
                self.position_sizing_factor = 1.0
                
                # 调整特征权重 - 均衡配置
                self.quantum_weights['trend_strength'] = 0.20
                self.quantum_weights['momentum'] = 0.20
                self.quantum_weights['volatility'] = 0.15
                self.quantum_weights['volume_structure'] = 0.15
                self.quantum_weights['reversal_pattern'] = 0.10
                self.quantum_weights['support_resistance'] = 0.10
                self.quantum_weights['quantum_oscillator'] = 0.10
                
                self.logger.info("中性市场模式：采用均衡交易设置")
            
            self.logger.debug(f"特征权重调整为: {self.quantum_weights}")
                
        except Exception as e:
            self.logger.error(f"自适应参数调整失败: {str(e)}")
            # 出错时使用默认配置

    # 自定义RSI计算函数
    def calculate_rsi(self, prices, period=14):
        """计算相对强弱指标"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0:
            return 100
        rs = up / down
        return 100 - (100 / (1 + rs))
    
    # 自定义布林带计算函数
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """计算布林带"""
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower
    
    # 自定义ATR计算函数
    def calculate_atr(self, high_prices, low_prices, close_prices, period=14):
        """计算平均真实范围"""
        if len(high_prices) < period + 1:
            return np.mean(high_prices - low_prices)
            
        true_ranges = []
        for i in range(1, len(close_prices)):
            true_range = max(
                high_prices[i] - low_prices[i],  # 当日范围
                abs(high_prices[i] - close_prices[i-1]),  # 当日高点与前日收盘价之差
                abs(low_prices[i] - close_prices[i-1])  # 当日低点与前日收盘价之差
            )
            true_ranges.append(true_range)
            
        return np.mean(true_ranges[-period:]) if true_ranges else 0
        
    def calculate_quantum_features(self, symbol: str, current_data: Dict, historical_data: List[Dict]) -> Dict[str, float]:
        """
        计算量子特征
        """
        try:
            # 创建必要的特征字典，初始化所有特征为0
            features = {
                'trend_strength': 0.0,
                'momentum': 0.0,
                'volatility': 0.0,
                'volume_structure': 0.0,
                'reversal_pattern': 0.0,
                'support_resistance': 0.0,
                'quantum_oscillator': 0.0
            }
            
            # 确保所有在quantum_weights中定义的特征都被初始化
            for feature_name in self.quantum_weights.keys():
                if feature_name not in features:
                    features[feature_name] = 0.0
            
            # 短期、中期和长期价格变化
            current_price = float(current_data['close'])
            
            # 确保有足够的历史数据
            if len(historical_data) < 10:
                return features
                
            # 提取价格序列
            prices = [float(bar['close']) for bar in historical_data[-20:]]
            prices.append(current_price)
            
            volumes = [float(bar['volume']) for bar in historical_data[-20:]]
            volumes.append(float(current_data['volume']))
            
            # 计算短期趋势 (5日)
            short_trend = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            
            # 计算中期趋势 (10日)
            mid_trend = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # 计算长期趋势 (20日)
            long_trend = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
            
            # 趋势一致性
            trend_consistency = 1.0
            if (short_trend > 0 and mid_trend > 0 and long_trend > 0) or \
               (short_trend < 0 and mid_trend < 0 and long_trend < 0):
                trend_consistency = 1.0
            elif (short_trend > 0 and mid_trend > 0) or (short_trend < 0 and mid_trend < 0):
                trend_consistency = 0.7
            else:
                trend_consistency = 0.3
                
            # 趋势强度
            trend_strength = abs(short_trend) * 0.5 + abs(mid_trend) * 0.3 + abs(long_trend) * 0.2
            features['trend_strength'] = min(1.0, trend_strength * 10) * trend_consistency
            
            # 计算量子动量
            price_diff = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            momentum = sum(price_diff[-5:]) / prices[-5] if len(price_diff) >= 5 else 0
            features['momentum'] = min(1.0, max(0.0, (momentum * 10 + 0.5)))
            
            # 计算波动率
            std = np.std(price_diff[-10:]) / np.mean(prices[-10:]) if len(price_diff) >= 10 else 0
            features['volatility'] = min(1.0, std * 100)
            
            # 计算量价结构
            vol_change = [volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1.0 for i in range(1, len(volumes))]
            price_vol_correlation = 0.5
            if len(price_diff) >= 5 and len(vol_change) >= 5:
                # 计算价格变化和成交量变化的相关性
                # 正相关表示量价齐升或量价齐跌
                corr = np.corrcoef(price_diff[-5:], vol_change[-5:])[0, 1] if not np.isnan(np.corrcoef(price_diff[-5:], vol_change[-5:])[0, 1]) else 0
                price_vol_correlation = (corr + 1) / 2  # 将相关系数从[-1,1]映射到[0,1]
            
            features['volume_structure'] = price_vol_correlation
            
            # 计算反转模式
            recent_high = max(prices[-5:]) if len(prices) >= 5 else prices[-1]
            recent_low = min(prices[-5:]) if len(prices) >= 5 else prices[-1]
            
            # 判断是否处于超买或超卖区域的简单方法
            range_ratio = (current_price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
            reversal_signal = 0.0
            
            if range_ratio > 0.8:  # 可能超买
                # 检查是否有下跌信号
                if current_price < prices[-2]:  # 最近有回落
                    reversal_signal = 0.8  # 看跌反转
            elif range_ratio < 0.2:  # 可能超卖
                # 检查是否有反弹信号
                if current_price > prices[-2]:  # 最近有反弹
                    reversal_signal = 0.8  # 看涨反转
                    
            features['reversal_pattern'] = reversal_signal
            
            # 支撑阻力计算
            if len(prices) >= 20:
                # 简单计算支撑阻力水平
                sorted_prices = sorted(prices[:-1])  # 不包括当前价
                n = len(sorted_prices)
                
                # 找出价格聚集区
                price_clusters = []
                for i in range(n):
                    if i > 0 and abs(sorted_prices[i] - sorted_prices[i-1]) / sorted_prices[i-1] < 0.01:
                        continue
                    count = sum(1 for p in prices[:-1] if abs(p - sorted_prices[i]) / sorted_prices[i] < 0.01)
                    if count >= 3:  # 至少在3个时间点附近出现
                        price_clusters.append((sorted_prices[i], count))
                
                # 根据当前价格与支撑阻力位的关系评分
                score = 0.5  # 默认中性评分
                for level, count in price_clusters:
                    if 0.99 * level <= current_price <= 1.01 * level:  # 当前价在支撑阻力位附近
                        break_strength = abs(current_price - level) / level
                        direction = 1 if current_price > level else -1  # 1表示价格高于阻力位，-1表示低于支撑位
                        
                        # 高于阻力位或低于支撑位的程度评分
                        score = 0.5 + direction * (0.5 - break_strength * 10)
                        break
                
                features['support_resistance'] = score
            else:
                features['support_resistance'] = 0.5
            
            # 量子震荡器 - 结合多个指标创建的综合性指标
            # 这里使用RSI作为示例
            if len(prices) >= 14:
                rsi = self.calculate_rsi(prices)
                # 正规化RSI值为0到1，这样RSI=50对应0.5
                norm_rsi = rsi / 100
                features['quantum_oscillator'] = norm_rsi
            else:
                features['quantum_oscillator'] = 0.5
            
            self.logger.debug(f"计算的量子特征: {', '.join([f'{k}={v:.2f}' for k,v in features.items()])}")
            return features
            
        except Exception as e:
            self.logger.error(f"计算量子特征时出错: {str(e)}")
            # 返回默认特征值
            default_features = {
                'trend_strength': 0.0,
                'momentum': 0.0,
                'volatility': 0.0,
                'volume_structure': 0.0,
                'reversal_pattern': 0.0,
                'support_resistance': 0.0,
                'quantum_oscillator': 0.0
            }
            # 确保所有权重中的特征都有默认值
            for feature_name in self.quantum_weights.keys():
                if feature_name not in default_features:
                    default_features[feature_name] = 0.0
            return default_features

    def calculate_quantum_score(self, features: Dict[str, float]) -> float:
        """
        计算综合量子评分
        """
        score = 0.0
        for feature, value in features.items():
            if feature in self.quantum_weights:
                score += value * self.quantum_weights[feature]
        
        # 添加非线性变换，增强信号对比度
        score = 1.0 / (1.0 + math.exp(-5 * (score - 0.5)))  # Sigmoid函数
        
        return score
    
    def generate_signals(self, event: MarketDataEvent) -> List[SignalEvent]:
        """
        生成交易信号
        """
        if event.type != EventType.MARKET_DATA or not self.data_handler:
            return []
            
        # 首先更新市场状态
        self.update_market_state(event)
        
        signals = []
        
        # 处理已有持仓的平仓信号
        for symbol in list(self.positions.keys()):
            if symbol in event.data:
                exit_signal = self._check_exit_signals(symbol, event.data[symbol], event.timestamp)
                if exit_signal:
                    signals.append(exit_signal)
        
        # 即使持仓已满，也计算潜在信号强度，以便在下一个循环考虑替换表现较差的持仓
        # 但只有在持仓未满时才实际生成买入信号
        max_positions_reached = len(self.positions) >= self.max_positions
        if max_positions_reached:
            self.logger.info(f"当前持仓数量已达到上限 {self.max_positions}，但仍继续评估潜在买入机会")
        
        # 创建候选信号清单
        candidate_signals = []
        
        # 处理新的买入信号
        stocks_processed = 0
        for symbol, data in event.data.items():
            # 跳过指数和已有持仓的股票
            if self._is_index(symbol) or symbol in self.positions:
                continue
                
            try:
                # 获取足够的历史数据用于分析
                historical_data = self.data_handler.get_historical_data(symbol, self.min_days_required)
                
                # 如果没有足够的历史数据，跳过此股票
                if len(historical_data) < self.min_days_required:
                    continue
                
                # 计算价格变化
                current_price = float(data['close'])
                prev_day_close = float(historical_data[-1]['close'])
                price_change = (current_price - prev_day_close) / prev_day_close
                
                # 计算相对成交量变化
                current_volume = float(data['volume'])
                avg_volume = sum(float(bar['volume']) for bar in historical_data[-5:]) / 5
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # 1. 优化价格动量计算 - 同时考虑短中长期
                price_momentum_short = 0
                price_momentum_mid = 0  
                price_momentum_long = 0
                
                if len(historical_data) >= 3:
                    price_momentum_short = (current_price - float(historical_data[-3]['close'])) / float(historical_data[-3]['close'])
                
                if len(historical_data) >= 8:
                    price_momentum_mid = (current_price - float(historical_data[-8]['close'])) / float(historical_data[-8]['close'])
                    
                if len(historical_data) >= 20:
                    price_momentum_long = (current_price - float(historical_data[-20]['close'])) / float(historical_data[-20]['close'])
                
                # 综合价格动量
                price_momentum = 0.5 * price_momentum_short + 0.3 * price_momentum_mid + 0.2 * price_momentum_long
                
                # 2. 计算量子特征
                quantum_features = self.calculate_quantum_features(symbol, data, historical_data)
                quantum_score = self.calculate_quantum_score(quantum_features)
                
                # 3. 计算MACD指标
                prices = [float(bar['close']) for bar in historical_data]
                prices.append(current_price)
                
                # 改进MACD计算
                ema12 = self._calculate_ema(prices, 12)
                ema26 = self._calculate_ema(prices, 26)
                macd_line = ema12 - ema26
                signal_line = self._calculate_ema([0] * (len(prices) - 10) + [macd_line], 9) if len(prices) > 10 else 0
                macd_histogram = macd_line - signal_line
                
                # MACD零线突破判断
                macd_zero_cross = 0
                if len(prices) > 27:
                    prev_macd = self._calculate_ema(prices[:-1], 12) - self._calculate_ema(prices[:-1], 26)
                    # 零线突破加分
                    if prev_macd < 0 and macd_line > 0:
                        macd_zero_cross = 1
                
                # 4. 计算布林带突破情况
                upper, middle, lower = self.calculate_bollinger_bands(prices)
                bb_position = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
                
                # 布林带突破判断
                bb_breakout = 0
                if current_price > upper:
                    # 向上突破布林带上轨
                    bb_breakout = 1.0
                elif current_price < lower:
                    # 向下突破布林带下轨
                    bb_breakout = -1.0
                
                # 5. 改进相对强弱指数计算和判断
                rsi = self.calculate_rsi(prices)
                
                # RSI超买超卖信号
                rsi_signal = 0
                if rsi < 30:  # 超卖
                    rsi_signal = (30 - rsi) / 30  # 越超卖信号越强
                elif rsi > 70:  # 超买
                    rsi_signal = (rsi - 70) / 30 * -1  # 超买为负向信号
                    
                # 6. 新增：趋势强度指标
                trend_score = 0
                if len(prices) >= 20:
                    # 计算20日移动平均线斜率
                    ma20_now = sum(prices[-20:]) / 20
                    ma20_prev = sum(prices[-21:-1]) / 20 if len(prices) >= 21 else ma20_now
                    ma_slope = (ma20_now - ma20_prev) / ma20_prev if ma20_prev > 0 else 0
                    
                    # 多均线系统
                    ma5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else prices[-1]
                    ma10 = sum(prices[-10:]) / 10 if len(prices) >= 10 else prices[-1]
                    
                    # 均线多头排列得分
                    if ma5 > ma10 > ma20_now:
                        ma_alignment = 1.0
                    elif ma5 > ma10 and ma10 < ma20_now:
                        ma_alignment = 0.5
                    else:
                        ma_alignment = 0
                    
                    # 判断价格是否站上均线
                    above_ma = current_price > ma20_now
                    
                    # 综合评分
                    trend_score = (0.5 * max(0, ma_slope * 100) + 0.3 * ma_alignment + 0.2 * (1 if above_ma else 0))
                
                # 7. 新增：波动率评分
                volatility_score = 0
                if len(prices) >= 10:
                    # 计算10日波动率
                    returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
                    volatility = np.std(returns[-10:])
                    
                    # 根据市场状态调整对波动率的评价
                    if self.market_state == 'bullish':
                        # 牛市更喜欢中等波动率
                        volatility_score = 1 - abs(volatility - 0.015) / 0.015 if volatility < 0.03 else 0
                    elif self.market_state == 'bearish':
                        # 熊市更喜欢低波动率
                        volatility_score = 1 - volatility / 0.02 if volatility < 0.02 else 0
                    else:
                        # 中性或震荡市场喜欢适度波动
                        volatility_score = 1 - abs(volatility - 0.02) / 0.02 if volatility < 0.04 else 0
                
                # 8. 新增：KDJ指标判断超买超卖
                kdj_score = 0
                if len(prices) >= 9:
                    # 计算最高价和最低价
                    highs = [float(bar['high']) for bar in historical_data[-9:]]
                    lows = [float(bar['low']) for bar in historical_data[-9:]]
                    
                    # 加入当天数据
                    highs.append(float(data['high']))
                    lows.append(float(data['low']))
                    
                    # 计算RSV
                    period_high = max(highs)
                    period_low = min(lows)
                    if period_high > period_low:
                        rsv = 100 * (current_price - period_low) / (period_high - period_low)
                    else:
                        rsv = 50
                    
                    # 简单K值计算
                    k_value = rsv
                    # 根据KDJ值评分
                    if k_value < 20:  # 超卖区
                        kdj_score = 0.8
                    elif k_value > 80:  # 超买区
                        kdj_score = -0.5
                
                # 9. 新增：量能分析
                volume_score = 0
                volume_days = min(5, len(historical_data))
                if volume_days >= 3:
                    # 检查连续放量情况
                    volume_increase_count = 0
                    for i in range(1, volume_days):
                        if float(historical_data[-i]['volume']) > float(historical_data[-(i+1)]['volume']):
                            volume_increase_count += 1
                    
                    # 连续放量评分
                    if volume_increase_count >= 2:
                        volume_score = 0.3 + 0.1 * min(volume_increase_count, 3)
                        
                    # 当日超额放量
                    if volume_ratio > 1.5:
                        volume_score += 0.3 * min((volume_ratio - 1.5) / 1.5, 1.0)
                
                # 组合各种因子计算最终信号强度
                # 根据市场状态调整各因子权重
                signal_strength = 0
                
                if self.market_state == 'bullish':
                    # 牛市更重视动量和趋势
                    signal_strength = (
                        0.15 * min(1.0, max(0, price_change / (self.price_threshold * 0.8))) +  # 降低阈值
                        0.12 * min(1.0, max(0, (volume_ratio - 1) / (self.volume_threshold * 0.9))) +  # 降低阈值
                        0.20 * quantum_score +
                        0.10 * max(0, macd_histogram / 0.008) +  # 降低阈值
                        0.08 * min(1.0, max(0, (rsi - 45) / 20)) +  # 降低RSI阈值
                        0.10 * trend_score +
                        0.08 * volatility_score +
                        0.05 * macd_zero_cross +
                        0.07 * kdj_score +
                        0.05 * volume_score
                    )
                elif self.market_state == 'bearish':
                    # 熊市更重视安全性和超跌反弹
                    signal_strength = (
                        0.12 * min(1.0, max(0, price_change / (self.price_threshold * 0.75))) +  # 更低阈值
                        0.10 * min(1.0, max(0, (volume_ratio - 1) / (self.volume_threshold * 0.9))) +
                        0.15 * quantum_score +
                        0.08 * max(0, macd_histogram / 0.006) +  # 更低阈值
                        0.15 * max(0, (30 - min(rsi, 30)) / 30) +  # 超卖反弹信号
                        0.08 * trend_score +
                        0.10 * volatility_score +
                        0.10 * rsi_signal +  # 重视RSI超卖
                        0.08 * max(0, kdj_score) +  # 只考虑正向KDJ信号
                        0.04 * bb_breakout  # 考虑布林带反弹
                    )
                else:
                    # 中性市场平衡考虑各因素
                    signal_strength = (
                        0.12 * min(1.0, max(0, price_change / (self.price_threshold * 0.8))) +
                        0.10 * min(1.0, max(0, (volume_ratio - 1) / (self.volume_threshold * 0.9))) +
                        0.18 * quantum_score +
                        0.12 * max(0, macd_histogram / 0.007) +
                        0.08 * rsi_signal +  # 使用RSI信号
                        0.10 * trend_score +
                        0.10 * volatility_score +
                        0.08 * kdj_score +
                        0.07 * max(0, bb_position - 0.5) +  # 价格位置靠上加分
                        0.05 * volume_score
                    )
                
                # 降低信号阈值
                signal_threshold = self.signal_threshold * 0.8
                
                # 评估是否生成买入信号
                if signal_strength > signal_threshold:
                    trade_size = self.calculate_trade_size(symbol, current_price)
                    if trade_size > 0:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type=SignalType.LONG,
                            strength=signal_strength,
                            trade_size=trade_size
                        )
                        candidate_signals.append((symbol, signal, signal_strength))
                        self.logger.info(f"生成买入信号候选 - 股票: {symbol}, 信号强度: {signal_strength:.4f}, 数量: {trade_size}")
                    
                stocks_processed += 1
            except Exception as e:
                self.logger.error(f"处理股票 {symbol} 时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        self.logger.info(f"分析了 {stocks_processed} 只股票作为潜在买入机会")
        
        # 如果持仓未满，根据信号强度选择最强的N个信号
        available_slots = self.max_positions - len(self.positions)
        if available_slots > 0 and candidate_signals:
            # 按信号强度排序
            candidate_signals.sort(key=lambda x: x[2], reverse=True)
            
            # 选择最强的N个信号
            selected_signals = candidate_signals[:available_slots]
            
            for _, signal, strength in selected_signals:
                signals.append(signal)
                self.logger.info(f"最终买入信号 - 股票: {signal.symbol}, 信号强度: {strength:.4f}, 数量: {signal.trade_size}")
        
        return signals

    # 新增：计算EMA的辅助函数
    def _calculate_ema(self, prices, period):
        """计算指数移动平均线"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
            
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
            
        return ema

    def _check_exit_signals(self, symbol: str, current_data: Dict, current_time: datetime) -> Optional[SignalEvent]:
        """
        检查是否应该平仓
        """
        try:
            if symbol not in self.positions:
                return None
                
            entry_price = self.position_entry_prices.get(symbol, 0)
            if entry_price <= 0:
                return None
                
            current_price = float(current_data['close'])
            price_change = (current_price - entry_price) / entry_price
            
            # 更新持仓天数和最高价
            self.position_hold_days[symbol] += 1
            self.position_high_prices[symbol] = max(self.position_high_prices[symbol], current_price)
            
            # 计算回撤
            max_price = self.position_high_prices[symbol]
            drawdown_from_max = (max_price - current_price) / max_price
            
            # 止损检查 - 包括固定止损和追踪止损
            stop_triggered = False
            stop_reason = ""
            
            # 1. 基础止损
            if price_change < -self.base_stop_loss:
                stop_triggered = True
                stop_reason = f"基础止损 ({price_change:.2%})"
                
            # 2. 追踪止损 (只有获利时才启用)
            elif price_change > 0.05 and drawdown_from_max > self.trailing_stop * price_change:
                stop_triggered = True
                stop_reason = f"追踪止损 (从最高点{max_price:.2f}回撤{drawdown_from_max:.2%})"
                
            # 3. 止盈检查
            elif price_change > self.take_profit:
                stop_triggered = True
                stop_reason = f"止盈 (+{price_change:.2%})"
                
            # 4. 持仓时间限制
            elif self.position_hold_days[symbol] >= self.max_hold_days:
                stop_triggered = True
                stop_reason = f"持仓时间到期 ({self.position_hold_days[symbol]}天)"
            
            if stop_triggered:
                # 清理持仓记录
                del self.positions[symbol]
                del self.position_hold_days[symbol]
                del self.position_entry_prices[symbol]
                del self.position_scores[symbol]
                del self.position_high_prices[symbol]
                
                self.logger.info(f"【卖出信号】{symbol}: {stop_reason}, 入场价={entry_price:.2f}, 当前价={current_price:.2f}, 收益={price_change:.2%}")
                
                return SignalEvent(
                    symbol=symbol,
                    datetime=current_time,
                    signal_type='EXIT',
                    strength=1.0,
                    price=current_price
                )
                
            return None
                
        except Exception as e:
            self.logger.error(f"检查{symbol}平仓信号时出错: {str(e)}")
            return None

    def _is_index_symbol(self, symbol):
        """
        判断是否为指数符号
        """
        # 上证指数、深证成指、沪深300、中证500等
        index_patterns = [
            '.SH',  # 上证系列指数
            '.SZ',  # 深证系列指数
            '.CSI', # 中证系列指数
            '.CSI300', # 沪深300
            '.SSE',  # 上交所指数
            '.SZSE'  # 深交所指数
        ]
        
        # 常见指数代码前缀 - 不限于中国市场
        index_prefixes = [
            '000001.SH',  # 上证指数
            '399001.SZ',  # 深证成指
            '000300.SH',  # 沪深300
            '000905.SH',  # 中证500
            '000016.SH',  # 上证50
            '399006.SZ',  # 创业板指
            '000688.SH'   # 科创50
        ]
        
        # 直接匹配特定指数代码
        if symbol in index_prefixes:
            self.logger.debug(f"识别到指数: {symbol}")
            return True
            
        # 检查后缀模式
        for pattern in index_patterns:
            if symbol.endswith(pattern):
                self.logger.debug(f"根据模式识别到指数: {symbol}")
                return True
                
        return False
        
    def _calculate_market_indicators(self):
        """计算市场指标"""
        try:
            # 确保市场数据已可用
            if not hasattr(self, 'market_index_data') or self.market_index_data is None or len(self.market_index_data) == 0:
                self.logger.error("没有可用的市场数据来计算指标")
                return
                
            self.logger.info(f"开始计算市场指标，可用数据点: {len(self.market_index_data)}")
            
            # 计算市场趋势指标
            prices = self.market_index_data['close'].values
            
            # 确保有足够的数据计算移动平均线
            if len(prices) < 20:
                self.logger.warning(f"数据点不足以计算市场指标（只有 {len(prices)} 个点）")
                return
                
            # 计算移动平均线
            short_ma_period = min(10, len(prices) - 1)
            long_ma_period = min(20, len(prices) - 1)
            
            ma_short = np.mean(prices[-short_ma_period:])
            ma_long = np.mean(prices[-long_ma_period:])
            
            # 计算市场波动率
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
            
            # 计算趋势强度
            trend_strength = (ma_short / ma_long - 1) * 100  # 百分比差异
            
            # 保存计算结果
            self.market_volatility = volatility
            self.market_trend = trend_strength
            
            self.logger.info(f"市场波动率: {volatility:.2f}, 趋势强度: {trend_strength:.2f}%")
            
        except Exception as e:
            self.logger.error(f"计算市场指标时出错: {str(e)}")
            # 设置默认值
            self.market_volatility = 0.15
            self.market_trend = 0.0
    
    def _update_quantum_features(self):
        """更新量子相关特征"""
        try:
            # 这里添加量子特征的计算逻辑
            # 简单模拟量子特征
            self.quantum_entropy = np.random.random() * 0.5 + 0.5  # 0.5-1.0之间的随机值
            self.quantum_coherence = np.random.random() * 0.5 + 0.5  # 0.5-1.0之间的随机值
            
            self.logger.debug(f"更新量子特征: 熵={self.quantum_entropy:.2f}, 相干性={self.quantum_coherence:.2f}")
            
        except Exception as e:
            self.logger.error(f"更新量子特征时出错: {str(e)}")
            self.quantum_entropy = 0.75
            self.quantum_coherence = 0.75
    
    def _update_market_mode(self):
        """更新市场状态和模式"""
        try:
            # 确保已计算市场趋势和波动率
            if not hasattr(self, 'market_trend') or not hasattr(self, 'market_volatility'):
                self.logger.warning("市场趋势或波动率未计算，使用默认市场模式")
                self.market_state = "neutral"
                return
                
            # 调整市场状态判断阈值，使其更容易捕捉到不同的市场环境
            if self.market_trend > 1.5:  # 降低牛市判断门槛 (从2.0降至1.5)
                self.market_state = "bullish"
            elif self.market_trend < -1.5:  # 降低熊市判断门槛 (从-2.0降至-1.5)
                self.market_state = "bearish"
            elif self.market_volatility > 0.18:  # 降低高波动判断门槛 (从0.20降至0.18)
                self.market_state = "volatile"
            else:
                self.market_state = "neutral"
                
            # 输出市场指标，用于调试
            self.logger.info(f"市场趋势: {self.market_trend:.2f}, 市场波动率: {self.market_volatility:.2f}")
                
            # 根据市场状态调整策略参数
            self._adapt_parameters()
            
            self.logger.info(f"市场状态更新为: {self.market_state}")
            
        except Exception as e:
            self.logger.error(f"更新市场模式时出错: {str(e)}")
            self.market_state = "neutral"  # 出错时默认为中性市场 

    def _is_index(self, symbol: str) -> bool:
        """
        判断股票代码是否为指数
        """
        index_symbols = ['000001.SH', '399001.SZ', '000300.SH', '399006.SZ', '000016.SH', '000905.SH', 
                         '000852.SH', '399005.SZ', '399300.SZ', '399673.SZ', '399550.SZ', '000688.SH']
        
        if symbol in index_symbols:
            return True
        
        # 检查常见指数格式
        if (symbol.startswith('000') and symbol.endswith('.SH')) or \
           (symbol.startswith('399') and symbol.endswith('.SZ')):
            return True
        
        return False 