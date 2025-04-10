#!/usr/bin/env python3
"""
超神量子共生系统 - 量子维度增强器
将系统维度从11维扩展到21维
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Union, Deque

# 配置日志
logger = logging.getLogger("QuantumDimension")

class QuantumDimensionEnhancer:
    """量子维度增强器 - 扩展市场数据的分析维度"""
    
    def __init__(self, config: Optional[Dict] = None, base_dimensions: int = 11, extended_dimensions: int = 10):
        """初始化量子维度增强器
        
        Args:
            config: 配置参数
            base_dimensions: 基础维度数量，默认为11
            extended_dimensions: 扩展维度数量，默认为10
        """
        logger.info("初始化量子维度增强器...")
        
        # 设置默认配置
        self.config = {
            'base_dimensions': base_dimensions,     # 基础维度数
            'extended_dimensions': extended_dimensions,  # 扩展维度数
            'history_window': 50,      # 历史窗口大小
            'weight_update_rate': 0.1, # 权重更新率
            'dimension_memory': 20,    # 维度记忆长度
            'correlation_threshold': 0.7,  # 相关性阈值
            'prediction_horizon': 5,   # 预测时长
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 总维度数
        self.total_dimensions = self.config['base_dimensions'] + self.config['extended_dimensions']
        
        logger.info(f"量子维度扩展器初始化完成，从{self.config['base_dimensions']}维扩展到{self.total_dimensions}维")
        
        # 初始化维度状态
        self._initialize_dimensions()
        
        # 初始化维度权重
        self._initialize_weights()
        
        # 初始化维度历史
        self.dimension_history = {}
        for dim in list(self.base_dimensions.keys()) + list(self.extended_dimensions.keys()):
            self.dimension_history[dim] = deque(maxlen=self.config['dimension_memory'])
        
        # 维度相关矩阵
        self.dimension_correlations = np.identity(self.total_dimensions)
        
        # 预测状态
        self.predictions = {
            'dimension_forecasts': {},
            'last_update': None,
            'accuracy': 0.0
        }
    
    def _initialize_dimensions(self):
        """初始化维度"""
        # 基础维度 (从原始数据中提取)
        self.base_dimensions = {
            'price': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'volume': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'momentum': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'volatility': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'liquidity': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'sentiment': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'trend_strength': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'cycle_position': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'market_breadth': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'relative_strength': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'value_growth_bias': {'value': 0.0, 'trend': 0.0, 'weight': 0.0}
        }
        
        # 扩展维度 (通过量子扩展计算)
        self.extended_dimensions = {
            'fractal': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'entropy': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'cycle_resonance': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'phase_coherence': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'energy_potential': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'network_flow': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'temporal_coherence': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'chaos_degree': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'quantum_state': {'value': 0.0, 'trend': 0.0, 'weight': 0.0},
            'nonlinear_coupling': {'value': 0.0, 'trend': 0.0, 'weight': 0.0}
        }
    
    def _initialize_weights(self):
        """初始化维度权重"""
        # 基础维度初始权重
        base_weights = {
            'price': 0.15,
            'volume': 0.12,
            'momentum': 0.10,
            'volatility': 0.10,
            'liquidity': 0.08,
            'sentiment': 0.09,
            'trend_strength': 0.09,
            'cycle_position': 0.08,
            'market_breadth': 0.07,
            'relative_strength': 0.07,
            'value_growth_bias': 0.05
        }
        
        # 扩展维度初始权重
        extended_weights = {
            'fractal': 0.12,
            'entropy': 0.12,
            'cycle_resonance': 0.10,
            'phase_coherence': 0.10,
            'energy_potential': 0.12,
            'network_flow': 0.09,
            'temporal_coherence': 0.11,
            'chaos_degree': 0.10,
            'quantum_state': 0.08,
            'nonlinear_coupling': 0.06
        }
        
        # 应用初始权重
        for dim, weight in base_weights.items():
            if dim in self.base_dimensions:
                self.base_dimensions[dim]['weight'] = weight
                
        for dim, weight in extended_weights.items():
            if dim in self.extended_dimensions:
                self.extended_dimensions[dim]['weight'] = weight
    
    def enhance_dimensions(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """增强市场数据维度
        
        Args:
            market_data: 市场数据
            
        Returns:
            增强后的数据
        """
        logger.info("开始量子维度扩展...")
        
        try:
            # 检查数据有效性
            if market_data is None or len(market_data) < 10:
                logger.error("市场数据不足")
                return market_data
            
            # 提取基础维度
            self._extract_base_dimensions(market_data)
            
            # 计算扩展维度
            self._calculate_extended_dimensions(market_data)
            
            # 更新维度历史
            self._update_dimension_history()
            
            # 更新维度相关性
            self._update_dimension_correlations()
            
            # 更新维度权重
            self._update_dimension_weights()
            
            # 生成增强数据
            enhanced_data = self._generate_enhanced_data(market_data)
            
            # 更新预测
            self._update_predictions()
            
            logger.info("量子维度扩展完成")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"量子维度扩展失败: {str(e)}")
            return market_data
    
    def _extract_base_dimensions(self, market_data: pd.DataFrame):
        """从市场数据提取基础维度
        
        Args:
            market_data: 市场数据
        """
        try:
            # 提取最近窗口的数据
            window = min(self.config['history_window'], len(market_data))
            recent_data = market_data.iloc[-window:].copy()
            
            # 1. 价格维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                current_price = prices[-1]
                prev_price = prices[0] if len(prices) > 1 else current_price
                price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
                
                # 归一化到0-1
                min_price = min(prices)
                max_price = max(prices)
                norm_price = (current_price - min_price) / (max_price - min_price) if max_price > min_price else 0.5
                
                self.base_dimensions['price']['value'] = current_price
                self.base_dimensions['price']['trend'] = price_change
            
            # 2. 成交量维度
            if 'volume' in recent_data.columns:
                volumes = recent_data['volume'].values
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes)
                volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
                
                self.base_dimensions['volume']['value'] = current_volume
                self.base_dimensions['volume']['trend'] = volume_change
            
            # 3. 动量维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                # 计算10日动量
                momentum_window = min(10, len(returns))
                if momentum_window > 0:
                    momentum = np.sum(returns[-momentum_window:])
                    prev_momentum = np.sum(returns[-momentum_window*2:-momentum_window]) if len(returns) >= momentum_window*2 else 0
                    momentum_change = momentum - prev_momentum
                    
                    self.base_dimensions['momentum']['value'] = momentum
                    self.base_dimensions['momentum']['trend'] = momentum_change
            
            # 4. 波动率维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                volatility_window = min(20, len(returns))
                if volatility_window > 0:
                    volatility = np.std(returns[-volatility_window:])
                    prev_volatility = np.std(returns[-volatility_window*2:-volatility_window]) if len(returns) >= volatility_window*2 else volatility
                    volatility_change = (volatility - prev_volatility) / prev_volatility if prev_volatility > 0 else 0
                    
                    self.base_dimensions['volatility']['value'] = volatility
                    self.base_dimensions['volatility']['trend'] = volatility_change
            
            # 5. 流动性维度
            if 'volume' in recent_data.columns and 'turnover_rate' in recent_data.columns:
                turnover = recent_data['turnover_rate'].values
                current_turnover = turnover[-1] if len(turnover) > 0 else 0
                avg_turnover = np.mean(turnover)
                turnover_change = (current_turnover - avg_turnover) / avg_turnover if avg_turnover > 0 else 0
                
                self.base_dimensions['liquidity']['value'] = current_turnover
                self.base_dimensions['liquidity']['trend'] = turnover_change
            
            # 6. 情绪维度
            # 这里使用价格变化率的累积和作为简单的情绪代理
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                sentiment_window = min(5, len(returns))
                if sentiment_window > 0:
                    sentiment = np.tanh(np.sum(returns[-sentiment_window:]) * 10)  # 使用tanh归一化到[-1,1]
                    prev_sentiment = np.tanh(np.sum(returns[-sentiment_window*2:-sentiment_window]) * 10) if len(returns) >= sentiment_window*2 else 0
                    sentiment_change = sentiment - prev_sentiment
                    
                    self.base_dimensions['sentiment']['value'] = sentiment
                    self.base_dimensions['sentiment']['trend'] = sentiment_change
            
            # 7. 趋势强度维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                
                # 计算短期与长期均线差
                short_period = min(5, len(prices))
                long_period = min(20, len(prices))
                
                if short_period > 0 and long_period > 0:
                    short_ma = np.mean(prices[-short_period:])
                    long_ma = np.mean(prices[-long_period:])
                    
                    # 归一化趋势强度
                    trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
                    trend_strength = np.tanh(trend_strength * 5)  # 使用tanh归一化
                    
                    # 计算趋势变化
                    prev_short_ma = np.mean(prices[-short_period-1:-1]) if len(prices) > short_period else short_ma
                    prev_long_ma = np.mean(prices[-long_period-1:-1]) if len(prices) > long_period else long_ma
                    prev_trend_strength = (prev_short_ma - prev_long_ma) / prev_long_ma if prev_long_ma > 0 else 0
                    prev_trend_strength = np.tanh(prev_trend_strength * 5)
                    
                    trend_change = trend_strength - prev_trend_strength
                    
                    self.base_dimensions['trend_strength']['value'] = trend_strength
                    self.base_dimensions['trend_strength']['trend'] = trend_change
            
            # 8. 周期位置维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                
                # 使用简单的差分和自相关来估计周期位置
                cycle_window = min(30, len(prices))
                if cycle_window > 10:
                    # 计算价格差分
                    diff = np.diff(prices[-cycle_window:])
                    
                    # 计算自相关
                    acf = self._autocorrelation(diff, min(10, len(diff)//3))
                    
                    # 寻找第一个自相关峰值作为周期估计
                    peaks = []
                    for i in range(1, len(acf)-1):
                        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.2:
                            peaks.append((i, acf[i]))
                    
                    # 如果找到峰值，计算当前在周期中的位置
                    if peaks:
                        cycle_length = peaks[0][0]  # 第一个峰值位置即为周期长度
                        if cycle_length > 0:
                            # 计算当前位置与周期的比率 (0-1)
                            position = (len(diff) % cycle_length) / cycle_length
                            
                            self.base_dimensions['cycle_position']['value'] = position
                            self.base_dimensions['cycle_position']['trend'] = 1/cycle_length  # 周期越短，变化越快
            
            # 9. 市场广度维度 (这里需要多股票数据，使用模拟值)
            self.base_dimensions['market_breadth']['value'] = 0.5 + np.random.normal(0, 0.1)
            self.base_dimensions['market_breadth']['trend'] = np.random.normal(0, 0.05)
            
            # 10. 相对强度维度 (这里需要基准数据，使用模拟值)
            if 'close' in recent_data.columns:
                self.base_dimensions['relative_strength']['value'] = np.random.normal(0.5, 0.2)
                self.base_dimensions['relative_strength']['trend'] = np.random.normal(0, 0.05)
            
            # 11. 价值/增长偏向维度 (这里需要风格数据，使用模拟值)
            self.base_dimensions['value_growth_bias']['value'] = np.tanh(np.random.normal(0, 0.5))
            self.base_dimensions['value_growth_bias']['trend'] = np.random.normal(0, 0.05)
            
        except Exception as e:
            logger.error(f"提取基础维度失败: {str(e)}")
    
    def _autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """计算自相关函数
        
        Args:
            data: 输入数据
            max_lag: 最大滞后
            
        Returns:
            自相关函数值
        """
        result = np.zeros(max_lag+1)
        n = len(data)
        
        if n <= 1:
            return result
            
        mean = np.mean(data)
        var = np.var(data)
        
        if var <= 0:
            return result
            
        for lag in range(max_lag+1):
            if lag == 0:
                result[lag] = 1.0
            else:
                cov = 0
                for i in range(n-lag):
                    cov += (data[i] - mean) * (data[i+lag] - mean)
                result[lag] = cov / ((n-lag) * var)
        
        return result

    def _calculate_extended_dimensions(self, market_data: pd.DataFrame):
        """计算扩展维度
        
        Args:
            market_data: 市场数据
        """
        try:
            # 提取最近窗口的数据
            window = min(self.config['history_window'], len(market_data))
            recent_data = market_data.iloc[-window:].copy()
            
            # 1. 分形维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                
                # 使用R/S分析法估计分形特性
                rs_values = []
                lags = [5, 10, 20]
                lags = [lag for lag in lags if lag <= len(prices)//2]
                
                for lag in lags:
                    segments = len(prices) // lag
                    rs = []
                    
                    for i in range(segments):
                        if i*lag + lag <= len(prices):
                            segment = prices[i*lag:i*lag+lag]
                            deviation = segment - np.mean(segment)
                            cumdev = np.cumsum(deviation)
                            r = np.max(cumdev) - np.min(cumdev)
                            s = np.std(segment)
                            if s > 0:
                                rs.append(r/s)
                    
                    if rs:
                        rs_values.append(np.mean(rs))
                
                # 计算赫斯特指数
                if len(rs_values) >= 2 and len(lags) >= 2:
                    try:
                        slope, _, _, _, _ = stats.linregress(np.log(lags[:len(rs_values)]), np.log(rs_values))
                        hurst = slope
                        
                        # 从赫斯特指数计算分形维度
                        fractal_dim = 2.0 - hurst
                        
                        # 归一化到0-1范围
                        fractal_value = (fractal_dim - 1.0) / 1.0
                        fractal_value = max(0.0, min(1.0, fractal_value))
                        
                        # 计算趋势
                        prev_fractal = self.extended_dimensions['fractal']['value']
                        fractal_trend = fractal_value - prev_fractal
                        
                        self.extended_dimensions['fractal']['value'] = fractal_value
                        self.extended_dimensions['fractal']['trend'] = fractal_trend
                    except:
                        # 默认值
                        self.extended_dimensions['fractal']['value'] = 0.5
                        self.extended_dimensions['fractal']['trend'] = 0.0
            
            # 2. 熵维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                if len(returns) > 5:
                    # 使用直方图估计熵
                    bins = min(10, len(returns)//5)
                    if bins >= 2:
                        hist, _ = np.histogram(returns, bins=bins, density=True)
                        bin_width = np.diff(_)[0]
                        probabilities = hist * bin_width
                        
                        # 过滤零概率
                        probabilities = probabilities[probabilities > 0]
                        
                        # 计算香农熵
                        entropy = -np.sum(probabilities * np.log2(probabilities))
                        
                        # 归一化 (最大熵是log2(bins))
                        max_entropy = np.log2(bins)
                        if max_entropy > 0:
                            entropy_value = entropy / max_entropy
                            entropy_value = max(0.01, min(0.99, entropy_value))
                            
                            # 计算趋势
                            prev_entropy = self.extended_dimensions['entropy']['value']
                            entropy_trend = entropy_value - prev_entropy
                            
                            self.extended_dimensions['entropy']['value'] = entropy_value
                            self.extended_dimensions['entropy']['trend'] = entropy_trend
            
            # 3. 周期共振维度
            if 'close' in recent_data.columns and 'cycle_position' in self.base_dimensions:
                cycle_position = self.base_dimensions['cycle_position']['value']
                
                # 检测多个周期的共振
                resonance_value = 0.0
                
                # 计算不同周期长度的自相关
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                if len(returns) > 20:
                    # 计算不同期间的自相关
                    acf5 = self._autocorrelation(returns, 5)
                    acf10 = self._autocorrelation(returns, 10)
                    acf20 = self._autocorrelation(returns, 20)
                    
                    # 寻找显著峰值
                    peaks5 = [i for i in range(1, len(acf5)-1) if acf5[i] > acf5[i-1] and acf5[i] > acf5[i+1] and acf5[i] > 0.2]
                    peaks10 = [i for i in range(1, len(acf10)-1) if acf10[i] > acf10[i-1] and acf10[i] > acf10[i+1] and acf10[i] > 0.2]
                    peaks20 = [i for i in range(1, len(acf20)-1) if acf20[i] > acf20[i-1] and acf20[i] > acf20[i+1] and acf20[i] > 0.2]
                    
                    # 计算共振分数
                    if peaks5 and peaks10:
                        ratio = min(peaks5[0], peaks10[0]) / max(peaks5[0], peaks10[0])
                        if ratio > 0.8:  # 若周期接近整数倍
                            resonance_value += 0.5
                    
                    if peaks10 and peaks20:
                        ratio = min(peaks10[0], peaks20[0]) / max(peaks10[0], peaks20[0])
                        if ratio > 0.8:
                            resonance_value += 0.5
                    
                    # 限制在0-1范围
                    resonance_value = min(1.0, resonance_value)
                
                # 计算趋势
                prev_resonance = self.extended_dimensions['cycle_resonance']['value']
                resonance_trend = resonance_value - prev_resonance
                
                self.extended_dimensions['cycle_resonance']['value'] = resonance_value
                self.extended_dimensions['cycle_resonance']['trend'] = resonance_trend
            
            # 4. 相位相干维度
            if 'trend_strength' in self.base_dimensions and 'momentum' in self.base_dimensions:
                trend = self.base_dimensions['trend_strength']['value']
                momentum = self.base_dimensions['momentum']['value']
                
                # 计算趋势和动量的相干性
                coherence = 1.0 - abs(np.tanh(trend*5) - np.tanh(momentum*5)) / 2.0
                
                # 限制在0-1范围
                coherence_value = max(0.0, min(1.0, coherence))
                
                # 计算趋势
                prev_coherence = self.extended_dimensions['phase_coherence']['value']
                coherence_trend = coherence_value - prev_coherence
                
                self.extended_dimensions['phase_coherence']['value'] = coherence_value
                self.extended_dimensions['phase_coherence']['trend'] = coherence_trend
            
            # 5. 能量势能维度
            if 'price' in self.base_dimensions and 'momentum' in self.base_dimensions:
                price_value = self.base_dimensions['price']['value']
                momentum = self.base_dimensions['momentum']['value']
                trend_strength = self.base_dimensions['trend_strength']['value']
                
                # 计算能量势能 (momentum * trend_strength的组合函数)
                energy = np.tanh(momentum * trend_strength * 3)
                
                # 转换到0-1范围
                energy_value = (energy + 1.0) / 2.0
                
                # 计算趋势
                prev_energy = self.extended_dimensions['energy_potential']['value']
                energy_trend = energy_value - prev_energy
                
                self.extended_dimensions['energy_potential']['value'] = energy_value
                self.extended_dimensions['energy_potential']['trend'] = energy_trend
            
            # 6. 网络流维度
            if 'volume' in self.base_dimensions and 'liquidity' in self.base_dimensions:
                volume = self.base_dimensions['volume']['value']
                liquidity = self.base_dimensions['liquidity']['value']
                
                # 计算资金流动强度
                volumes = recent_data['volume'].values if 'volume' in recent_data.columns else [0]
                avg_volume = np.mean(volumes) if len(volumes) > 0 else 1
                
                # 归一化网络流
                flow_value = min(1.0, volume / (avg_volume * 3)) if avg_volume > 0 else 0.5
                
                # 计算趋势
                prev_flow = self.extended_dimensions['network_flow']['value']
                flow_trend = flow_value - prev_flow
                
                self.extended_dimensions['network_flow']['value'] = flow_value
                self.extended_dimensions['network_flow']['trend'] = flow_trend
            
            # 7. 时间相干维度
            if 'close' in recent_data.columns:
                prices = recent_data['close'].values
                returns = np.diff(prices) / prices[:-1]
                
                if len(returns) > 20:
                    # 计算序列的自相关性
                    acf = self._autocorrelation(returns, min(20, len(returns)//2))
                    
                    # 计算平均自相关强度
                    mean_acf = np.mean(np.abs(acf[1:]))
                    
                    # 转换到0-1范围 (高值表示强自相关)
                    temporal_value = min(1.0, mean_acf * 5)
                    
                    # 计算趋势
                    prev_temporal = self.extended_dimensions['temporal_coherence']['value']
                    temporal_trend = temporal_value - prev_temporal
                    
                    self.extended_dimensions['temporal_coherence']['value'] = temporal_value
                    self.extended_dimensions['temporal_coherence']['trend'] = temporal_trend
            
            # 8. 混沌度维度
            if 'volatility' in self.base_dimensions and 'fractal' in self.extended_dimensions:
                volatility = self.base_dimensions['volatility']['value']
                fractal = self.extended_dimensions['fractal']['value']
                
                # 混沌度是波动率和分形特性的组合
                chaos_value = min(1.0, (volatility * 10) * fractal)
                
                # 计算趋势
                prev_chaos = self.extended_dimensions['chaos_degree']['value']
                chaos_trend = chaos_value - prev_chaos
                
                self.extended_dimensions['chaos_degree']['value'] = chaos_value
                self.extended_dimensions['chaos_degree']['trend'] = chaos_trend
            
            # 9. 量子态维度
            # 这是一个综合多个基本维度的合成维度
            momentum = self.base_dimensions['momentum']['value']
            trend_strength = self.base_dimensions['trend_strength']['value']
            volatility = self.base_dimensions['volatility']['value']
            
            # 计算量子态 (动量、趋势强度和波动率的非线性组合)
            quantum_value = np.tanh((momentum * 2 + trend_strength - volatility * 3) / 2)
            
            # 转换到0-1范围
            quantum_value = (quantum_value + 1.0) / 2.0
            
            # 计算趋势
            prev_quantum = self.extended_dimensions['quantum_state']['value']
            quantum_trend = quantum_value - prev_quantum
            
            self.extended_dimensions['quantum_state']['value'] = quantum_value
            self.extended_dimensions['quantum_state']['trend'] = quantum_trend
            
            # 10. 非线性耦合维度
            if 'price' in self.base_dimensions and 'volume' in self.base_dimensions:
                prices = recent_data['close'].values if 'close' in recent_data.columns else [0]
                volumes = recent_data['volume'].values if 'volume' in recent_data.columns else [0]
                
                if len(prices) > 5 and len(volumes) > 5:
                    # 计算价格和成交量的相关性
                    if np.std(prices) > 0 and np.std(volumes) > 0:
                        price_vol_corr = np.corrcoef(prices, volumes)[0, 1]
                        
                        # 计算非线性耦合强度 (相关性的非线性函数)
                        coupling_value = abs(price_vol_corr)
                        
                        # 计算趋势
                        prev_coupling = self.extended_dimensions['nonlinear_coupling']['value']
                        coupling_trend = coupling_value - prev_coupling
                        
                        self.extended_dimensions['nonlinear_coupling']['value'] = coupling_value
                        self.extended_dimensions['nonlinear_coupling']['trend'] = coupling_trend
            
        except Exception as e:
            logger.error(f"计算扩展维度失败: {str(e)}")
    
    def _update_dimension_history(self):
        """更新维度历史"""
        # 更新基础维度历史
        for dim, state in self.base_dimensions.items():
            self.dimension_history[dim].append({
                'value': state['value'],
                'trend': state['trend'],
                'weight': state['weight']
            })
        
        # 更新扩展维度历史
        for dim, state in self.extended_dimensions.items():
            self.dimension_history[dim].append({
                'value': state['value'],
                'trend': state['trend'],
                'weight': state['weight']
            })
    
    def _update_dimension_correlations(self):
        """更新维度相关性矩阵"""
        try:
            # 创建所有维度的值列表
            all_dimensions = list(self.base_dimensions.keys()) + list(self.extended_dimensions.keys())
            n_dims = len(all_dimensions)
            
            # 创建相关矩阵
            corr_matrix = np.zeros((n_dims, n_dims))
            
            # 对每对维度
            for i, dim1 in enumerate(all_dimensions):
                for j, dim2 in enumerate(all_dimensions):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        continue
                    
                    # 获取历史值
                    values1 = [entry['value'] for entry in self.dimension_history[dim1]]
                    values2 = [entry['value'] for entry in self.dimension_history[dim2]]
                    
                    # 确保有足够的数据点
                    min_len = min(len(values1), len(values2))
                    if min_len >= 3:
                        values1 = values1[-min_len:]
                        values2 = values2[-min_len:]
                        
                        # 计算相关系数
                        if np.std(values1) > 0 and np.std(values2) > 0:
                            corr = np.corrcoef(values1, values2)[0, 1]
                            corr_matrix[i, j] = corr
            
            # 存储相关矩阵
            self.dimension_correlations = corr_matrix
            
        except Exception as e:
            logger.error(f"更新维度相关性失败: {str(e)}")
    
    def _update_dimension_weights(self):
        """更新维度权重"""
        try:
            # 根据不同维度的预测性能更新权重
            update_rate = self.config['weight_update_rate']
            
            # 计算各维度与目标变量(如价格变化)的相关性
            price_changes = []
            for entry in self.dimension_history['price']:
                price_changes.append(entry['trend'])
            
            # 对每个维度
            for dim_dict in [self.base_dimensions, self.extended_dimensions]:
                for dim, state in dim_dict.items():
                    if dim == 'price':
                        continue
                        
                    # 获取维度历史值
                    dim_values = []
                    for entry in self.dimension_history[dim]:
                        dim_values.append(entry['value'])
                    
                    # 确保有足够的数据点
                    min_len = min(len(price_changes), len(dim_values))
                    if min_len >= 3:
                        price_changes_subset = price_changes[-min_len:]
                        dim_values_subset = dim_values[-min_len:]
                        
                        # 计算与价格变化的相关性
                        if np.std(dim_values_subset) > 0 and np.std(price_changes_subset) > 0:
                            corr = abs(np.corrcoef(dim_values_subset, price_changes_subset)[0, 1])
                            
                            # 更新权重
                            old_weight = state['weight']
                            # 相关性越高，权重越大
                            new_weight = (1 - update_rate) * old_weight + update_rate * corr
                            
                            # 确保权重在合理范围内
                            new_weight = max(0.01, min(0.5, new_weight))
                            
                            # 更新权重
                            state['weight'] = new_weight
            
            # 归一化权重
            self._normalize_weights()
            
        except Exception as e:
            logger.error(f"更新维度权重失败: {str(e)}")
    
    def _normalize_weights(self):
        """归一化权重"""
        # 归一化基础维度权重
        base_sum = sum(state['weight'] for state in self.base_dimensions.values())
        if base_sum > 0:
            for state in self.base_dimensions.values():
                state['weight'] /= base_sum
        
        # 归一化扩展维度权重
        ext_sum = sum(state['weight'] for state in self.extended_dimensions.values())
        if ext_sum > 0:
            for state in self.extended_dimensions.values():
                state['weight'] /= ext_sum
    
    def _generate_enhanced_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """生成增强数据
        
        Args:
            market_data: 市场数据
            
        Returns:
            增强后的数据
        """
        try:
            # 创建增强数据的副本
            enhanced_data = market_data.copy()
            
            # 添加基础维度列
            for dim, state in self.base_dimensions.items():
                column_name = f"{dim}_dimension"
                enhanced_data[column_name] = np.nan
                if len(enhanced_data) > 0:
                    enhanced_data.loc[enhanced_data.index[-1], column_name] = state['value']
            
            # 添加扩展维度列
            for dim, state in self.extended_dimensions.items():
                column_name = f"{dim}_dimension"
                enhanced_data[column_name] = np.nan
                if len(enhanced_data) > 0:
                    enhanced_data.loc[enhanced_data.index[-1], column_name] = state['value']
            
            # 计算综合维度指标
            quantum_index = self._calculate_quantum_index()
            enhanced_data['quantum_dimension_index'] = np.nan
            if len(enhanced_data) > 0:
                enhanced_data.loc[enhanced_data.index[-1], 'quantum_dimension_index'] = quantum_index
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"生成增强数据失败: {str(e)}")
            return market_data
    
    def _calculate_quantum_index(self) -> float:
        """计算量子维度综合指标
        
        Returns:
            量子指标值
        """
        try:
            # 基础维度综合
            base_index = sum(state['value'] * state['weight'] for state in self.base_dimensions.values())
            
            # 扩展维度综合
            ext_index = sum(state['value'] * state['weight'] for state in self.extended_dimensions.values())
            
            # 组合指标 (可以使用不同比例)
            quantum_index = 0.4 * base_index + 0.6 * ext_index
            
            return quantum_index
            
        except Exception as e:
            logger.error(f"计算量子指标失败: {str(e)}")
            return 0.5
    
    def _update_predictions(self):
        """更新维度预测"""
        try:
            # 清除旧预测
            self.predictions['dimension_forecasts'] = {}
            
            # 对每个维度生成预测
            for dim_name, dim_state in {**self.base_dimensions, **self.extended_dimensions}.items():
                # 获取历史值
                values = [entry['value'] for entry in self.dimension_history[dim_name]]
                trends = [entry['trend'] for entry in self.dimension_history[dim_name]]
                
                if len(values) < 3:
                    continue
                
                # 计算简单预测 (当前值 + 趋势 * 衰减因子)
                horizon = self.config['prediction_horizon']
                forecasts = []
                
                current_value = values[-1]
                current_trend = trends[-1]
                
                for h in range(1, horizon+1):
                    # 趋势衰减
                    decay = np.exp(-0.3 * h)
                    # 预测值
                    pred = current_value + current_trend * decay * h
                    # 约束在0-1范围
                    pred = max(0.0, min(1.0, pred))
                    forecasts.append(pred)
                
                # 存储预测
                self.predictions['dimension_forecasts'][dim_name] = {
                    'current': current_value,
                    'horizon': horizon,
                    'values': forecasts
                }
            
            # 更新时间戳
            self.predictions['last_update'] = pd.Timestamp.now()
            
        except Exception as e:
            logger.error(f"更新维度预测失败: {str(e)}")
    
    def get_dimension_state(self) -> Dict:
        """获取维度状态
        
        Returns:
            维度状态字典
        """
        # 合并基础和扩展维度
        all_dimensions = {}
        
        for dim, state in self.base_dimensions.items():
            all_dimensions[dim] = {
                'value': state['value'],
                'trend': state['trend'],
                'weight': state['weight'],
                'type': 'base'
            }
        
        for dim, state in self.extended_dimensions.items():
            all_dimensions[dim] = {
                'value': state['value'],
                'trend': state['trend'],
                'weight': state['weight'],
                'type': 'extended'
            }
        
        # 添加预测
        if self.predictions['dimension_forecasts']:
            for dim, forecast in self.predictions['dimension_forecasts'].items():
                if dim in all_dimensions:
                    all_dimensions[dim]['forecast'] = forecast['values']
        
        return all_dimensions
    
    def get_quantum_index(self) -> float:
        """获取量子维度综合指标
        
        Returns:
            量子指标值
        """
        return self._calculate_quantum_index()
    
    def get_predictions(self) -> Dict:
        """获取维度预测
        
        Returns:
            预测字典
        """
        return self.predictions

def get_dimension_enhancer(config: Optional[Dict] = None) -> QuantumDimensionEnhancer:
    """工厂函数 - 创建并返回量子维度增强器实例"""
    return QuantumDimensionEnhancer(config)

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # 创建测试数据
    dates = pd.date_range(start='2023-01-01', periods=50, freq='B')
    test_data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(3000, 50, 50),
        'high': np.random.normal(3050, 60, 50),
        'low': np.random.normal(2950, 60, 50),
        'close': np.random.normal(3000, 50, 50),
        'volume': np.random.normal(1e9, 2e8, 50),
        'turnover_rate': np.random.uniform(0.5, 2.0, 50)
    })
    
    # 创建量子维度增强器
    enhancer = get_dimension_enhancer()
    
    # 增强维度
    enhanced_data = enhancer.enhance_dimensions(test_data)
    
    # 获取维度状态
    dim_state = enhancer.get_dimension_state()
    
    # 打印维度状态
    print("\n维度状态:")
    print("基础维度:")
    for dim, state in dim_state.items():
        if state['type'] == 'base':
            print(f"{dim}: 值={state['value']:.3f}, 趋势={state['trend']:.3f}, 权重={state['weight']:.3f}")
    
    print("\n扩展维度:")
    for dim, state in dim_state.items():
        if state['type'] == 'extended':
            print(f"{dim}: 值={state['value']:.3f}, 趋势={state['trend']:.3f}, 权重={state['weight']:.3f}") 