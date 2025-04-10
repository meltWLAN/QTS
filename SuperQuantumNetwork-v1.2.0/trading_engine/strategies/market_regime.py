#!/usr/bin/env python3
"""
超神量子共生系统 - 市场体制分析模块
用于分析当前市场所处的宏观环境和微观状态
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

# 设置日志
logger = logging.getLogger("TradingEngine.MarketRegime")

class MarketRegimeType(Enum):
    """市场体制类型枚举"""
    BULLISH = "bullish"           # 牛市
    BEARISH = "bearish"           # 熊市
    SIDEWAYS = "sideways"         # 盘整
    VOLATILE = "volatile"         # 震荡
    TRENDING_UP = "trending_up"   # 上升趋势
    TRENDING_DOWN = "trending_down"  # 下降趋势
    REVERSAL_UP = "reversal_up"   # 向上反转
    REVERSAL_DOWN = "reversal_down"  # 向下反转
    BREAKOUT = "breakout"         # 突破
    BREAKDOWN = "breakdown"       # 跌破
    EXPANSION = "expansion"       # 波动扩大
    CONTRACTION = "contraction"   # 波动收缩
    CYCLE_TOP = "cycle_top"       # 周期顶部
    CYCLE_BOTTOM = "cycle_bottom" # 周期底部
    UNKNOWN = "unknown"           # 未知

class MarketRegimeResult:
    """市场体制分析结果"""
    
    def __init__(self, 
                 regime_type: MarketRegimeType,
                 confidence: float,
                 sub_type: Optional[str] = None,
                 metrics: Optional[Dict] = None,
                 timestamp: Optional[datetime] = None):
        """
        初始化市场体制分析结果
        
        参数:
            regime_type: 市场体制类型
            confidence: 置信度 (0-1)
            sub_type: 子类型
            metrics: 相关指标
            timestamp: 时间戳
        """
        self.regime_type = regime_type
        self.confidence = max(0.0, min(1.0, confidence))  # 限制在0-1之间
        self.sub_type = sub_type
        self.metrics = metrics or {}
        self.timestamp = timestamp or datetime.now()
    
    def __str__(self):
        """字符串表示"""
        confidence_pct = int(self.confidence * 100)
        return f"市场体制: {self.regime_type.value} (置信度: {confidence_pct}%)" + \
               (f", 子类型: {self.sub_type}" if self.sub_type else "")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "regime_type": self.regime_type.value,
            "confidence": self.confidence,
            "sub_type": self.sub_type,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'MarketRegimeResult':
        """从字典创建对象"""
        regime_type = MarketRegimeType(data.get("regime_type", "unknown"))
        confidence = data.get("confidence", 0.0)
        sub_type = data.get("sub_type")
        metrics = data.get("metrics", {})
        timestamp = datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        
        return MarketRegimeResult(regime_type, confidence, sub_type, metrics, timestamp)

class MarketRegimeAnalyzer:
    """市场体制分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化市场体制分析器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 分析参数
        self.short_window = self.config.get("short_window", 20)   # 短期窗口
        self.medium_window = self.config.get("medium_window", 60)  # 中期窗口
        self.long_window = self.config.get("long_window", 120)    # 长期窗口
        self.vix_threshold = self.config.get("vix_threshold", 20)  # VIX阈值
        self.volume_ratio_threshold = self.config.get("volume_ratio_threshold", 1.5)  # 成交量比率阈值
        self.min_data_points = self.config.get("min_data_points", 30)  # 最小数据点
        
        # 分析结果缓存
        self.last_result = None
        self.cached_results = {}  # symbol -> result
        
        logger.info("市场体制分析器初始化完成")
    
    def analyze(self, market_data: Dict[str, pd.DataFrame], 
                index_symbol: Optional[str] = None) -> MarketRegimeResult:
        """
        分析市场体制
        
        参数:
            market_data: 市场数据 {symbol: dataframe}
            index_symbol: 指数符号 (如果不指定，将自动尝试查找主要指数)
            
        返回:
            MarketRegimeResult: 市场体制分析结果
        """
        # 尝试找到主要指数
        if index_symbol is None:
            for symbol in ["000001.SH", "399001.SZ", "^GSPC", "^DJI"]:
                if symbol in market_data:
                    index_symbol = symbol
                    break
        
        # 没有找到指数数据，尝试使用任意可用的数据
        if index_symbol is None or index_symbol not in market_data:
            if not market_data:
                logger.warning("没有市场数据可供分析")
                return MarketRegimeResult(MarketRegimeType.UNKNOWN, 0.0)
            
            index_symbol = list(market_data.keys())[0]
            logger.warning(f"未找到指数数据，使用 {index_symbol} 作为替代")
        
        # 获取指数数据
        index_data = market_data[index_symbol]
        
        # 检查数据是否足够
        if len(index_data) < self.min_data_points:
            logger.warning(f"数据点不足，需要至少 {self.min_data_points} 个数据点")
            return MarketRegimeResult(MarketRegimeType.UNKNOWN, 0.0)
        
        # 计算技术指标
        metrics = self._calculate_metrics(index_data)
        
        # 判断市场体制
        result = self._determine_market_regime(metrics)
        
        # 缓存结果
        self.last_result = result
        self.cached_results[index_symbol] = result
        
        return result
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """
        计算技术指标
        
        参数:
            data: 价格数据
            
        返回:
            Dict: 计算的指标
        """
        metrics = {}
        
        # 确保数据包含所需的列
        required_columns = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_columns):
            logger.warning("数据缺少必要的列")
            return metrics
        
        # 将数据排序为时间升序
        data = data.sort_index()
        
        # 计算收益率
        returns = data['close'].pct_change().dropna()
        metrics['returns'] = returns
        
        # 计算波动率 (20日年化标准差)
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        metrics['volatility'] = volatility.iloc[-1] if len(volatility) > 0 else None
        metrics['volatility_trend'] = self._calculate_trend(volatility.values)
        
        # 计算移动平均线
        metrics['ma_short'] = data['close'].rolling(window=self.short_window).mean().iloc[-1]
        metrics['ma_medium'] = data['close'].rolling(window=self.medium_window).mean().iloc[-1]
        metrics['ma_long'] = data['close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # 计算均线排列 (短期/中期/长期)
        ma_alignment = 0
        if metrics['ma_short'] > metrics['ma_medium'] > metrics['ma_long']:
            ma_alignment = 1  # 多头排列
        elif metrics['ma_short'] < metrics['ma_medium'] < metrics['ma_long']:
            ma_alignment = -1  # 空头排列
        metrics['ma_alignment'] = ma_alignment
        
        # 计算RSI
        delta = data['close'].diff().dropna()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        metrics['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if len(rs) > 0 else None
        
        # 计算MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        metrics['macd'] = macd.iloc[-1]
        metrics['macd_signal'] = signal.iloc[-1]
        metrics['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
        # 计算布林带
        sma = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        metrics['bollinger_upper'] = sma + (std * 2)
        metrics['bollinger_lower'] = sma - (std * 2)
        metrics['bollinger_width'] = (metrics['bollinger_upper'].iloc[-1] - metrics['bollinger_lower'].iloc[-1]) / sma.iloc[-1]
        
        # 计算ATR (平均真实范围)
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        metrics['atr'] = tr.rolling(window=14).mean().iloc[-1]
        
        # 计算动量 (收盘价变化率)
        metrics['momentum_10'] = (data['close'].iloc[-1] / data['close'].iloc[-10] - 1) if len(data) >= 10 else None
        metrics['momentum_20'] = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1) if len(data) >= 20 else None
        metrics['momentum_60'] = (data['close'].iloc[-1] / data['close'].iloc[-60] - 1) if len(data) >= 60 else None
        
        # 计算价格趋势
        metrics['price_trend'] = self._calculate_trend(data['close'].values)
        
        # 计算Hurst指数 (如果数据足够)
        if len(data) >= 100:
            try:
                metrics['hurst_exponent'] = self._calculate_hurst_exponent(data['close'].values)
            except:
                metrics['hurst_exponent'] = 0.5  # 默认为随机游走
        
        # 如果有成交量数据，计算相关指标
        if 'volume' in data.columns:
            metrics['volume_trend'] = self._calculate_trend(data['volume'].values)
            metrics['volume_ma'] = data['volume'].rolling(window=20).mean().iloc[-1]
            metrics['volume_ratio'] = data['volume'].iloc[-1] / metrics['volume_ma']
        
        return metrics
    
    def _calculate_trend(self, values: np.ndarray, window: int = 20) -> float:
        """
        计算趋势强度
        
        参数:
            values: 值序列
            window: 窗口大小
            
        返回:
            float: 趋势强度 (-1 到 1)
        """
        if len(values) < window:
            return 0.0
        
        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        slope, _ = np.polyfit(x, recent_values, 1)
        
        # 归一化斜率到 -1 到 1
        normalized_slope = np.tanh(slope * 100 / np.mean(recent_values))
        return normalized_slope
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        计算Hurst指数 (衡量时间序列的持续性)
        
        参数:
            prices: 价格序列
            
        返回:
            float: Hurst指数 (0-1)
                > 0.5: 持续性 (趋势)
                = 0.5: 随机游走
                < 0.5: 均值回归
        """
        # 计算对数收益率
        returns = np.diff(np.log(prices))
        if len(returns) < 50:
            return 0.5
        
        # 计算不同lags的R/S统计量
        lags = [2, 4, 8, 16, 32, 64]
        rs_values = []
        
        for lag in lags:
            if lag >= len(returns):
                continue
                
            # 将收益率分成多个子序列
            chunks = len(returns) // lag
            if chunks < 1:
                continue
                
            rs_array = np.zeros(chunks)
            
            for i in range(chunks):
                chunk = returns[i * lag:(i + 1) * lag]
                
                # 计算累积离差
                mean = np.mean(chunk)
                deviation = chunk - mean
                cumulative = np.cumsum(deviation)
                
                # 计算R/S值
                r = np.max(cumulative) - np.min(cumulative)  # 极差
                s = np.std(chunk)  # 标准差
                if s > 0:
                    rs_array[i] = r / s
                
            # 计算当前lag的平均R/S值
            rs_values.append(np.mean(rs_array))
        
        if not rs_values or len(rs_values) < 2:
            return 0.5
            
        # 使用log-log回归计算Hurst指数
        x = np.log(lags[:len(rs_values)])
        y = np.log(rs_values)
        slope, _ = np.polyfit(x, y, 1)
        
        return slope
    
    def _determine_market_regime(self, metrics: Dict) -> MarketRegimeResult:
        """
        根据技术指标确定市场体制
        
        参数:
            metrics: 技术指标
            
        返回:
            MarketRegimeResult: 市场体制分析结果
        """
        # 检查关键指标是否存在
        if 'ma_alignment' not in metrics or 'price_trend' not in metrics:
            return MarketRegimeResult(MarketRegimeType.UNKNOWN, 0.0)
        
        # 初始化评分
        scores = {regime_type: 0.0 for regime_type in MarketRegimeType}
        
        # 趋势判断
        price_trend = metrics.get('price_trend', 0)
        ma_alignment = metrics.get('ma_alignment', 0)
        
        # 牛市特征评分
        if ma_alignment > 0:  # 多头排列
            scores[MarketRegimeType.BULLISH] += 0.4
            scores[MarketRegimeType.TRENDING_UP] += 0.3
        
        if price_trend > 0.3:  # 强上升趋势
            scores[MarketRegimeType.BULLISH] += 0.3
            scores[MarketRegimeType.TRENDING_UP] += 0.4
        elif price_trend > 0.1:  # 弱上升趋势
            scores[MarketRegimeType.BULLISH] += 0.2
            scores[MarketRegimeType.TRENDING_UP] += 0.2
        
        # 熊市特征评分
        if ma_alignment < 0:  # 空头排列
            scores[MarketRegimeType.BEARISH] += 0.4
            scores[MarketRegimeType.TRENDING_DOWN] += 0.3
        
        if price_trend < -0.3:  # 强下降趋势
            scores[MarketRegimeType.BEARISH] += 0.3
            scores[MarketRegimeType.TRENDING_DOWN] += 0.4
        elif price_trend < -0.1:  # 弱下降趋势
            scores[MarketRegimeType.BEARISH] += 0.2
            scores[MarketRegimeType.TRENDING_DOWN] += 0.2
        
        # 震荡市评分
        volatility = metrics.get('volatility', 0)
        volatility_trend = metrics.get('volatility_trend', 0)
        
        if abs(price_trend) < 0.1:  # 无明显趋势
            scores[MarketRegimeType.SIDEWAYS] += 0.4
        
        if volatility > 0.25:  # 高波动
            scores[MarketRegimeType.VOLATILE] += 0.4
            if volatility_trend > 0.2:
                scores[MarketRegimeType.EXPANSION] += 0.3
        elif volatility < 0.15:  # 低波动
            scores[MarketRegimeType.SIDEWAYS] += 0.2
            if volatility_trend < -0.2:
                scores[MarketRegimeType.CONTRACTION] += 0.3
        
        # 反转评分
        macd_histogram = metrics.get('macd_histogram', 0)
        rsi = metrics.get('rsi', 50)
        
        if price_trend < -0.1 and macd_histogram > 0:  # 下跌趋势中MACD金叉
            scores[MarketRegimeType.REVERSAL_UP] += 0.3
        
        if price_trend > 0.1 and macd_histogram < 0:  # 上涨趋势中MACD死叉
            scores[MarketRegimeType.REVERSAL_DOWN] += 0.3
        
        if rsi < 30:  # 超卖
            scores[MarketRegimeType.REVERSAL_UP] += 0.2
            scores[MarketRegimeType.CYCLE_BOTTOM] += 0.3
        elif rsi > 70:  # 超买
            scores[MarketRegimeType.REVERSAL_DOWN] += 0.2
            scores[MarketRegimeType.CYCLE_TOP] += 0.3
        
        # 突破评分
        momentum_10 = metrics.get('momentum_10', 0)
        
        if momentum_10 > 0.05:  # 短期动量强劲
            scores[MarketRegimeType.BREAKOUT] += 0.3
        elif momentum_10 < -0.05:  # 短期动量显著下跌
            scores[MarketRegimeType.BREAKDOWN] += 0.3
        
        # 成交量判断
        volume_ratio = metrics.get('volume_ratio', 0)
        volume_trend = metrics.get('volume_trend', 0)
        
        if volume_ratio > self.volume_ratio_threshold and price_trend > 0:
            # 放量上涨
            scores[MarketRegimeType.BULLISH] += 0.1
            scores[MarketRegimeType.BREAKOUT] += 0.2
        elif volume_ratio > self.volume_ratio_threshold and price_trend < 0:
            # 放量下跌
            scores[MarketRegimeType.BEARISH] += 0.1
            scores[MarketRegimeType.BREAKDOWN] += 0.2
        
        # Hurst指数判断
        hurst = metrics.get('hurst_exponent', 0.5)
        
        if hurst > 0.6:  # 趋势持续性高
            if price_trend > 0:
                scores[MarketRegimeType.TRENDING_UP] += 0.2
            elif price_trend < 0:
                scores[MarketRegimeType.TRENDING_DOWN] += 0.2
        elif hurst < 0.4:  # 均值回归特性强
            scores[MarketRegimeType.SIDEWAYS] += 0.2
        
        # 找出得分最高的体制类型
        top_regime = max(scores.items(), key=lambda x: x[1])
        regime_type, confidence = top_regime
        
        # 如果最高分太低，标记为未知
        if confidence < 0.3:
            regime_type = MarketRegimeType.UNKNOWN
            confidence = 0.0
        
        # 确定子类型
        sub_type = None
        if regime_type in [MarketRegimeType.BULLISH, MarketRegimeType.BEARISH]:
            # 强中弱判断
            strength = abs(price_trend) * 2
            if strength > 0.7:
                sub_type = "strong"
            elif strength > 0.4:
                sub_type = "medium"
            else:
                sub_type = "weak"
        elif regime_type in [MarketRegimeType.REVERSAL_UP, MarketRegimeType.REVERSAL_DOWN]:
            # 反转确认度
            if abs(macd_histogram) > 0.02:
                sub_type = "confirmed"
            else:
                sub_type = "unconfirmed"
        
        # 返回结果
        return MarketRegimeResult(
            regime_type=regime_type,
            confidence=confidence,
            sub_type=sub_type,
            metrics={
                'price_trend': price_trend,
                'ma_alignment': ma_alignment,
                'volatility': metrics.get('volatility'),
                'rsi': metrics.get('rsi'),
                'macd_histogram': metrics.get('macd_histogram'),
                'hurst_exponent': metrics.get('hurst_exponent'),
                'volume_ratio': metrics.get('volume_ratio')
            }
        )
    
    def get_last_result(self) -> Optional[MarketRegimeResult]:
        """获取最近一次分析结果"""
        return self.last_result
    
    def get_result_for_symbol(self, symbol: str) -> Optional[MarketRegimeResult]:
        """获取特定符号的分析结果"""
        return self.cached_results.get(symbol)

# 测试函数
def test_market_regime_analyzer():
    """测试市场体制分析器"""
    
    # 创建模拟数据
    dates = pd.date_range(start='2025-01-01', periods=200)
    
    # 模拟牛市趋势
    bull_market = pd.DataFrame({
        'open': np.linspace(100, 150, 200) + np.random.normal(0, 2, 200),
        'high': np.linspace(102, 155, 200) + np.random.normal(1, 2, 200),
        'low': np.linspace(98, 145, 200) + np.random.normal(-1, 2, 200),
        'close': np.linspace(100, 150, 200) + np.random.normal(0, 1, 200),
        'volume': np.random.normal(1000000, 200000, 200) * (1 + np.linspace(0, 0.5, 200))
    }, index=dates)
    
    # 模拟熊市趋势
    bear_market = pd.DataFrame({
        'open': np.linspace(150, 100, 200) + np.random.normal(0, 2, 200),
        'high': np.linspace(152, 105, 200) + np.random.normal(1, 2, 200),
        'low': np.linspace(148, 95, 200) + np.random.normal(-1, 2, 200),
        'close': np.linspace(150, 100, 200) + np.random.normal(0, 1, 200),
        'volume': np.random.normal(1000000, 200000, 200) * (1 + np.linspace(0, 0.5, 200))
    }, index=dates)
    
    # 模拟盘整市场
    sideways_market = pd.DataFrame({
        'open': np.linspace(100, 100, 200) + np.random.normal(0, 3, 200),
        'high': np.linspace(100, 100, 200) + np.random.normal(3, 2, 200),
        'low': np.linspace(100, 100, 200) + np.random.normal(-3, 2, 200),
        'close': np.linspace(100, 100, 200) + np.random.normal(0, 2, 200),
        'volume': np.random.normal(800000, 150000, 200)
    }, index=dates)
    
    # 模拟高波动市场
    volatile_market = pd.DataFrame({
        'open': 100 + 15 * np.sin(np.linspace(0, 10, 200)) + np.random.normal(0, 5, 200),
        'high': 100 + 15 * np.sin(np.linspace(0, 10, 200)) + np.random.normal(5, 3, 200),
        'low': 100 + 15 * np.sin(np.linspace(0, 10, 200)) + np.random.normal(-5, 3, 200),
        'close': 100 + 15 * np.sin(np.linspace(0, 10, 200)) + np.random.normal(0, 3, 200),
        'volume': np.random.normal(1200000, 300000, 200) * (1.2 + 0.5 * np.sin(np.linspace(0, 10, 200)))
    }, index=dates)
    
    # 创建数据集合
    market_data = {
        "bull_market": bull_market,
        "bear_market": bear_market,
        "sideways_market": sideways_market,
        "volatile_market": volatile_market
    }
    
    # 创建分析器
    analyzer = MarketRegimeAnalyzer()
    
    # 分析各种市场类型
    for market_type, data in market_data.items():
        result = analyzer.analyze({market_type: data}, market_type)
        print(f"\n分析 {market_type}:")
        print(f"  {result}")
        print(f"  主要指标: ")
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")
    
    return analyzer

if __name__ == "__main__":
    # 设置日志输出
    logging.basicConfig(level=logging.INFO)
    
    # 测试市场体制分析器
    test_market_regime_analyzer() 