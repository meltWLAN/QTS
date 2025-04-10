#!/usr/bin/env python3
"""
超神量子共生系统 - 交易信号生成器模块
负责生成交易信号和执行交易决策
"""

import os
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 设置日志
logger = logging.getLogger("TradingSignalGenerator")

class TradingSignalGenerator:
    """交易信号生成器基类"""
    
    def __init__(self, config=None):
        """初始化交易信号生成器
        
        参数:
            config: 配置字典
        """
        self.logger = logger
        self.config = config or {}
        self.name = "基础信号生成器"
        self.description = "基础交易信号生成算法"
        self.signals = []
        self.historical_signals = []  # 保存历史信号
        self.signal_performance = {}  # 信号表现评估
        self.accuracy_metrics = {     # 准确性指标
            'total': 0,
            'correct': 0,
            'false_positive': 0,
            'false_negative': 0,
            'accuracy': 0.0
        }
        self.initialized = False
        self.confidence_threshold = 0.6  # 信号置信度阈值
        
    def initialize(self):
        """初始化生成器"""
        self.initialized = True
        self.logger.info(f"{self.name}初始化完成")
        return True
        
    def generate_signals(self, data):
        """生成交易信号
        
        参数:
            data: 市场数据对象
            
        返回:
            list: 交易信号列表
        """
        if not self.initialized:
            self.initialize()
            
        self.logger.info(f"{self.name}开始生成交易信号")
        self.signals = []
        
        # 基础实现仅返回空列表
        # 子类应重写此方法实现具体信号生成逻辑
        
        return self.signals
        
    def get_signal_stats(self):
        """获取信号统计信息
        
        返回:
            dict: 信号统计信息
        """
        return {
            "generator": self.name,
            "total_signals": len(self.signals),
            "buy_signals": len([s for s in self.signals if s.get("action") == "BUY"]),
            "sell_signals": len([s for s in self.signals if s.get("action") == "SELL"]),
            "hold_signals": len([s for s in self.signals if s.get("action") == "HOLD"]),
        }

    def evaluate_signal_performance(self, signal, actual_result):
        """评估信号表现
        
        参数:
            signal: 信号字典
            actual_result: 实际结果（1表示成功，0表示失败）
        """
        if not self.initialized:
            self.initialize()
        
        self.accuracy_metrics['total'] += 1
        
        # 信号ID
        signal_id = signal.get('id', str(len(self.historical_signals)))
        
        # 记录信号表现
        if actual_result == 1:
            self.accuracy_metrics['correct'] += 1
            self.signal_performance[signal_id] = True
        else:
            if signal.get('action') in ['BUY', 'SELL']:
                self.accuracy_metrics['false_positive'] += 1
            else:
                self.accuracy_metrics['false_negative'] += 1
            self.signal_performance[signal_id] = False
        
        # 更新准确率
        if self.accuracy_metrics['total'] > 0:
            self.accuracy_metrics['accuracy'] = self.accuracy_metrics['correct'] / self.accuracy_metrics['total']
        
        # 将信号添加到历史记录
        self.historical_signals.append({
            'signal': signal,
            'result': actual_result,
            'evaluation_time': datetime.now().isoformat()
        })
        
        self.logger.info(f"信号{signal_id}评估完成，当前准确率: {self.accuracy_metrics['accuracy']:.2f}")
        
        return self.accuracy_metrics['accuracy']

    def filter_signals(self, signals, min_confidence=None):
        """基于置信度过滤信号
        
        参数:
            signals: 信号列表
            min_confidence: 最小置信度，默认使用self.confidence_threshold
            
        返回:
            list: 过滤后的信号列表
        """
        if min_confidence is None:
            min_confidence = self.confidence_threshold
        
        filtered_signals = [s for s in signals if s.get('confidence', 0) >= min_confidence]
        
        self.logger.info(f"信号过滤: {len(signals)} 个原始信号，{len(filtered_signals)} 个有效信号")
        return filtered_signals

# 尝试导入量子信号生成器
try:
    from trading_signals.quantum_signal_generator import QuantumSignalGenerator
    logger.info("成功导入量子信号生成器")
except ImportError:
    logger.warning("无法导入量子信号生成器，将使用基础信号生成器")
    
    class QuantumSignalGenerator(TradingSignalGenerator):
        """量子信号生成器"""
        
        def __init__(self, config=None):
            """初始化量子信号生成器"""
            super().__init__(config)
            self.name = "量子信号生成器"
            self.description = "基于量子原理的交易信号生成算法"
            
            # 量子信号特定参数
            self.quantum_noise_factor = self.config.get('quantum_noise_factor', 0.15)
            self.entanglement_threshold = self.config.get('entanglement_threshold', 0.75)
            self.phase_detection_window = self.config.get('phase_detection_window', 14)
            self.confidence_threshold = 0.7  # 更高的置信度阈值
            
            # 多指标配置
            self.indicators = {
                'trend_indicators': ['MA', 'EMA', 'DEMA'],
                'oscillator_indicators': ['RSI', 'MACD', 'Stochastic'],
                'volatility_indicators': ['Bollinger', 'ATR', 'Keltner'],
                'volume_indicators': ['OBV', 'Volume_ROC', 'CMF']
            }
            
            # 特殊量子指标权重
            self.quantum_weights = {
                'entanglement': 0.3,
                'superposition': 0.25,
                'phase_coherence': 0.2,
                'uncertainty': 0.15,
                'decoherence': 0.1
            }
        
        def initialize(self):
            """初始化生成器"""
            super().initialize()
            self.logger.info("初始化量子信号生成系统...")
            
            # 加载自适应参数（如果有）
            # 这里可以添加从配置文件或数据库加载历史训练好的参数
            
            return True
        
        def generate_signals(self, data):
            """生成交易信号
            
            参数:
                data: 市场数据对象
                
            返回:
                list: 交易信号列表
            """
            if not self.initialized:
                self.initialize()
                
            self.logger.info(f"{self.name}开始生成交易信号")
            self.signals = []
            
            if data is None or data.empty:
                self.logger.warning("提供的市场数据为空，无法生成信号")
                return self.signals
                
            try:
                # 数据准备和预处理
                data = self._prepare_data(data)
                
                # 1. 生成传统技术指标
                indicators = self._calculate_traditional_indicators(data)
                
                # 2. 计算量子特征
                quantum_features = self._calculate_quantum_features(data)
                
                # 3. 生成信号候选
                signal_candidates = self._generate_signal_candidates(data, indicators, quantum_features)
                
                # 4. 计算置信度并评分
                scored_signals = self._score_signals(signal_candidates, data)
                
                # 5. 过滤低置信度信号
                self.signals = self.filter_signals(scored_signals)
                
                # 日志记录
                self.logger.info(f"生成了 {len(self.signals)} 个交易信号，置信度阈值: {self.confidence_threshold}")
                
                # 打印信号详情
                for i, signal in enumerate(self.signals):
                    self.logger.info(f"信号 {i+1}: {signal['action']} {signal['symbol']} @ {signal['price']:.2f}, "
                                   f"置信度: {signal['confidence']:.2f}, 原因: {signal['reason']}")
                
            except Exception as e:
                self.logger.error(f"生成量子信号时出错: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
            return self.signals
        
        def _prepare_data(self, data):
            """预处理数据"""
            # 确保数据是副本，避免警告
            df = data.copy()
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"数据缺少必要的列: {missing_columns}")
                # 对缺失列进行估算
                if 'close' in df.columns:
                    for col in missing_columns:
                        if col == 'open' and 'close' in df.columns:
                            df['open'] = df['close']
                        elif col == 'high' and 'close' in df.columns:
                            df['high'] = df['close'] * 1.005  # 估算高点
                        elif col == 'low' and 'close' in df.columns:
                            df['low'] = df['close'] * 0.995   # 估算低点
                        elif col == 'volume':
                            df['volume'] = 1000000  # 默认成交量
            
            return df
        
        def _calculate_traditional_indicators(self, data):
            """计算传统技术指标"""
            df = data.copy()
            indicators = {}
            
            # 计算移动平均线
            windows = [5, 10, 20, 60, 120]
            for window in windows:
                indicators[f'MA{window}'] = df['close'].rolling(window=window).mean()
                indicators[f'EMA{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
            # 计算MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            indicators['MACD'] = ema12 - ema26
            indicators['Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
            indicators['Histogram'] = indicators['MACD'] - indicators['Signal']
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # 布林带
            indicators['MA20'] = df['close'].rolling(window=20).mean()
            indicators['STD20'] = df['close'].rolling(window=20).std()
            indicators['Upper_Band'] = indicators['MA20'] + (indicators['STD20'] * 2)
            indicators['Lower_Band'] = indicators['MA20'] - (indicators['STD20'] * 2)
            
            # 收集所有指标结果
            result = pd.DataFrame(indicators)
            return result
        
        def _calculate_quantum_features(self, data):
            """计算量子特征"""
            df = data.copy()
            features = {}
            
            # 1. 相位相干性 - 使用价格的自相关性衡量
            price_diff = df['close'].diff().dropna()
            if len(price_diff) > 1:
                autocorr = np.correlate(price_diff, price_diff, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                features['phase_coherence'] = autocorr[1:self.phase_detection_window+1] / autocorr[0]
            else:
                features['phase_coherence'] = np.zeros(self.phase_detection_window)
            
            # 2. 量子纠缠 - 使用价格和成交量的相关性
            if 'volume' in df.columns:
                price_norm = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min() + 1e-10)
                volume_norm = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min() + 1e-10)
                rolling_corr = price_norm.rolling(window=self.phase_detection_window).corr(volume_norm)
                features['entanglement'] = rolling_corr.abs()  # 取绝对值表示纠缠程度
            else:
                features['entanglement'] = pd.Series(0.5, index=df.index)  # 默认值
            
            # 3. 叠加态 - 使用波动性与趋势的平衡
            if all(col in df.columns for col in ['high', 'low', 'close']):
                # 计算波动率
                volatility = ((df['high'] - df['low']) / df['close']).rolling(window=10).std()
                # 计算趋势强度
                trend = df['close'].diff(5).abs() / df['close'].rolling(window=5).std()
                # 叠加态 - 两者之间的平衡
                features['superposition'] = (volatility * trend) / (volatility + trend + 1e-10)
            else:
                features['superposition'] = pd.Series(0.5, index=df.index)
            
            # 4. 量子不确定性 - 使用价格预测的困难度
            close_diff = df['close'].diff().dropna()
            if len(close_diff) > 5:
                # 计算布林带宽度，代表价格的不确定性
                ma = df['close'].rolling(window=20).mean()
                std = df['close'].rolling(window=20).std()
                features['uncertainty'] = std / ma  # 布林带宽度与均价的比例
            else:
                features['uncertainty'] = pd.Series(0.2, index=df.index)
            
            # 5. 退相干性 - 趋势突然中断的可能性
            features['decoherence'] = pd.Series(0, index=df.index)
            for i in range(10, len(df)):
                # 检测趋势突然逆转
                if i >= 10:
                    prev_trend = df['close'].iloc[i-10:i-5].mean() - df['close'].iloc[i-5:i].mean()
                    curr_trend = df['close'].iloc[i-5:i].mean() - df['close'].iloc[i:i+1].mean()
                    if prev_trend * curr_trend < 0:  # 趋势逆转
                        features['decoherence'].iloc[i] = abs(prev_trend - curr_trend) / (abs(prev_trend) + abs(curr_trend) + 1e-10)
                    
            # 将特征转为DataFrame
            result = pd.DataFrame(features)
            return result
        
        def _generate_signal_candidates(self, data, indicators, quantum_features):
            """生成信号候选"""
            candidates = []
            df = data.copy()
            
            if len(df) < 20:  # 确保有足够的数据
                return candidates
            
            # 最后一根K线
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # 获取当前量子特征
            current_quantum = {k: v.iloc[-1] if isinstance(v, pd.Series) else v[-1] for k, v in quantum_features.items()}
            
            # 检查各种信号条件
            
            # 1. MACD金叉信号
            if (indicators['MACD'].iloc[-2] < indicators['Signal'].iloc[-2] and 
                indicators['MACD'].iloc[-1] > indicators['Signal'].iloc[-1]):
                macd_confidence = min(0.5 + abs(indicators['Histogram'].iloc[-1]) / 0.5, 0.9)
                candidates.append({
                    'symbol': current.get('code', df.index[-1]),
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                    'action': 'BUY',
                    'price': current['close'],
                    'reason': 'MACD金叉信号',
                    'source': self.name,
                    'confidence': macd_confidence,
                    'quantum_influence': current_quantum
                })
            
            # 2. MACD死叉信号
            if (indicators['MACD'].iloc[-2] > indicators['Signal'].iloc[-2] and 
                indicators['MACD'].iloc[-1] < indicators['Signal'].iloc[-1]):
                macd_confidence = min(0.5 + abs(indicators['Histogram'].iloc[-1]) / 0.5, 0.9)
                candidates.append({
                    'symbol': current.get('code', df.index[-1]),
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                    'action': 'SELL',
                    'price': current['close'],
                    'reason': 'MACD死叉信号',
                    'source': self.name,
                    'confidence': macd_confidence,
                    'quantum_influence': current_quantum
                })
            
            # 3. RSI超买超卖信号
            if 'RSI' in indicators and not indicators['RSI'].iloc[-1] != indicators['RSI'].iloc[-1]:  # 检查不是NaN
                # RSI超卖信号 (RSI < 30)
                if indicators['RSI'].iloc[-1] < 30:
                    rsi_confidence = 0.5 + (30 - indicators['RSI'].iloc[-1]) / 60
                    candidates.append({
                        'symbol': current.get('code', df.index[-1]),
                        'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                        'action': 'BUY',
                        'price': current['close'],
                        'reason': 'RSI超卖信号',
                        'source': self.name,
                        'confidence': rsi_confidence,
                        'quantum_influence': current_quantum
                    })
                
                # RSI超买信号 (RSI > 70)
                if indicators['RSI'].iloc[-1] > 70:
                    rsi_confidence = 0.5 + (indicators['RSI'].iloc[-1] - 70) / 60
                    candidates.append({
                        'symbol': current.get('code', df.index[-1]),
                        'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                        'action': 'SELL',
                        'price': current['close'],
                        'reason': 'RSI超买信号',
                        'source': self.name,
                        'confidence': rsi_confidence,
                        'quantum_influence': current_quantum
                    })
            
            # 4. 布林带突破信号
            if 'Upper_Band' in indicators and 'Lower_Band' in indicators:
                # 价格突破上轨
                if previous['close'] <= indicators['Upper_Band'].iloc[-2] and current['close'] > indicators['Upper_Band'].iloc[-1]:
                    bb_confidence = 0.6 + min((current['close'] - indicators['Upper_Band'].iloc[-1]) / current['close'], 0.3)
                    candidates.append({
                        'symbol': current.get('code', df.index[-1]),
                        'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                        'action': 'SELL',
                        'price': current['close'],
                        'reason': '布林带上轨突破',
                        'source': self.name,
                        'confidence': bb_confidence,
                        'quantum_influence': current_quantum
                    })
                
                # 价格突破下轨
                if previous['close'] >= indicators['Lower_Band'].iloc[-2] and current['close'] < indicators['Lower_Band'].iloc[-1]:
                    bb_confidence = 0.6 + min((indicators['Lower_Band'].iloc[-1] - current['close']) / current['close'], 0.3)
                    candidates.append({
                        'symbol': current.get('code', df.index[-1]),
                        'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                        'action': 'BUY',
                        'price': current['close'],
                        'reason': '布林带下轨突破',
                        'source': self.name,
                        'confidence': bb_confidence,
                        'quantum_influence': current_quantum
                    })
            
            # 5. 量子特有信号 - 相位相干性高 + 纠缠度高
            if (current_quantum.get('phase_coherence', 0) > 0.8 and 
                current_quantum.get('entanglement', 0) > self.entanglement_threshold):
                # 判断方向 - 使用短期趋势
                short_trend = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
                action = 'BUY' if short_trend > 0 else 'SELL'
                q_confidence = (current_quantum.get('phase_coherence', 0) + current_quantum.get('entanglement', 0)) / 2
                candidates.append({
                    'symbol': current.get('code', df.index[-1]),
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                    'action': action,
                    'price': current['close'],
                    'reason': '量子相干性+纠缠信号',
                    'source': self.name,
                    'confidence': q_confidence,
                    'quantum_influence': current_quantum
                })
                
            # 6. 量子退相干预警信号
            if current_quantum.get('decoherence', 0) > 0.6:
                # 这是一个强烈的趋势反转预警
                current_ma5 = df['close'].iloc[-5:].mean()
                current_ma20 = df['close'].iloc[-20:].mean()
                action = 'SELL' if current_ma5 > current_ma20 else 'BUY'  # 反转当前趋势
                candidates.append({
                    'symbol': current.get('code', df.index[-1]),
                    'timestamp': df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                    'action': action,
                    'price': current['close'],
                    'reason': '量子退相干预警',
                    'source': self.name,
                    'confidence': current_quantum.get('decoherence', 0),
                    'quantum_influence': current_quantum
                })
            
            return candidates
        
        def _score_signals(self, signal_candidates, data):
            """对信号进行评分"""
            scored_signals = []
            
            for signal in signal_candidates:
                # 获取信号原始置信度
                base_confidence = signal.get('confidence', 0.5)
                
                # 获取量子影响因素
                quantum_influence = signal.get('quantum_influence', {})
                
                # 计算量子加权评分
                quantum_score = 0
                for feature, weight in self.quantum_weights.items():
                    if feature in quantum_influence:
                        quantum_score += quantum_influence[feature] * weight
                
                # 量子调整后的最终置信度
                final_confidence = base_confidence * 0.7 + quantum_score * 0.3
                final_confidence = min(max(final_confidence, 0), 1)  # 确保在0-1范围内
                
                # 生成唯一ID
                signal_id = f"{signal['symbol']}_{signal['action']}_{int(datetime.now().timestamp())}"
                
                # 创建最终信号
                final_signal = signal.copy()
                final_signal['confidence'] = final_confidence
                final_signal['id'] = signal_id
                final_signal['time_generated'] = datetime.now().isoformat()
                
                scored_signals.append(final_signal)
            
            # 按置信度排序
            scored_signals.sort(key=lambda x: x['confidence'], reverse=True)
            return scored_signals

# 创建工厂函数用于获取信号生成器实例
def get_signal_generator(generator_type="basic", config=None):
    """获取信号生成器实例
    
    参数:
        generator_type: 生成器类型，支持 "basic", "quantum"
        config: 配置字典
        
    返回:
        TradingSignalGenerator: 信号生成器实例
    """
    if generator_type.lower() == "quantum":
        return QuantumSignalGenerator(config)
    else:
        return TradingSignalGenerator(config)

# 测试函数
def test_signal_generator():
    """测试信号生成器"""
    # 创建模拟数据
    dates = pd.date_range(start='2025-01-01', periods=30)
    data = pd.DataFrame({
        'open': np.random.normal(100, 2, 30),
        'high': np.random.normal(102, 2, 30),
        'low': np.random.normal(98, 2, 30),
        'close': np.random.normal(101, 2, 30),
        'volume': np.random.normal(10000, 1000, 30),
        'code': ['000001.SZ'] * 30
    }, index=dates)
    
    # 测试基础信号生成器
    basic_gen = get_signal_generator("basic")
    basic_signals = basic_gen.generate_signals(data)
    print(f"基础信号生成器生成信号数量: {len(basic_signals)}")
    
    # 测试量子信号生成器
    quantum_gen = get_signal_generator("quantum")
    quantum_signals = quantum_gen.generate_signals(data)
    print(f"量子信号生成器生成信号数量: {len(quantum_signals)}")
    
    return basic_signals, quantum_signals

if __name__ == "__main__":
    # 设置日志输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("测试交易信号生成器...")
    basic_signals, quantum_signals = test_signal_generator()
    print("测试完成!") 