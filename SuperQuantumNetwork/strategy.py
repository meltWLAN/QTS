from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import logging
from event import EventType, MarketDataEvent, SignalEvent
from data import DataHandler
import pandas as pd

logger = logging.getLogger(__name__)

class Strategy(ABC):
    def __init__(self, data_handler: Optional[DataHandler] = None):
        self.data_handler = data_handler

    @abstractmethod
    def generate_signals(self, event: MarketDataEvent) -> List[SignalEvent]:
        """生成交易信号"""
        pass

class QuantumBurstStrategy(Strategy):
    """量子爆发策略 - 专注于识别短期爆发性上涨的股票"""
    
    def __init__(self, data_handler=None):
        super().__init__(data_handler)
        self.positions = {}  # 当前持仓
        self.position_hold_days = {}  # 持仓天数
        self.max_positions = 5  # 最大持仓数
        self.lookback_period = 10  # 回看天数
        self.price_threshold = 0.15  # 价格变动阈值 (15%)
        self.volume_threshold = 1.2  # 成交量放大阈值 (降低)
        self.signal_threshold = 0.2  # 信号强度阈值 (降低)
        self.stop_loss = 0.05  # 止损比例
        self.take_profit = 0.2  # 止盈比例
        self.position_hold_days_limit = 5  # 最大持仓天数
        self.min_days_for_signal = 3  # 最小观察天数 (3个交易日)
        
        # 量子特征权重
        self.quantum_weights = {
            'momentum': 0.7,  # 增加动量权重
            'coherence': 0.2,
            'entropy': 0.05,  # 降低熵权重
            'resonance': 0.05  # 降低共振权重
        }

    def generate_signals(self, event):
        """生成交易信号"""
        signals = []
        
        try:
            if event.type != EventType.MARKET_DATA:
                return signals
            
            for symbol, current_data in event.data.items():
                try:
                    # 如果已经持仓，检查是否需要平仓
                    if symbol in self.positions:
                        exit_signal = self._check_exit_signals(symbol, current_data)
                        if exit_signal:
                            signals.append(exit_signal)
                        continue
                        
                    # 如果持仓数量已达到上限，跳过
                    if len(self.positions) >= self.max_positions:
                        continue

                    # 获取历史数据
                    historical_data = self.data_handler.get_historical_data(symbol, self.lookback_period)
                    if len(historical_data) < self.lookback_period:
                        continue

                    # 计算3日价格变化
                    current_price = float(current_data['close'])
                    if len(historical_data) >= self.min_days_for_signal:
                        price_3days_ago = float(historical_data[-self.min_days_for_signal]['close'])
                        price_change_3days = (current_price - price_3days_ago) / price_3days_ago
                    else:
                        price_change_3days = 0
                        
                    # 计算当日价格变化
                    prev_price = float(historical_data[-2]['close'])
                    price_change = (current_price - prev_price) / prev_price

                    # 计算成交量变化
                    current_volume = float(current_data['volume'])
                    avg_volume = sum(float(d['volume']) for d in historical_data[:-1]) / (len(historical_data) - 1)
                    volume_change = current_volume / avg_volume if avg_volume > 0 else 0

                    # 计算量子特征
                    momentum = self._calculate_momentum(historical_data)
                    coherence = self._calculate_coherence(historical_data)
                    entropy = self._calculate_entropy(historical_data)
                    resonance = self._calculate_resonance(historical_data)

                    # 计算综合得分
                    quantum_score = (
                        self.quantum_weights['momentum'] * momentum +
                        self.quantum_weights['coherence'] * coherence +
                        self.quantum_weights['entropy'] * entropy +
                        self.quantum_weights['resonance'] * resonance
                    )

                    # 生成买入信号 - 主要条件是3日涨幅超过15%
                    if (price_change_3days > self.price_threshold and 
                        volume_change > self.volume_threshold and 
                        quantum_score > self.signal_threshold):
                        
                        signal = SignalEvent(
                            symbol=symbol,
                            datetime=event.datetime,
                            signal_type='LONG',
                            strength=quantum_score,
                            price=current_price
                        )
                        self.positions[symbol] = current_price
                        self.position_hold_days[symbol] = 0
                        signals.append(signal)
                        
                        logger.info(f"生成买入信号: {symbol}, 3日涨幅: {price_change_3days:.2%}, 量子得分: {quantum_score:.2f}")

                except Exception as e:
                    logger.error(f"生成 {symbol} 的信号时发生错误: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"生成信号时发生错误: {str(e)}")
            
        return signals

    def _calculate_momentum(self, data):
        """计算动量特征"""
        try:
            prices = [float(d['close']) for d in data]
            returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
            momentum = sum(returns) / len(returns) if returns else 0
            return min(max(momentum * 25, 0), 1)  # 放大25倍并限制在[0,1]范围内
        except:
            return 0

    def _calculate_coherence(self, data):
        """计算相干性特征"""
        try:
            prices = [float(d['close']) for d in data]
            ma3 = sum(prices[-3:]) / 3  # 使用3日均线
            ma5 = sum(prices[-5:]) / 5
            coherence = 1 - abs(ma3 - ma5) / ma5
            return min(max(coherence, 0), 1)
        except:
            return 0

    def _calculate_entropy(self, data):
        """计算熵特征"""
        try:
            prices = [float(d['close']) for d in data]
            returns = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
            positive_returns = len([r for r in returns if r > 0])
            entropy = positive_returns / len(returns) if returns else 0.5
            return min(max(entropy * 1.5, 0), 1)  # 降低放大倍数
        except:
            return 0

    def _calculate_resonance(self, data):
        """计算共振特征"""
        try:
            prices = [float(d['close']) for d in data]
            volumes = [float(d['volume']) for d in data]
            price_change = (prices[-1] - prices[0]) / prices[0]
            volume_change = (volumes[-1] - volumes[0]) / volumes[0]
            resonance = price_change * volume_change
            return min(max(resonance * 15, 0), 1)  # 放大15倍并限制在[0,1]范围内
        except:
            return 0

    def _check_exit_signals(self, symbol, current_data):
        """检查是否需要平仓"""
        try:
            entry_price = self.positions[symbol]
            current_price = float(current_data['close'])
            price_change = (current_price - entry_price) / entry_price
            
            # 更新持仓天数
            self.position_hold_days[symbol] += 1
            
            # 止损检查
            if price_change < -self.stop_loss:
                return self._create_exit_signal(symbol, current_price, "止损")
                
            # 止盈检查
            if price_change > self.take_profit:
                return self._create_exit_signal(symbol, current_price, "止盈")
                
            # 最大持仓期限检查
            if self.position_hold_days[symbol] >= self.position_hold_days_limit:
                return self._create_exit_signal(symbol, current_price, "超过持仓期限")
                
            return None
            
        except Exception as e:
            logger.error(f"检查平仓信号时发生错误 {symbol}: {str(e)}")
            return None
            
    def _create_exit_signal(self, symbol, price, reason):
        """创建平仓信号"""
        logger.info(f"生成平仓信号 - {symbol}: {reason}")
        # 清理持仓记录
        del self.positions[symbol]
        del self.position_hold_days[symbol]
        
        return SignalEvent(
            symbol=symbol,
            datetime=datetime.now(),
            signal_type='EXIT',
            strength=1.0,
            price=price
        )

    def update_positions(self, event):
        """更新持仓状态"""
        if event.type != EventType.MARKET_DATA:
            return

        for symbol in list(self.positions.keys()):
            try:
                current_data = event.data.get(symbol)
                if not current_data:
                    continue

                entry_price = self.positions[symbol]
                current_price = float(current_data['close'])
                returns = (current_price - entry_price) / entry_price
                days_held = self.position_hold_days[symbol]

                # 更新持仓天数
                self.position_hold_days[symbol] += 1

                # 检查是否需要平仓
                if (returns <= self.stop_loss or  # 止损
                    returns >= self.take_profit or  # 止盈
                    days_held >= self.position_hold_days_limit):  # 超过最大持仓天数
                    
                    # 生成卖出信号
                    signal = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        signal_type=-1,  # -1表示卖出信号
                        strength=1.0,
                        price=current_price
                    )
                    self.events_engine.put(signal)
                    
                    # 移除持仓记录
                    del self.positions[symbol]
                    del self.position_hold_days[symbol]

            except Exception as e:
                logging.error(f"更新 {symbol} 的持仓状态时发生错误: {str(e)}")

    def update_position(self, symbol: str, is_buy: bool):
        """更新持仓状态"""
        if is_buy:
            self.positions[symbol] = None  # 使用字典存储持仓状态
        else:
            if symbol in self.positions:
                del self.positions[symbol]
            
    def calculate_quantum_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算量子特征"""
        features = {}
        
        # 1. 动量特征 (量子动量)
        price_changes = data['close'].pct_change()
        features['momentum'] = abs(price_changes.mean()) * 20  # 进一步放大动量效应
        
        # 2. 相干性特征 (价格与成交量的协同性)
        volume_changes = data['volume'].pct_change()
        correlation = price_changes.corr(volume_changes)
        features['coherence'] = max(0, correlation)  # 只关注正相关
        
        # 3. 熵特征 (价格波动的无序程度)
        volatility = price_changes.std()
        features['entropy'] = 1 - min(1, volatility * 2)  # 降低熵的敏感度
        
        # 4. 共振特征 (多周期价格共振)
        ma5 = data['close'].rolling(5).mean()
        ma10 = data['close'].rolling(10).mean()
        resonance = (ma5.iloc[-1] > ma10.iloc[-1]).astype(int)  # 简化共振条件
        features['resonance'] = resonance
        
        return features
        
    def generate_signals(self, event: MarketDataEvent) -> List[SignalEvent]:
        """生成交易信号"""
        signals = []
        
        # 如果已达到最大持仓数量，不生成新信号
        if len(self.positions) >= self.max_positions:
            return signals
            
        for symbol, current_data in event.data.items():
            try:
                # 跳过已有持仓的标的
                if symbol in self.positions:
                    continue
                
                # 获取历史数据
                historical_data = self.data_handler.get_historical_data(
                    symbol,
                    (event.timestamp - timedelta(days=self.lookback_period)).strftime('%Y%m%d'),
                    event.timestamp.strftime('%Y%m%d')
                )
                
                if not historical_data or len(historical_data) < 2:
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(historical_data)
                
                # 计算量子特征
                quantum_features = self.calculate_quantum_features(df)
                
                # 计算价格变化
                current_price = float(current_data['close'])
                prev_price = float(df['close'].iloc[-2])
                price_change = (current_price - prev_price) / prev_price
                
                # 计算成交量变化
                current_volume = float(current_data['volume'])
                historical_volumes = df['volume'].iloc[:-1].astype(float)
                avg_volume = historical_volumes.mean()
                volume_change = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                # 计算量子加权得分
                quantum_score = sum(
                    quantum_features[feature] * weight 
                    for feature, weight in self.quantum_weights.items()
                )
                
                # 生成信号条件：
                # 1. 价格变化超过阈值
                # 2. 成交量显著放大
                # 3. 量子特征得分高
                if (price_change > self.price_threshold and 
                    volume_change > self.volume_threshold and 
                    quantum_score > self.signal_threshold):
                    
                    signal = SignalEvent(
                        type=EventType.SIGNAL,
                        symbol=symbol,
                        timestamp=event.timestamp,
                        signal_type="QUANTUM_BURST",
                        strength=quantum_score,
                        direction="BUY",
                        price=current_price
                    )
                    signals.append(signal)
                    self.update_position(symbol, True)
                    
                    logger.info(f"生成买入信号: {symbol}")
                    logger.info(f"- 价格变化: {price_change:.2%}")
                    logger.info(f"- 成交量变化: {volume_change:.2f}倍")
                    logger.info(f"- 量子得分: {quantum_score:.2f}")
                    logger.info(f"- 量子特征: {quantum_features}")
                    
            except Exception as e:
                logger.error(f"生成 {symbol} 的信号时发生错误: {str(e)}")
                continue
                
        return signals 