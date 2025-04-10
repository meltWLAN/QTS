#!/usr/bin/env python3
"""
中国市场分析核心模块
提供A股市场分析的核心功能
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import traceback
from enum import Enum, auto
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("ChinaMarketCore")

class MarketFactor(Enum):
    """市场因子"""
    POLICY_SENSITIVITY = auto()  # 政策敏感度
    LIQUIDITY_IMPACT = auto()    # 流动性影响
    SECTOR_ROTATION = auto()     # 板块轮动
    RETAIL_SENTIMENT = auto()    # 散户情绪
    INSTITUTIONAL_FLOW = auto()  # 机构资金流向
    REGULATORY_RISK = auto()     # 监管风险
    GLOBAL_CORRELATION = auto()  # 全球关联性
    FUNDAMENTAL_FACTOR = auto()  # 基本面因子
    TECHNICAL_PATTERN = auto()   # 技术形态
    VALUATION_LEVEL = auto()     # 估值水平

class MarketCycle(Enum):
    """市场周期"""
    POLICY_EASING = auto()       # 政策宽松期
    POLICY_TIGHTENING = auto()   # 政策收紧期
    GROWTH_LEADING = auto()      # 成长引领期
    VALUE_REVERTING = auto()     # 价值回归期
    RISK_AVERSION = auto()       # 风险规避期
    RISK_APPETITE = auto()       # 风险偏好期
    SECTOR_DIVERGENCE = auto()   # 板块分化期
    MARKET_INTEGRATION = auto()  # 市场一体化期

class ChinaMarketCore:
    """中国市场分析核心"""
    
    def __init__(self, config_path=None):
        """
        初始化中国市场分析核心
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger("ChinaMarketCore")
        self.logger.info("初始化中国市场分析核心...")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 市场状态记录
        self.market_state = {
            'current_cycle': MarketCycle.POLICY_EASING,
            'cycle_confidence': 0.6,
            'dominant_factors': [],
            'last_update': datetime.now().strftime('%Y-%m-%d'),
            'market_sentiment': 0.0,  # -1到1，负值表示悲观，正值表示乐观
            'divergence_level': 0.5,  # 0到1，表示市场分化程度
            'policy_direction': 0.2,  # -1到1，负值表示收紧，正值表示宽松
            'foreign_capital_flow': 0.0, # -1到1，负值表示流出，正值表示流入
            'anomaly_detected': False
        }
        
        # 因子重要性
        self.factor_importance = {factor.name: 1.0 for factor in MarketFactor}
        
        # 初始化子系统
        self.policy_analyzer = None
        self.sector_rotation_tracker = None
        self.regulation_monitor = None
        
        # 市场数据缓存
        self.market_data_cache = {}
        
        # 市场异常检测阈值
        self.anomaly_thresholds = {
            'volatility': 2.5,  # 波动率标准差倍数
            'volume_surge': 3.0,  # 成交量标准差倍数
            'sector_divergence': 0.8,  # 板块分化系数
            'policy_shift_magnitude': 0.6  # 政策转变幅度
        }
        
        self.market_data = None
        self.analysis_results = {}
        
        self.logger.info("中国市场分析核心初始化完成")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'data_update_frequency': 'daily',
            'use_quantum_perception': True,
            'policy_sensitivity_level': 'high',
            'sector_rotation_detection': True,
            'anomaly_detection_sensitivity': 0.7,
            'use_institutional_data': True,
            'use_alternative_data': True,
            'global_market_correlation': True,
            'retail_sentiment_weight': 0.6,
            'factor_learning_rate': 0.05
        }
        
        if not config_path or not os.path.exists(config_path):
            self.logger.info("使用默认配置")
            return default_config
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"配置文件加载成功: {config_path}")
            return {**default_config, **config}
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return default_config
    
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场状态"""
        try:
            self.market_data = data
            
            results = {
                'current_cycle': self._analyze_market_cycle(),
                'cycle_confidence': self._calculate_cycle_confidence(),
                'market_sentiment': self._analyze_market_sentiment(),
                'anomalies': self._detect_anomalies()
            }
            
            self.analysis_results = results
            return results
            
        except Exception as e:
            self.logger.error(f"市场分析失败: {str(e)}")
            return {}
            
    def _analyze_market_cycle(self) -> str:
        """分析市场周期"""
        try:
            if self.market_data is None or len(self.market_data) < 20:
                return "数据不足"
                
            # 计算技术指标
            ma20 = self.market_data['close'].rolling(window=20).mean()
            ma60 = self.market_data['close'].rolling(window=60).mean()
            
            # 计算趋势
            current_price = self.market_data['close'].iloc[-1]
            price_ma20 = ma20.iloc[-1]
            price_ma60 = ma60.iloc[-1]
            
            # 计算动量
            momentum = self.market_data['close'].pct_change(20).iloc[-1]
            
            # 判断周期
            if current_price > price_ma20 > price_ma60 and momentum > 0:
                return "上升期"
            elif current_price < price_ma20 < price_ma60 and momentum < 0:
                return "下降期"
            elif price_ma20 < current_price < price_ma60:
                return "盘整期"
            elif current_price < price_ma20 and momentum > -0.01:
                return "筑底期"
            elif current_price > price_ma20 and momentum < 0.01:
                return "顶部区域"
            else:
                return "积累期"
                
        except Exception as e:
            self.logger.error(f"周期分析失败: {str(e)}")
            return "未知"
            
    def _calculate_cycle_confidence(self) -> float:
        """计算周期判断的置信度"""
        try:
            if self.market_data is None:
                return 0.0
                
            # 计算多个指标的一致性
            indicators = []
            
            # 1. 趋势一致性
            ma5 = self.market_data['close'].rolling(window=5).mean()
            ma10 = self.market_data['close'].rolling(window=10).mean()
            ma20 = self.market_data['close'].rolling(window=20).mean()
            
            trend_consistency = 0
            if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
                trend_consistency = 1
            elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
                trend_consistency = 1
            indicators.append(trend_consistency)
            
            # 2. 成交量支撑
            volume_ma5 = self.market_data['volume'].rolling(window=5).mean()
            volume_ma20 = self.market_data['volume'].rolling(window=20).mean()
            
            volume_support = 0
            if volume_ma5.iloc[-1] > volume_ma20.iloc[-1]:
                volume_support = 1
            indicators.append(volume_support)
            
            # 3. 波动率稳定性
            returns = self.market_data['close'].pct_change()
            current_vol = returns.rolling(window=20).std().iloc[-1]
            hist_vol = returns.rolling(window=60).std().iloc[-1]
            
            volatility_stability = 1 - abs(current_vol - hist_vol) / hist_vol
            indicators.append(volatility_stability)
            
            # 计算综合置信度
            confidence = np.mean(indicators)
            return min(0.99, max(0.1, confidence))
            
        except Exception as e:
            self.logger.error(f"置信度计算失败: {str(e)}")
            return 0.5
            
    def _analyze_market_sentiment(self) -> float:
        """分析市场情绪"""
        try:
            if self.market_data is None:
                return 0.0
                
            # 计算多个情绪指标
            sentiments = []
            
            # 1. 价格动量
            returns = self.market_data['close'].pct_change()
            momentum = returns.rolling(window=20).mean().iloc[-1]
            sentiments.append(np.tanh(momentum * 100))  # 标准化到[-1,1]
            
            # 2. 成交量变化
            volume_change = self.market_data['volume'].pct_change()
            volume_momentum = volume_change.rolling(window=20).mean().iloc[-1]
            sentiments.append(np.tanh(volume_momentum * 50))
            
            # 3. 波动率情绪
            volatility = returns.rolling(window=20).std().iloc[-1]
            vol_sentiment = 1 - min(1, volatility * 100)  # 波动率越低，情绪越稳定
            sentiments.append(vol_sentiment * 2 - 1)  # 转换到[-1,1]
            
            # 4. 趋势强度
            ma20 = self.market_data['close'].rolling(window=20).mean()
            ma60 = self.market_data['close'].rolling(window=60).mean()
            trend_strength = (ma20.iloc[-1] / ma60.iloc[-1] - 1) * 10
            sentiments.append(np.tanh(trend_strength))
            
            # 计算加权平均情绪
            weights = [0.4, 0.2, 0.2, 0.2]  # 价格动量权重最大
            sentiment = np.average(sentiments, weights=weights)
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"情绪分析失败: {str(e)}")
            return 0.0
            
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测市场异常"""
        try:
            if self.market_data is None:
                return []
                
            anomalies = []
            
            # 1. 检测价格异常
            returns = self.market_data['close'].pct_change()
            std = returns.std()
            mean = returns.mean()
            
            latest_return = returns.iloc[-1]
            if abs(latest_return - mean) > 3 * std:  # 3倍标准差
                anomalies.append({
                    'type': '价格异常波动',
                    'position': '最新交易日',
                    'confidence': min(0.99, abs(latest_return - mean) / (4 * std))
                })
                
            # 2. 检测成交量异常
            volume_change = self.market_data['volume'].pct_change()
            vol_std = volume_change.std()
            vol_mean = volume_change.mean()
            
            latest_vol_change = volume_change.iloc[-1]
            if abs(latest_vol_change - vol_mean) > 3 * vol_std:
                anomalies.append({
                    'type': '成交量异常',
                    'position': '最新交易日',
                    'confidence': min(0.99, abs(latest_vol_change - vol_mean) / (4 * vol_std))
                })
                
            # 3. 检测背离
            price_ma5 = self.market_data['close'].rolling(window=5).mean()
            volume_ma5 = self.market_data['volume'].rolling(window=5).mean()
            
            price_trend = price_ma5.iloc[-1] > price_ma5.iloc[-5]  # 价格趋势
            volume_trend = volume_ma5.iloc[-1] > volume_ma5.iloc[-5]  # 成交量趋势
            
            if price_trend != volume_trend:
                anomalies.append({
                    'type': '量价背离',
                    'position': '近5个交易日',
                    'confidence': 0.8
                })
                
            # 4. 检测跳空缺口
            gaps = []
            for i in range(1, len(self.market_data)):
                prev_close = self.market_data['close'].iloc[i-1]
                curr_open = self.market_data['open'].iloc[i]
                
                gap_size = abs(curr_open - prev_close) / prev_close
                if gap_size > 0.02:  # 2%以上的缺口
                    gaps.append(gap_size)
                    
            if gaps and gaps[-1] > 0.03:  # 最新的大缺口
                anomalies.append({
                    'type': '跳空缺口',
                    'position': '最新交易日',
                    'confidence': min(0.99, gaps[-1] * 10)
                })
                
            return anomalies
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {str(e)}")
            return []
            
    def get_market_status(self) -> Dict[str, Any]:
        """获取市场状态"""
        if not self.analysis_results:
            return {}
        return self.analysis_results
        
    def calibrate(self):
        """校准分析参数"""
        self.logger.info("校准市场分析核心")
        # 实现校准逻辑
        
    def adjust_sensitivity(self):
        """调整灵敏度"""
        self.logger.info("调整市场分析核心灵敏度")
        # 实现灵敏度调整逻辑
    
    def _update_data_cache(self, market_data):
        """更新市场数据缓存"""
        # 提取当前日期
        if 'date' in market_data.columns:
            current_date = market_data['date'].max()
        else:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
        # 更新缓存
        self.market_data_cache[current_date] = market_data
        
        # 保持缓存大小在合理范围
        if len(self.market_data_cache) > 30:  # 保留30天数据
            oldest_date = min(self.market_data_cache.keys())
            del self.market_data_cache[oldest_date]
    
    def _calculate_market_indicators(self, market_data):
        """计算市场指标"""
        indicators = {}
        
        try:
            # 确保市场数据包含必要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in market_data.columns for col in required_columns):
                self.logger.warning("市场数据缺少必要的列")
                return indicators
            
            # 计算市场收益率
            if 'close' in market_data.columns:
                market_data['return'] = market_data['close'].pct_change()
                indicators['daily_return'] = market_data['return'].iloc[-1] if len(market_data) > 0 else 0
                indicators['5d_return'] = market_data['close'].pct_change(5).iloc[-1] if len(market_data) > 5 else 0
                indicators['20d_return'] = market_data['close'].pct_change(20).iloc[-1] if len(market_data) > 20 else 0
                
                # 计算波动率
                if len(market_data) >= 20:
                    indicators['volatility'] = market_data['return'].rolling(20).std().iloc[-1] * np.sqrt(252)
                
            # 计算成交量变化
            if 'volume' in market_data.columns and len(market_data) > 5:
                indicators['volume_change'] = market_data['volume'].iloc[-1] / market_data['volume'].iloc[-6:-1].mean() - 1
            
            # 计算技术指标
            if len(market_data) >= 30:
                # 计算RSI
                delta = market_data['close'].diff()
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = -delta.where(delta < 0, 0).fillna(0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
                
                # 计算MACD
                exp1 = market_data['close'].ewm(span=12, adjust=False).mean()
                exp2 = market_data['close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                indicators['macd'] = macd.iloc[-1]
                indicators['macd_signal'] = signal.iloc[-1]
                indicators['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
                
                # 趋势判断
                indicators['trend'] = self._determine_trend(market_data)
            
            # 计算A股特有指标
            indicators['market_breadth'] = 0.5  # 在实际中应从成分股涨跌家数计算
            indicators['northbound_flow'] = 0.0  # 在实际中应从沪深港通数据计算
            
            # 填充默认值
            default_indicators = {
                'daily_return': 0,
                'volatility': 0,
                'volume_change': 0,
                'rsi': 50,
                'trend': 'neutral',
                'market_breadth': 0.5,
                'northbound_flow': 0,
                'sector_divergence': 0.5
            }
            
            # 使用默认值填充缺失指标
            for key, value in default_indicators.items():
                if key not in indicators:
                    indicators[key] = value
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算市场指标失败: {str(e)}")
            traceback.print_exc()
            return indicators
    
    def _determine_trend(self, market_data):
        """判断市场趋势"""
        # 计算短期和长期移动平均
        ma20 = market_data['close'].rolling(20).mean().iloc[-1]
        ma60 = market_data['close'].rolling(60).mean().iloc[-1] if len(market_data) >= 60 else market_data['close'].mean()
        
        current_price = market_data['close'].iloc[-1]
        
        # 趋势判断
        if current_price > ma20 > ma60:
            return 'bullish'
        elif current_price < ma20 < ma60:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_sector_rotation(self, market_data):
        """分析板块轮动"""
        # 注意：在实际实现中，这里需要接入多个行业板块的数据
        # 以下是简化的示例实现
        
        sector_analysis = {
            'rotation_detected': False,
            'rotation_direction': 'none',
            'leading_sectors': [],
            'lagging_sectors': [],
            'rotation_strength': 0.0,
            'sector_divergence': 0.5
        }
        
        # 如果有板块轮动跟踪器，调用其分析方法
        if self.sector_rotation_tracker:
            detailed_analysis = self.sector_rotation_tracker.analyze()
            sector_analysis.update(detailed_analysis)
        
        return sector_analysis
    
    def _analyze_policy_sensitivity(self, market_data, additional_data=None):
        """分析政策敏感度"""
        policy_analysis = {
            'policy_direction': 0.0,  # -1到1，负表示收紧，正表示宽松
            'monetary_policy': 0.0,
            'fiscal_policy': 0.0,
            'regulatory_news': [],
            'policy_impact_sectors': [],
            'policy_uncertainty': 0.5
        }
        
        # 如果有政策分析器，调用其分析方法
        if self.policy_analyzer:
            detailed_analysis = self.policy_analyzer.analyze(additional_data)
            policy_analysis.update(detailed_analysis)
        
        # 根据附加数据更新政策方向
        if additional_data and 'policy_news' in additional_data:
            policy_news = additional_data['policy_news']
            # 在实际实现中，这里应该有一个基于NLP的政策新闻分析
            policy_analysis['policy_direction'] = 0.2  # 示例值
        
        return policy_analysis
    
    def _detect_market_anomalies(self, market_indicators):
        """检测市场异常"""
        anomalies = []
        
        # 检查波动率异常
        if 'volatility' in market_indicators and market_indicators['volatility'] > self.anomaly_thresholds['volatility']:
            anomalies.append({
                'type': 'volatility_surge',
                'value': market_indicators['volatility'],
                'threshold': self.anomaly_thresholds['volatility'],
                'severity': market_indicators['volatility'] / self.anomaly_thresholds['volatility']
            })
        
        # 检查成交量异常
        if 'volume_change' in market_indicators and market_indicators['volume_change'] > self.anomaly_thresholds['volume_surge']:
            anomalies.append({
                'type': 'volume_surge',
                'value': market_indicators['volume_change'],
                'threshold': self.anomaly_thresholds['volume_surge'],
                'severity': market_indicators['volume_change'] / self.anomaly_thresholds['volume_surge']
            })
        
        # 检查板块分化异常
        if 'sector_divergence' in market_indicators and market_indicators['sector_divergence'] > self.anomaly_thresholds['sector_divergence']:
            anomalies.append({
                'type': 'sector_divergence',
                'value': market_indicators['sector_divergence'],
                'threshold': self.anomaly_thresholds['sector_divergence'],
                'severity': market_indicators['sector_divergence'] / self.anomaly_thresholds['sector_divergence']
            })
        
        return anomalies
    
    def _identify_market_cycle(self, market_indicators, sector_analysis, policy_analysis):
        """识别当前市场周期"""
        cycle_scores = {cycle: 0.0 for cycle in MarketCycle}
        
        # 政策周期判断
        if policy_analysis['policy_direction'] > 0.3:
            cycle_scores[MarketCycle.POLICY_EASING] += 0.7
        elif policy_analysis['policy_direction'] < -0.3:
            cycle_scores[MarketCycle.POLICY_TIGHTENING] += 0.7
        
        # 市场风格判断
        if 'trend' in market_indicators:
            if market_indicators['trend'] == 'bullish' and market_indicators.get('rsi', 50) > 60:
                cycle_scores[MarketCycle.GROWTH_LEADING] += 0.6
                cycle_scores[MarketCycle.RISK_APPETITE] += 0.5
            elif market_indicators['trend'] == 'bearish' and market_indicators.get('rsi', 50) < 40:
                cycle_scores[MarketCycle.VALUE_REVERTING] += 0.6
                cycle_scores[MarketCycle.RISK_AVERSION] += 0.5
        
        # 板块分化判断
        if sector_analysis['sector_divergence'] > 0.7:
            cycle_scores[MarketCycle.SECTOR_DIVERGENCE] += 0.8
        elif sector_analysis['sector_divergence'] < 0.3:
            cycle_scores[MarketCycle.MARKET_INTEGRATION] += 0.6
        
        # 寻找得分最高的周期
        current_cycle = max(cycle_scores.items(), key=lambda x: x[1])
        
        # 如果最高分低于阈值，默认为政策宽松期
        if current_cycle[1] < 0.4:
            return MarketCycle.POLICY_EASING, 0.4
        
        return current_cycle[0], current_cycle[1]
    
    def _update_market_state(self, current_cycle, confidence, market_indicators):
        """更新市场状态"""
        self.market_state['current_cycle'] = current_cycle
        self.market_state['cycle_confidence'] = confidence
        self.market_state['last_update'] = datetime.now().strftime('%Y-%m-%d')
        
        # 更新市场情绪
        if 'rsi' in market_indicators:
            # 将RSI从0-100映射到-1到1
            self.market_state['market_sentiment'] = (market_indicators['rsi'] - 50) / 50
        
        # 更新政策方向
        if 'policy_direction' in market_indicators:
            self.market_state['policy_direction'] = market_indicators['policy_direction']
        
        # 更新分化程度
        if 'sector_divergence' in market_indicators:
            self.market_state['divergence_level'] = market_indicators['sector_divergence']
        
        # 更新资金流向
        if 'northbound_flow' in market_indicators:
            self.market_state['foreign_capital_flow'] = market_indicators['northbound_flow']
    
    def _determine_dominant_factors(self, market_indicators, sector_analysis, policy_analysis):
        """确定主导因子"""
        factor_scores = {}
        
        # 为每个因子评分
        # 政策敏感度
        factor_scores[MarketFactor.POLICY_SENSITIVITY.name] = abs(policy_analysis['policy_direction']) * 2.0
        
        # 流动性影响
        if 'volume_change' in market_indicators:
            factor_scores[MarketFactor.LIQUIDITY_IMPACT.name] = abs(market_indicators['volume_change']) * 1.5
        
        # 板块轮动
        factor_scores[MarketFactor.SECTOR_ROTATION.name] = sector_analysis['rotation_strength'] * 1.8
        
        # 散户情绪
        if 'rsi' in market_indicators:
            sentiment_impact = abs(market_indicators['rsi'] - 50) / 50
            factor_scores[MarketFactor.RETAIL_SENTIMENT.name] = sentiment_impact * 1.3
        
        # 机构资金流向
        if 'northbound_flow' in market_indicators:
            factor_scores[MarketFactor.INSTITUTIONAL_FLOW.name] = abs(market_indicators['northbound_flow']) * 2.0
        
        # 技术形态
        if 'trend' in market_indicators and market_indicators['trend'] != 'neutral':
            factor_scores[MarketFactor.TECHNICAL_PATTERN.name] = 0.8
        else:
            factor_scores[MarketFactor.TECHNICAL_PATTERN.name] = 0.2
        
        # 全球关联性
        factor_scores[MarketFactor.GLOBAL_CORRELATION.name] = 0.5  # 默认值
        
        # 估值水平
        factor_scores[MarketFactor.VALUATION_LEVEL.name] = 0.5  # 默认值
        
        # 应用因子重要性权重
        weighted_scores = {k: v * self.factor_importance.get(k, 1.0) for k, v in factor_scores.items()}
        
        # 选择前3个主导因子
        dominant_factors = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return [{'factor': factor, 'score': round(score, 2)} for factor, score in dominant_factors]
    
    def _adjust_factor_importance(self, market_indicators, anomalies):
        """根据市场表现调整因子重要性"""
        # 学习率
        learning_rate = self.config['factor_learning_rate']
        
        # 异常情况下提高相关因子的重要性
        for anomaly in anomalies:
            if anomaly['type'] == 'volatility_surge':
                self.factor_importance[MarketFactor.TECHNICAL_PATTERN.name] += learning_rate * 0.5
                self.factor_importance[MarketFactor.GLOBAL_CORRELATION.name] += learning_rate * 0.3
                
            elif anomaly['type'] == 'volume_surge':
                self.factor_importance[MarketFactor.LIQUIDITY_IMPACT.name] += learning_rate * 0.5
                self.factor_importance[MarketFactor.INSTITUTIONAL_FLOW.name] += learning_rate * 0.4
                
            elif anomaly['type'] == 'sector_divergence':
                self.factor_importance[MarketFactor.SECTOR_ROTATION.name] += learning_rate * 0.6
                self.factor_importance[MarketFactor.POLICY_SENSITIVITY.name] += learning_rate * 0.3
        
        # 根据市场情绪调整散户情绪因子的重要性
        if 'rsi' in market_indicators:
            sentiment_extremeness = abs(market_indicators['rsi'] - 50) / 50
            if sentiment_extremeness > 0.7:  # 情绪极端时提高其重要性
                self.factor_importance[MarketFactor.RETAIL_SENTIMENT.name] += learning_rate * 0.5
        
        # 确保因子重要性在合理范围内
        for factor in self.factor_importance:
            self.factor_importance[factor] = max(0.5, min(3.0, self.factor_importance[factor]))
    
    def _generate_recommendations(self, current_cycle, dominant_factors):
        """生成市场建议"""
        recommendations = []
        
        # 基于市场周期的建议
        cycle_recommendations = {
            MarketCycle.POLICY_EASING: "政策宽松期，关注政策受益板块，如基建、消费等",
            MarketCycle.POLICY_TIGHTENING: "政策收紧期，关注防御性板块，控制仓位",
            MarketCycle.GROWTH_LEADING: "成长引领期，关注科技、医药等高增长板块",
            MarketCycle.VALUE_REVERTING: "价值回归期，关注低估值蓝筹股和高股息板块",
            MarketCycle.RISK_AVERSION: "风险规避期，增加现金持有，关注债券和必需消费品",
            MarketCycle.RISK_APPETITE: "风险偏好期，可适当提高权益仓位，关注高beta板块",
            MarketCycle.SECTOR_DIVERGENCE: "板块分化期，精选行业龙头，关注超额收益",
            MarketCycle.MARKET_INTEGRATION: "市场一体化期，适合指数投资，降低个股风险"
        }
        
        if current_cycle in cycle_recommendations:
            recommendations.append({
                'type': 'cycle_based',
                'content': cycle_recommendations[current_cycle]
            })
        
        # 基于主导因子的建议
        for factor_info in dominant_factors:
            factor = factor_info['factor']
            
            if factor == MarketFactor.POLICY_SENSITIVITY.name:
                recommendations.append({
                    'type': 'factor_based',
                    'factor': factor,
                    'content': "政策敏感度高，密切关注政策变化，调整持仓结构"
                })
            
            elif factor == MarketFactor.LIQUIDITY_IMPACT.name:
                recommendations.append({
                    'type': 'factor_based',
                    'factor': factor,
                    'content': "流动性因素主导，关注市场资金面变化，顺势而为"
                })
            
            elif factor == MarketFactor.SECTOR_ROTATION.name:
                recommendations.append({
                    'type': 'factor_based',
                    'factor': factor,
                    'content': "板块轮动明显，适当减持滞涨板块，增持热点板块"
                })
            
            elif factor == MarketFactor.RETAIL_SENTIMENT.name:
                recommendations.append({
                    'type': 'factor_based',
                    'factor': factor,
                    'content': "散户情绪主导，警惕羊群效应，保持独立思考"
                })
            
            elif factor == MarketFactor.INSTITUTIONAL_FLOW.name:
                recommendations.append({
                    'type': 'factor_based',
                    'factor': factor,
                    'content': "机构资金流向重要，关注北向资金动向和机构重仓股"
                })
        
        return recommendations
    
    def register_policy_analyzer(self, analyzer):
        """注册政策分析器"""
        self.policy_analyzer = analyzer
        self.logger.info("政策分析器注册成功")
    
    def register_sector_rotation_tracker(self, tracker):
        """注册板块轮动跟踪器"""
        self.sector_rotation_tracker = tracker
        self.logger.info("板块轮动跟踪器注册成功")
    
    def register_regulation_monitor(self, monitor):
        """注册监管监控器"""
        self.regulation_monitor = monitor
        self.logger.info("监管监控器注册成功")
    
    def save_state(self, file_path):
        """
        保存当前状态
        
        参数:
            file_path: 保存路径
        """
        try:
            state = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_state': self.market_state,
                'factor_importance': self.factor_importance,
                'anomaly_thresholds': self.anomaly_thresholds,
                'config': self.config
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"状态已保存到: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_state(self, file_path):
        """
        加载状态
        
        参数:
            file_path: 加载路径
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"状态文件不存在: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 加载状态
            self.market_state = state['market_state']
            self.factor_importance = state['factor_importance']
            self.anomaly_thresholds = state['anomaly_thresholds']
            self.config.update(state['config'])
            
            self.logger.info(f"状态已从 {file_path} 加载")
            
            # 将市场周期从字符串转换为枚举
            if 'current_cycle' in self.market_state and isinstance(self.market_state['current_cycle'], str):
                try:
                    self.market_state['current_cycle'] = MarketCycle[self.market_state['current_cycle']]
                except KeyError:
                    self.market_state['current_cycle'] = MarketCycle.POLICY_EASING
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载状态失败: {str(e)}")
            traceback.print_exc()
            return False

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建中国市场分析核心
    market_core = ChinaMarketCore()
    
    # 输出诊断信息
    print("\n" + "="*60)
    print("超神量子共生系统 - 中国市场分析核心")
    print("="*60 + "\n")
    
    print("市场周期类型:")
    for cycle in MarketCycle:
        print(f"- {cycle.name}")
    
    print("\n市场因子:")
    for factor in MarketFactor:
        importance = market_core.factor_importance.get(factor.name, 1.0)
        print(f"- {factor.name}: 重要性 {importance:.2f}")
    
    print("\n当前市场状态:")
    print(f"- 周期: {market_core.market_state['current_cycle'].name}")
    print(f"- 置信度: {market_core.market_state['cycle_confidence']:.2f}")
    print(f"- 市场情绪: {market_core.market_state['market_sentiment']:.2f}")
    print(f"- 政策方向: {market_core.market_state['policy_direction']:.2f}")
    
    print("\n系统准备就绪，可以开始分析中国市场。")
    print("="*60 + "\n") 