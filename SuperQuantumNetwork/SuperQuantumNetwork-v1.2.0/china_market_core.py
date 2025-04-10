#!/usr/bin/env python3
"""
超神量子共生系统 - 中国市场分析核心
专为中国A股市场特性设计的高维分析引擎
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import json
import traceback
from enum import Enum, auto
try:
    from numba import jit, njit, prange, NumbaWarning
    from sklearn.ensemble import IsolationForest
    from scipy import signal
    import statsmodels.api as sm
    NUMBA_AVAILABLE = True
    ML_AVAILABLE = True
    import warnings
    warnings.filterwarnings('ignore', category=NumbaWarning)
except ImportError as e:
    # 定义空装饰器，在没有Numba时不会影响代码执行
    def jit(nopython=True, parallel=False):
        def decorator(func):
            return func
        return decorator
        
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    def prange(*args):
        return range(*args)
    
    NUMBA_AVAILABLE = False
    ML_AVAILABLE = False
    print(f"高级功能受限: {str(e)}")

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
            'policy_shift_magnitude': 0.6,  # 政策转变幅度
            'chaos_metric': 0.75,  # 混沌度量阈值
            'fractality': 0.7,   # 分形特征阈值
            'turning_point': 0.8  # 拐点概率阈值
        }
        
        # 高级市场预警系统
        self.advanced_warning_system = {
            'enabled': True,
            'history_length': 120,  # 历史数据长度
            'warning_signals': [],  # 当前预警信号
            'warning_history': [],  # 历史预警记录
            'turning_points': [],   # 检测到的拐点
            'ml_models': {}         # 机器学习模型
        }
        
        # 市场结构分析
        self.market_structure = {
            'fractal_dimension': 0.0,  # 分形维度
            'hurst_exponent': 0.5,     # 赫斯特指数
            'entropy': 0.0,            # 市场熵值
            'complexity': 0.0,         # 复杂度
            'regime': 'unknown',       # 市场结构状态
            'stability': 0.5           # 稳定性
        }
        
        # 初始化异常检测模型
        if ML_AVAILABLE:
            self._initialize_anomaly_detection()
        
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
    
    def analyze_market(self, market_data, additional_data=None):
        """
        分析中国市场状态
        
        参数:
            market_data: DataFrame, 市场数据
            additional_data: dict, 附加数据
            
        返回:
            dict: 市场分析结果
        """
        self.logger.info("开始中国市场分析...")
        
        try:
            # 更新市场数据缓存
            self._update_data_cache(market_data)
            
            # 计算市场指标
            market_indicators = self._calculate_market_indicators(market_data)
            
            # 分析板块轮动
            sector_analysis = self._analyze_sector_rotation(market_data)
            
            # 政策敏感度分析
            policy_analysis = self._analyze_policy_sensitivity(market_data, additional_data)
            
            # 标准市场异常检测
            anomalies = self._detect_market_anomalies(market_indicators)
            
            # 高级市场预警检测
            if self.advanced_warning_system['enabled']:
                warnings = self._run_advanced_warning_system(market_data)
                # 合并异常和预警
                anomalies.extend(warnings)
                
                # 分析市场结构
                self._analyze_market_structure(market_data)
            
            # 识别当前市场周期
            current_cycle, confidence = self._identify_market_cycle(
                market_indicators,
                sector_analysis,
                policy_analysis
            )
            
            # 更新市场状态
            self._update_market_state(current_cycle, confidence, market_indicators)
            
            # 确定主导因子
            dominant_factors = self._determine_dominant_factors(
                market_indicators,
                sector_analysis,
                policy_analysis
            )
            
            # 根据分析调整因子重要性
            self._adjust_factor_importance(market_indicators, anomalies)
            
            # 整合分析结果
            analysis_result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_cycle': current_cycle.name,
                'cycle_confidence': confidence,
                'market_sentiment': self.market_state['market_sentiment'],
                'dominant_factors': dominant_factors,
                'policy_direction': self.market_state['policy_direction'],
                'divergence_level': self.market_state['divergence_level'],
                'sector_rotation': sector_analysis['rotation_direction'],
                'detected_anomalies': anomalies,
                'factor_importance': {k: round(v, 2) for k, v in self.factor_importance.items()},
                'recommendations': self._generate_recommendations(current_cycle, dominant_factors),
                'market_structure': self.market_structure if self.advanced_warning_system['enabled'] else None,
                'warning_signals': self.advanced_warning_system['warning_signals'] if self.advanced_warning_system['enabled'] else None
            }
            
            self.logger.info(f"市场分析完成，当前周期: {current_cycle.name}，置信度: {confidence:.2f}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"市场分析失败: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}
    
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
    
    def update_market_data(self, market_data):
        """
        更新市场数据
        
        参数:
            market_data: DataFrame, 市场数据
        """
        self.logger.info("更新市场数据...")
        try:
            # 更新市场数据缓存
            self._update_data_cache(market_data)
            self.market_data = market_data
            self.logger.info("市场数据更新成功")
        except Exception as e:
            self.logger.error(f"市场数据更新失败: {str(e)}")
            traceback.print_exc()
    
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
                # 将数据转换为NumPy数组以加速计算
                close_prices = market_data['close'].values
                
                if len(close_prices) > 0:
                    indicators['daily_return'] = self._calculate_return(close_prices, 1)
                    
                if len(close_prices) > 5:
                    indicators['5d_return'] = self._calculate_return(close_prices, 5)
                    
                if len(close_prices) > 20:
                    indicators['20d_return'] = self._calculate_return(close_prices, 20)
                    
                    # 计算波动率
                    if len(market_data) >= 20:
                        returns = np.diff(close_prices) / close_prices[:-1]
                        indicators['volatility'] = self._calculate_volatility(returns, 20) * np.sqrt(252)
                
            # 计算成交量变化
            if 'volume' in market_data.columns and len(market_data) > 5:
                volume_data = market_data['volume'].values
                indicators['volume_change'] = self._calculate_volume_change(volume_data)
            
            # 计算技术指标
            if len(market_data) >= 30:
                close_prices = market_data['close'].values
                
                # 计算RSI
                if len(close_prices) > 14:
                    delta = np.diff(close_prices)
                    indicators['rsi'] = self._calculate_rsi(delta, 14)
                
                # 计算MACD
                if len(close_prices) > 26:
                    macd, signal, hist = self._calculate_macd(close_prices)
                    indicators['macd'] = macd
                    indicators['macd_signal'] = signal
                    indicators['macd_histogram'] = hist
                
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

    # 使用Numba优化的数值计算函数
    @staticmethod
    @njit
    def _calculate_return(prices, period):
        """
        使用Numba加速的收益率计算
        
        参数:
            prices: ndarray, 价格序列
            period: int, 周期
            
        返回:
            float: 收益率
        """
        if len(prices) <= period:
            return 0.0
            
        return prices[-1] / prices[-period-1] - 1
    
    @staticmethod
    @njit
    def _calculate_volatility(returns, window):
        """
        使用Numba加速的波动率计算
        
        参数:
            returns: ndarray, 收益率序列
            window: int, 窗口期
            
        返回:
            float: 波动率
        """
        if len(returns) < window:
            return 0.0
            
        # 使用滚动窗口计算标准差
        recent_returns = returns[-window:]
        return np.std(recent_returns)
    
    @staticmethod
    @njit
    def _calculate_volume_change(volume):
        """
        使用Numba加速的成交量变化计算
        
        参数:
            volume: ndarray, 成交量序列
            
        返回:
            float: 成交量变化率
        """
        if len(volume) < 6:
            return 0.0
            
        current = volume[-1]
        prev_avg = np.mean(volume[-6:-1])
        
        if prev_avg > 0:
            return current / prev_avg - 1
        else:
            return 0.0
    
    @staticmethod
    @njit
    def _calculate_rsi(delta, window):
        """
        使用Numba加速的RSI计算
        
        参数:
            delta: ndarray, 价格变化序列
            window: int, RSI窗口
            
        返回:
            float: RSI值
        """
        up = np.zeros(len(delta))
        down = np.zeros(len(delta))
        
        for i in range(len(delta)):
            if delta[i] > 0:
                up[i] = delta[i]
            else:
                down[i] = -delta[i]
        
        # 计算平均上涨和下跌
        up_avg = np.mean(up[-window:])
        down_avg = np.mean(down[-window:])
        
        if down_avg == 0:
            return 100.0
            
        rs = up_avg / down_avg
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    @njit
    def _calculate_macd(prices):
        """
        使用Numba加速的MACD计算
        
        参数:
            prices: ndarray, 价格序列
            
        返回:
            tuple: MACD线, 信号线, 柱状图
        """
        # 计算EMA
        ema12 = np.zeros_like(prices)
        ema26 = np.zeros_like(prices)
        
        # 初始值
        ema12[0] = prices[0]
        ema26[0] = prices[0]
        
        # EMA计算系数
        k12 = 2.0 / (12 + 1)
        k26 = 2.0 / (26 + 1)
        k9 = 2.0 / (9 + 1)
        
        # 计算12日和26日EMA
        for i in range(1, len(prices)):
            ema12[i] = prices[i] * k12 + ema12[i-1] * (1 - k12)
            ema26[i] = prices[i] * k26 + ema26[i-1] * (1 - k26)
        
        # 计算MACD线
        macd_line = ema12 - ema26
        
        # 计算信号线（MACD的9日EMA）
        signal_line = np.zeros_like(macd_line)
        signal_line[0] = macd_line[0]
        
        for i in range(1, len(macd_line)):
            signal_line[i] = macd_line[i] * k9 + signal_line[i-1] * (1 - k9)
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line[-1], signal_line[-1], histogram[-1]
    
    @staticmethod
    @njit(parallel=True)
    def _calculate_correlation_matrix(returns_matrix):
        """
        使用Numba加速的相关性矩阵计算
        
        参数:
            returns_matrix: ndarray, 形状为(n_assets, n_periods)的收益率矩阵
            
        返回:
            ndarray: 相关性矩阵
        """
        n_assets = returns_matrix.shape[0]
        correlation_matrix = np.zeros((n_assets, n_assets))
        
        # 计算每个资产的标准差
        stds = np.zeros(n_assets)
        for i in range(n_assets):
            stds[i] = np.std(returns_matrix[i])
        
        # 并行计算相关性矩阵上三角部分
        for i in prange(n_assets):
            for j in range(i, n_assets):
                if stds[i] == 0 or stds[j] == 0:
                    correlation_matrix[i, j] = 0.0
                else:
                    # 计算协方差
                    cov = np.mean((returns_matrix[i] - np.mean(returns_matrix[i])) * 
                                  (returns_matrix[j] - np.mean(returns_matrix[j])))
                    # 计算相关系数
                    correlation_matrix[i, j] = cov / (stds[i] * stds[j])
                    correlation_matrix[j, i] = correlation_matrix[i, j]  # 对称填充
        
        return correlation_matrix

    def _initialize_anomaly_detection(self):
        """初始化异常检测模型"""
        try:
            # 初始化隔离森林模型用于异常检测
            self.advanced_warning_system['ml_models']['isolation_forest'] = IsolationForest(
                n_estimators=100, 
                contamination=0.05,  # 预期异常比例
                random_state=42
            )
            self.logger.info("异常检测模型初始化完成")
        except Exception as e:
            self.logger.error(f"初始化异常检测模型失败: {str(e)}")
    
    def _run_advanced_warning_system(self, market_data):
        """
        运行高级市场预警系统
        
        参数:
            market_data: DataFrame, 市场数据
            
        返回:
            list: 预警信号列表
        """
        warnings = []
        
        try:
            # 清除旧的预警信号
            self.advanced_warning_system['warning_signals'] = []
            
            # 检测拐点
            turning_points = self._detect_turning_points(market_data)
            if turning_points:
                for tp in turning_points:
                    warnings.append({
                        'type': 'turning_point',
                        'value': tp['confidence'],
                        'description': tp['description'],
                        'threshold': self.anomaly_thresholds['turning_point'],
                        'severity': tp['confidence'] / self.anomaly_thresholds['turning_point']
                    })
                    # 添加到预警信号
                    self.advanced_warning_system['warning_signals'].append({
                        'type': 'turning_point',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'description': tp['description'],
                        'confidence': tp['confidence']
                    })
            
            # 非线性混沌检测
            if len(market_data) >= 50:
                chaos_metrics = self._calculate_chaos_metrics(market_data)
                if chaos_metrics['lyapunov_exponent'] > 0.2:  # 正Lyapunov指数表示混沌行为
                    chaos_severity = chaos_metrics['lyapunov_exponent'] / self.anomaly_thresholds['chaos_metric']
                    warnings.append({
                        'type': 'chaos_detected',
                        'value': chaos_metrics['lyapunov_exponent'],
                        'description': f"检测到市场混沌行为，Lyapunov指数：{chaos_metrics['lyapunov_exponent']:.3f}",
                        'threshold': self.anomaly_thresholds['chaos_metric'],
                        'severity': chaos_severity
                    })
                    # 添加到预警信号
                    self.advanced_warning_system['warning_signals'].append({
                        'type': 'chaos_warning',
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'description': "市场进入混沌状态，波动可能加剧",
                        'confidence': min(chaos_severity, 1.0)
                    })
            
            # 机器学习异常检测
            if ML_AVAILABLE and len(market_data) >= 30:
                ml_anomalies = self._detect_ml_anomalies(market_data)
                warnings.extend(ml_anomalies)
            
            # 历史记录
            if warnings:
                self.advanced_warning_system['warning_history'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'warnings': warnings
                })
                # 限制历史记录长度
                if len(self.advanced_warning_system['warning_history']) > 50:
                    self.advanced_warning_system['warning_history'] = self.advanced_warning_system['warning_history'][-50:]
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"高级预警系统运行失败: {str(e)}")
            traceback.print_exc()
            return []
    
    def _detect_turning_points(self, market_data):
        """
        检测市场拐点
        
        参数:
            market_data: DataFrame, 市场数据
            
        返回:
            list: 拐点信息
        """
        turning_points = []
        
        try:
            if len(market_data) < 30:
                return turning_points
            
            prices = market_data['close'].values
            
            # 三重滤波法寻找拐点
            # 1. 短期移动平均拐点检测
            short_ma = self._calculate_moving_average(prices, 5)
            short_ma_diff = np.diff(short_ma)
            short_sign_change = np.sign(short_ma_diff[:-1]) != np.sign(short_ma_diff[1:])
            
            # 2. 中期移动平均拐点检测
            medium_ma = self._calculate_moving_average(prices, 20)
            medium_ma_diff = np.diff(medium_ma)
            medium_sign_change = np.sign(medium_ma_diff[:-1]) != np.sign(medium_ma_diff[1:])
            
            # 3. MACD指标拐点检测
            _, _, macd_hist = self._calculate_full_macd(prices)
            macd_sign_change = np.sign(macd_hist[:-1]) != np.sign(macd_hist[1:])
            
            # 多重信号确认
            # 找到短期均线转向点附近的MACD转向点
            for i in range(len(short_sign_change)):
                if not short_sign_change[i]:
                    continue
                
                # 检查前后5个点内是否有MACD方向变化
                macd_window = slice(max(0, i-5), min(len(macd_sign_change), i+5))
                if any(macd_sign_change[macd_window]):
                    # 计算MACD直方图变化强度
                    recent_hist = macd_hist[max(0, i-3):min(len(macd_hist), i+4)]
                    hist_change = abs(np.mean(recent_hist))
                    
                    # 判断是向上还是向下的拐点
                    direction = "上升" if short_ma_diff[i] > 0 else "下降"
                    
                    # 计算确信度
                    confidence = min(hist_change * 5, 1.0)  # 标准化到0-1
                    
                    if confidence > self.anomaly_thresholds['turning_point']:
                        turning_points.append({
                            'index': i,
                            'direction': direction,
                            'confidence': confidence,
                            'description': f"检测到市场可能{direction}拐点，确信度：{confidence:.2f}"
                        })
            
            # 保存拐点信息
            self.advanced_warning_system['turning_points'] = turning_points
            
            return turning_points
            
        except Exception as e:
            self.logger.error(f"拐点检测失败: {str(e)}")
            return []
    
    def _calculate_moving_average(self, values, window):
        """计算移动平均"""
        weights = np.ones(window) / window
        return np.convolve(values, weights, mode='valid')
    
    def _calculate_full_macd(self, prices):
        """
        计算完整MACD序列
        
        参数:
            prices: ndarray, 价格数据
            
        返回:
            tuple: MACD线, 信号线, 柱状图
        """
        # 计算EMA
        ema12 = np.zeros_like(prices)
        ema26 = np.zeros_like(prices)
        
        # 初始值
        ema12[0] = prices[0]
        ema26[0] = prices[0]
        
        # EMA计算系数
        k12 = 2.0 / (12 + 1)
        k26 = 2.0 / (26 + 1)
        k9 = 2.0 / (9 + 1)
        
        # 计算12日和26日EMA
        for i in range(1, len(prices)):
            ema12[i] = prices[i] * k12 + ema12[i-1] * (1 - k12)
            ema26[i] = prices[i] * k26 + ema26[i-1] * (1 - k26)
        
        # 计算MACD线
        macd_line = ema12 - ema26
        
        # 计算信号线（MACD的9日EMA）
        signal_line = np.zeros_like(macd_line)
        signal_line[0] = macd_line[0]
        
        for i in range(1, len(macd_line)):
            signal_line[i] = macd_line[i] * k9 + signal_line[i-1] * (1 - k9)
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _detect_ml_anomalies(self, market_data):
        """
        使用机器学习检测市场异常
        
        参数:
            market_data: DataFrame, 市场数据
            
        返回:
            list: 异常信息
        """
        anomalies = []
        
        try:
            if 'isolation_forest' not in self.advanced_warning_system['ml_models']:
                return anomalies
            
            # 提取特征
            features = []
            
            # 价格波动特征
            returns = market_data['close'].pct_change().fillna(0).values[-30:]
            volume = market_data['volume'].values[-30:]
            volume_change = (volume[1:] / volume[:-1] - 1)
            volume_change = np.append(volume_change, 0)
            
            # 合并特征
            for i in range(len(returns)):
                features.append([
                    returns[i], 
                    volume_change[i],
                    abs(returns[i]),  # 波动幅度
                    1 if returns[i] > 0 else -1  # 方向
                ])
            
            features = np.array(features)
            
            # 训练模型 (在线学习)
            model = self.advanced_warning_system['ml_models']['isolation_forest']
            model.fit(features)
            
            # 预测异常
            recent_features = features[-5:]  # 最近5个数据点
            predictions = model.predict(recent_features)
            
            # 异常分数
            scores = model.decision_function(recent_features)
            
            # 检查最近的数据点是否有异常
            has_anomaly = any(p == -1 for p in predictions)
            
            if has_anomaly:
                # 找出异常分数最低的点作为严重程度
                min_score = min(scores)
                severity = abs(min_score) / 0.5  # 标准化
                
                anomalies.append({
                    'type': 'ml_anomaly',
                    'value': abs(min_score),
                    'description': "机器学习算法检测到市场行为异常",
                    'threshold': 0.5,
                    'severity': severity
                })
                
                # 添加到预警信号
                self.advanced_warning_system['warning_signals'].append({
                    'type': 'ml_anomaly',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'description': "市场行为模式异常，可能存在风险",
                    'confidence': min(severity, 1.0)
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"机器学习异常检测失败: {str(e)}")
            traceback.print_exc()
            return []
    
    def _calculate_chaos_metrics(self, market_data):
        """
        计算市场混沌指标
        
        参数:
            market_data: DataFrame, 市场数据
            
        返回:
            dict: 混沌指标
        """
        metrics = {
            'lyapunov_exponent': 0,
            'hurst_exponent': 0.5,
            'entropy': 0
        }
        
        try:
            # 提取收盘价
            prices = market_data['close'].values
            returns = np.diff(np.log(prices))
            
            # 计算Lyapunov指数近似值（局部发散率）
            if len(returns) >= 50:
                # 简化的局部发散率计算
                local_divergence = []
                for i in range(10, len(returns) - 1):
                    # 计算局部相邻点发散程度
                    div = abs((returns[i+1] - returns[i]) / (returns[i] + 1e-10))
                    local_divergence.append(np.log(div + 1e-10))
                
                # Lyapunov指数为局部发散率的平均值
                if local_divergence:
                    metrics['lyapunov_exponent'] = np.mean(local_divergence)
            
            # 计算Hurst指数（使用R/S分析法的简化版本）
            if len(returns) >= 100:
                metrics['hurst_exponent'] = self._calculate_hurst_exponent(returns)
            
            # 计算样本熵
            if len(returns) >= 100:
                metrics['entropy'] = self._calculate_sample_entropy(returns)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算混沌指标失败: {str(e)}")
            return metrics
    
    def _calculate_hurst_exponent(self, time_series, lags=20):
        """
        计算赫斯特指数
        
        参数:
            time_series: ndarray, 时间序列
            lags: int, 计算的最大滞后期
            
        返回:
            float: 赫斯特指数
        """
        lags = min(lags, len(time_series) // 4)
        tau = np.arange(2, lags)
        
        # 方差和重标范围
        var = np.std(time_series)
        rs_values = []
        
        for lag in tau:
            # 将时间序列分段
            parts = len(time_series) // lag
            if parts < 1:
                break
                
            # 计算每个分段的R/S值
            rs_temp = []
            for i in range(parts):
                segment = time_series[i*lag:(i+1)*lag]
                mean_adj = segment - np.mean(segment)
                cumsum = np.cumsum(mean_adj)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(segment) + 1e-10  # 防止除零
                rs_temp.append(r/s)
            
            # 计算平均R/S
            rs_values.append(np.mean(rs_temp))
        
        if not rs_values or len(tau) < 2:
            return 0.5
        
        # 通过对数回归计算赫斯特指数
        log_tau = np.log(tau)
        log_rs = np.log(rs_values)
        
        # 计算斜率，即赫斯特指数
        hurst = np.polyfit(log_tau, log_rs, 1)[0]
        
        return hurst
    
    def _calculate_sample_entropy(self, time_series, m=2, r=0.2):
        """
        计算样本熵
        
        参数:
            time_series: ndarray, 时间序列
            m: int, 模式长度
            r: float, 相似度容差
            
        返回:
            float: 样本熵
        """
        # 标准化序列
        time_series = (time_series - np.mean(time_series)) / (np.std(time_series) + 1e-10)
        r = r * np.std(time_series)
        
        # 简化的样本熵计算
        def count_matches(template, data, r):
            """计算匹配数"""
            matches = 0
            for i in range(len(data) - len(template) + 1):
                match = True
                for j in range(len(template)):
                    if abs(data[i+j] - template[j]) > r:
                        match = False
                        break
                if match:
                    matches += 1
            return matches
        
        # 截取适当的长度计算熵值
        if len(time_series) > 500:
            time_series = time_series[-500:]
        
        # 计算m和m+1的匹配数
        N = len(time_series)
        B = 0.0  # m的匹配计数
        A = 0.0  # m+1的匹配计数
        
        # 只随机抽取一些模板来降低计算复杂度
        templates_m = [time_series[i:i+m] for i in range(0, N-m, 10)]
        templates_m1 = [time_series[i:i+m+1] for i in range(0, N-m-1, 10)]
        
        for template in templates_m:
            B += count_matches(template, time_series, r)
        
        for template in templates_m1:
            A += count_matches(template, time_series, r)
        
        # 避免除零
        if A == 0 or B == 0:
            return 0.0
            
        return -np.log(A / B)
    
    def _analyze_market_structure(self, market_data):
        """
        分析市场结构
        
        参数:
            market_data: DataFrame, 市场数据
        """
        try:
            # 提取收盘价
            prices = market_data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) < 50:
                return
            
            # 计算赫斯特指数
            hurst = self._calculate_hurst_exponent(returns)
            
            # 计算分形维度（使用赫斯特指数的关系：D = 2 - H）
            fractal_dim = 2.0 - hurst
            
            # 计算样本熵
            entropy = self._calculate_sample_entropy(returns)
            
            # 计算复杂度（基于样本熵和分形维度）
            complexity = (entropy * 0.5 + (fractal_dim - 1) * 0.5) / 1.0  # 标准化到0-1
            
            # 确定市场结构状态
            if hurst < 0.4:
                regime = 'mean_reverting'  # 均值回归
                stability = max(0, 1 - (0.4 - hurst) * 2)
            elif hurst > 0.6:
                regime = 'trending'  # 趋势性
                stability = max(0, 1 - (hurst - 0.6) * 2)
            else:
                regime = 'random_walk'  # 随机游走
                stability = 1.0 - abs(hurst - 0.5) * 2
            
            # 更新市场结构
            self.market_structure = {
                'fractal_dimension': fractal_dim,
                'hurst_exponent': hurst,
                'entropy': entropy,
                'complexity': complexity,
                'regime': regime,
                'stability': stability
            }
            
            # 添加市场结构预警
            if complexity > self.anomaly_thresholds['fractality']:
                self.advanced_warning_system['warning_signals'].append({
                    'type': 'market_structure',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'description': f"市场结构复杂性增加，当前为{self._translate_regime(regime)}状态",
                    'confidence': complexity
                })
            
        except Exception as e:
            self.logger.error(f"市场结构分析失败: {str(e)}")
    
    def _translate_regime(self, regime):
        """翻译市场状态类型"""
        translations = {
            'mean_reverting': '均值回归',
            'trending': '趋势性',
            'random_walk': '随机游走'
        }
        return translations.get(regime, regime)
    
    def activate_advanced_warning_system(self, enabled=True):
        """
        激活或关闭高级市场预警系统
        
        参数:
            enabled: bool, 是否激活
            
        返回:
            bool: 操作结果
        """
        try:
            self.advanced_warning_system['enabled'] = enabled
            status = "激活" if enabled else "关闭"
            self.logger.info(f"高级市场预警系统已{status}")
            return True
        except Exception as e:
            self.logger.error(f"修改高级市场预警系统状态失败: {str(e)}")
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