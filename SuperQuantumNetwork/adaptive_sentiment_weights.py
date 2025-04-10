#!/usr/bin/env python3
"""
超神量子共生系统 - 自适应情绪权重系统
动态调整情绪因子对市场预测的影响
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import traceback
from enum import Enum, auto
from collections import deque
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("AdaptiveSentimentWeights")

class MarketRegime(Enum):
    """市场状态枚举"""
    BULLISH_TREND = auto()       # 牛市趋势
    BEARISH_TREND = auto()       # 熊市趋势
    SIDEWAYS = auto()            # 震荡市
    HIGH_VOLATILITY = auto()     # 高波动
    LOW_VOLATILITY = auto()      # 低波动
    MARKET_REVERSAL = auto()     # 市场反转
    MOMENTUM_DRIVEN = auto()     # 动量驱动
    VALUE_DRIVEN = auto()        # 价值驱动
    NEWS_SENSITIVE = auto()      # 新闻敏感
    POLICY_DRIVEN = auto()       # 政策驱动
    LIQUIDITY_DRIVEN = auto()    # 流动性驱动
    QUANTUM_ANOMALY = auto()     # 量子异常状态

class SentimentType(Enum):
    """情绪类型枚举"""
    NEWS_SENTIMENT = auto()      # 新闻情绪
    SOCIAL_MEDIA = auto()        # 社交媒体情绪
    EXPERT_OPINION = auto()      # 专家观点
    INVESTOR_SENTIMENT = auto()  # 投资者情绪
    MARKET_MOOD = auto()         # 市场氛围
    FEAR_GREED = auto()          # 恐惧贪婪指数
    COSMIC_RESONANCE = auto()    # 宇宙共振
    QUANTUM_VIBRATION = auto()   # 量子振动
    TIME_FIELD_DISTORTION = auto() # 时间场扭曲

class AdaptiveSentimentWeightSystem:
    """自适应情绪权重系统"""
    
    def __init__(self, config_path=None):
        """
        初始化自适应情绪权重系统
        
        参数:
            config_path: 配置文件路径
        """
        self.logger = logging.getLogger("AdaptiveSentimentWeightSystem")
        self.logger.info("初始化自适应情绪权重系统...")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化权重
        self.base_weights = {
            SentimentType.NEWS_SENTIMENT.name: 0.35,
            SentimentType.SOCIAL_MEDIA.name: 0.25,
            SentimentType.EXPERT_OPINION.name: 0.20,
            SentimentType.INVESTOR_SENTIMENT.name: 0.40,
            SentimentType.MARKET_MOOD.name: 0.30,
            SentimentType.FEAR_GREED.name: 0.45,
            SentimentType.COSMIC_RESONANCE.name: 0.15,
            SentimentType.QUANTUM_VIBRATION.name: 0.10,
            SentimentType.TIME_FIELD_DISTORTION.name: 0.05
        }
        
        # 当前权重（会动态调整）
        self.current_weights = self.base_weights.copy()
        
        # 当前感知到的市场状态
        self.current_regime = MarketRegime.SIDEWAYS
        
        # 权重调整系数
        self.adjustment_factors = {
            MarketRegime.BULLISH_TREND.name: {
                SentimentType.NEWS_SENTIMENT.name: 0.8,
                SentimentType.SOCIAL_MEDIA.name: 1.2,
                SentimentType.FEAR_GREED.name: 1.5,
                SentimentType.COSMIC_RESONANCE.name: 0.7
            },
            MarketRegime.BEARISH_TREND.name: {
                SentimentType.NEWS_SENTIMENT.name: 1.5,
                SentimentType.SOCIAL_MEDIA.name: 0.9,
                SentimentType.FEAR_GREED.name: 1.7,
                SentimentType.COSMIC_RESONANCE.name: 0.6
            },
            MarketRegime.HIGH_VOLATILITY.name: {
                SentimentType.NEWS_SENTIMENT.name: 1.8,
                SentimentType.SOCIAL_MEDIA.name: 1.6,
                SentimentType.FEAR_GREED.name: 2.0,
                SentimentType.COSMIC_RESONANCE.name: 1.2
            },
            MarketRegime.NEWS_SENSITIVE.name: {
                SentimentType.NEWS_SENTIMENT.name: 2.5,
                SentimentType.SOCIAL_MEDIA.name: 1.8,
                SentimentType.FEAR_GREED.name: 1.2,
                SentimentType.COSMIC_RESONANCE.name: 0.5
            },
            MarketRegime.QUANTUM_ANOMALY.name: {
                SentimentType.NEWS_SENTIMENT.name: 0.3,
                SentimentType.SOCIAL_MEDIA.name: 0.4,
                SentimentType.FEAR_GREED.name: 0.5,
                SentimentType.COSMIC_RESONANCE.name: 3.0,
                SentimentType.QUANTUM_VIBRATION.name: 4.0,
                SentimentType.TIME_FIELD_DISTORTION.name: 2.5
            }
        }
        
        # 情绪影响评估历史
        self.sentiment_impact_history = {
            sentiment_type.name: deque(maxlen=100) 
            for sentiment_type in SentimentType
        }
        
        # 预测表现监控
        self.prediction_performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'regime_performance': {
                regime.name: {'total': 0, 'correct': 0}
                for regime in MarketRegime
            },
            'sentiment_impact': {
                sentiment_type.name: {'positive': 0, 'negative': 0, 'neutral': 0}
                for sentiment_type in SentimentType
            }
        }
        
        # 自适应学习参数
        self.learning_rate = 0.05
        self.exploration_rate = 0.1  # 探索率，用于偶尔尝试新的权重组合
        
        # 情绪能量场状态
        self.sentiment_field = {
            'energy_level': 0.5,
            'polarity': 0.0,
            'coherence': 0.7,
            'resonance_frequency': 7.83,  # 初始设为地球谐振频率
            'entropy': 0.3
        }
        
        self.logger.info("自适应情绪权重系统初始化完成")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'base_learning_rate': 0.05,
            'weight_limits': {
                'min': 0.0,
                'max': 5.0
            },
            'adaptation_speed': 'medium',
            'enable_quantum_factors': True,
            'memory_length': 100,
            'logging_level': 'INFO'
        }
        
        if not config_path or not os.path.exists(config_path):
            self.logger.info("使用默认配置")
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.logger.info(f"配置文件加载成功: {config_path}")
            return {**default_config, **config}  # 合并默认配置和加载的配置
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return default_config
    
    def detect_market_regime(self, market_data, additional_indicators=None):
        """
        检测当前市场状态
        
        参数:
            market_data: DataFrame, 市场数据
            additional_indicators: dict, 附加指标
            
        返回:
            MarketRegime: 当前市场状态
        """
        self.logger.info("检测当前市场状态...")
        
        try:
            # 确保市场数据有足够的历史
            if len(market_data) < 20:
                self.logger.warning("市场数据不足，无法可靠检测市场状态")
                return MarketRegime.SIDEWAYS
            
            # 计算市场指标
            returns = market_data['close'].pct_change().dropna()
            recent_returns = returns[-20:]
            
            # 计算趋势指标
            sma20 = market_data['close'].rolling(20).mean().iloc[-1]
            sma50 = market_data['close'].rolling(50).mean().iloc[-1] if len(market_data) >= 50 else np.nan
            close = market_data['close'].iloc[-1]
            
            # 计算波动率
            volatility_20d = recent_returns.std() * np.sqrt(252)
            
            # 计算动量
            momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-20] - 1)
            
            # 逻辑判断
            regimes = []
            
            # 趋势判断
            if close > sma20 and (np.isnan(sma50) or close > sma50) and momentum > 0.05:
                regimes.append((MarketRegime.BULLISH_TREND, 0.8))
            elif close < sma20 and (np.isnan(sma50) or close < sma50) and momentum < -0.05:
                regimes.append((MarketRegime.BEARISH_TREND, 0.8))
            else:
                regimes.append((MarketRegime.SIDEWAYS, 0.6))
            
            # 波动性判断
            if volatility_20d > 0.25:  # 假设25%年化波动率为高波动阈值
                regimes.append((MarketRegime.HIGH_VOLATILITY, 0.7))
            elif volatility_20d < 0.10:  # 假设10%年化波动率为低波动阈值
                regimes.append((MarketRegime.LOW_VOLATILITY, 0.7))
            
            # 检查是否为动量驱动
            if abs(momentum) > 0.1:
                regimes.append((MarketRegime.MOMENTUM_DRIVEN, 0.6))
            
            # 使用附加指标
            if additional_indicators:
                # 检查新闻敏感度
                if additional_indicators.get('news_sensitivity', 0) > 0.7:
                    regimes.append((MarketRegime.NEWS_SENSITIVE, 0.7))
                
                # 检查政策驱动
                if additional_indicators.get('policy_impact', 0) > 0.7:
                    regimes.append((MarketRegime.POLICY_DRIVEN, 0.7))
                
                # 检查流动性驱动
                if additional_indicators.get('liquidity_factor', 0) > 0.7:
                    regimes.append((MarketRegime.LIQUIDITY_DRIVEN, 0.65))
                
                # 检查量子异常
                if additional_indicators.get('quantum_anomaly', 0) > 0.6:
                    regimes.append((MarketRegime.QUANTUM_ANOMALY, 0.7))
            
            # 选择置信度最高的状态
            regimes.sort(key=lambda x: x[1], reverse=True)
            selected_regime = regimes[0][0]
            
            self.logger.info(f"检测到市场状态: {selected_regime.name}")
            self.current_regime = selected_regime
            
            return selected_regime
            
        except Exception as e:
            self.logger.error(f"检测市场状态失败: {str(e)}")
            traceback.print_exc()
            return MarketRegime.SIDEWAYS  # 默认返回震荡市状态
    
    def update_weights(self, market_data=None, additional_indicators=None, performance_feedback=None):
        """
        更新情绪权重
        
        参数:
            market_data: DataFrame, 可选，市场数据
            additional_indicators: dict, 可选，附加指标
            performance_feedback: dict, 可选，预测性能反馈
            
        返回:
            dict: 更新后的权重
        """
        self.logger.info("更新情绪权重...")
        
        try:
            # 如果提供了市场数据，检测市场状态
            if market_data is not None:
                self.detect_market_regime(market_data, additional_indicators)
            
            # 基于当前市场状态调整权重
            regime_factors = self.adjustment_factors.get(
                self.current_regime.name, 
                {sentiment: 1.0 for sentiment in self.base_weights}
            )
            
            # 应用市场状态调整因子
            for sentiment_type, base_weight in self.base_weights.items():
                adjustment = regime_factors.get(sentiment_type, 1.0)
                self.current_weights[sentiment_type] = base_weight * adjustment
            
            # 如果有性能反馈，进一步优化权重
            if performance_feedback:
                self._apply_performance_feedback(performance_feedback)
            
            # 应用探索率，偶尔随机调整权重以探索更好的组合
            if np.random.random() < self.exploration_rate:
                sentiment_type = np.random.choice(list(self.current_weights.keys()))
                exploration_adjustment = np.random.uniform(0.8, 1.2)
                self.current_weights[sentiment_type] *= exploration_adjustment
                
                self.logger.info(f"探索性调整: {sentiment_type} 权重调整为 {self.current_weights[sentiment_type]:.2f}")
            
            # 确保权重在合理范围内
            min_weight = self.config['weight_limits']['min']
            max_weight = self.config['weight_limits']['max']
            
            for sentiment_type in self.current_weights:
                self.current_weights[sentiment_type] = max(min_weight, min(max_weight, self.current_weights[sentiment_type]))
            
            # 更新情绪能量场
            self._update_sentiment_field()
            
            self.logger.info(f"权重更新完成，当前市场状态: {self.current_regime.name}")
            
            # 添加日志，显示主要情绪类型的权重
            key_sentiments = [SentimentType.NEWS_SENTIMENT.name, SentimentType.FEAR_GREED.name, 
                             SentimentType.COSMIC_RESONANCE.name, SentimentType.QUANTUM_VIBRATION.name]
            for sent_type in key_sentiments:
                self.logger.info(f"  {sent_type}: {self.current_weights[sent_type]:.2f}")
            
            return self.current_weights
            
        except Exception as e:
            self.logger.error(f"更新权重失败: {str(e)}")
            traceback.print_exc()
            return self.current_weights
    
    def _apply_performance_feedback(self, feedback):
        """应用性能反馈来调整权重"""
        if 'accuracy' in feedback:
            accuracy = feedback['accuracy']
            
            # 更新整体性能指标
            self.prediction_performance['total_predictions'] += 1
            if accuracy >= 0.5:  # 假设正确率>=50%为正确预测
                self.prediction_performance['correct_predictions'] += 1
            
            # 更新特定市场状态的性能
            regime_name = self.current_regime.name
            self.prediction_performance['regime_performance'][regime_name]['total'] += 1
            if accuracy >= 0.5:
                self.prediction_performance['regime_performance'][regime_name]['correct'] += 1
            
            # 根据性能调整学习率
            if 'regime_history' in feedback and len(feedback['regime_history']) > 0:
                # 检查当前市场状态的历史表现
                regime_name = self.current_regime.name
                regime_perf = self.prediction_performance['regime_performance'][regime_name]
                
                if regime_perf['total'] > 10:  # 足够的样本量
                    regime_accuracy = regime_perf['correct'] / regime_perf['total'] if regime_perf['total'] > 0 else 0
                    
                    # 如果表现不佳，增加学习率以加速适应
                    if regime_accuracy < 0.6:
                        self.learning_rate = min(0.2, self.learning_rate * 1.2)
                    # 如果表现良好，减小学习率以稳定权重
                    elif regime_accuracy > 0.8:
                        self.learning_rate = max(0.01, self.learning_rate * 0.9)
        
        # 如果提供了具体情绪类型的影响评估
        if 'sentiment_impact' in feedback:
            for sentiment_type, impact in feedback['sentiment_impact'].items():
                if sentiment_type not in self.current_weights:
                    continue
                
                # 记录情绪影响历史
                if sentiment_type in self.sentiment_impact_history:
                    self.sentiment_impact_history[sentiment_type].append(impact)
                
                # 调整权重 - 如果情绪类型对预测有帮助，增加其权重
                adjustment = self.learning_rate * impact
                self.current_weights[sentiment_type] += adjustment
                
                # 更新情绪影响统计
                if impact > 0.1:
                    self.prediction_performance['sentiment_impact'][sentiment_type]['positive'] += 1
                elif impact < -0.1:
                    self.prediction_performance['sentiment_impact'][sentiment_type]['negative'] += 1
                else:
                    self.prediction_performance['sentiment_impact'][sentiment_type]['neutral'] += 1
    
    def _update_sentiment_field(self):
        """更新情绪能量场状态"""
        # 能量水平 - 取决于当前情绪权重和市场状态
        top_sentiment_types = sorted(
            self.current_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # 能量水平 - 基于前三大情绪因子的平均权重
        self.sentiment_field['energy_level'] = sum(weight for _, weight in top_sentiment_types) / 3
        
        # 极性 - 基于情绪因子的影响方向
        if self.sentiment_impact_history:
            recent_impacts = []
            for sentiment_type in self.current_weights:
                if sentiment_type in self.sentiment_impact_history and self.sentiment_impact_history[sentiment_type]:
                    recent_impacts.extend(list(self.sentiment_impact_history[sentiment_type])[-10:])
            
            if recent_impacts:
                self.sentiment_field['polarity'] = sum(recent_impacts) / len(recent_impacts)
        
        # 相干性 - 在高波动市场中降低
        if self.current_regime == MarketRegime.HIGH_VOLATILITY:
            self.sentiment_field['coherence'] = max(0.3, self.sentiment_field['coherence'] * 0.9)
        else:
            self.sentiment_field['coherence'] = min(0.9, self.sentiment_field['coherence'] * 1.05)
        
        # 共振频率 - 基于市场状态和主导情绪类型
        if self.current_regime == MarketRegime.BULLISH_TREND:
            self.sentiment_field['resonance_frequency'] = 10.5  # 高频表示积极市场
        elif self.current_regime == MarketRegime.BEARISH_TREND:
            self.sentiment_field['resonance_frequency'] = 4.5   # 低频表示消极市场
        elif self.current_regime == MarketRegime.QUANTUM_ANOMALY:
            self.sentiment_field['resonance_frequency'] = 21.0  # 量子异常状态下的特殊频率
        else:
            self.sentiment_field['resonance_frequency'] = 7.83  # 默认地球谐振频率
        
        # 熵 - 衡量情绪场的有序性/无序性
        weights_std = np.std(list(self.current_weights.values()))
        self.sentiment_field['entropy'] = min(0.9, max(0.1, weights_std / 2))
    
    def get_sentiment_field(self):
        """获取当前情绪能量场状态"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_regime': self.current_regime.name,
            'energy_field': self.sentiment_field,
            'dominant_sentiments': sorted(
                self.current_weights.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        }
    
    def apply_weights(self, sentiment_data):
        """
        应用权重到情绪数据
        
        参数:
            sentiment_data: dict, 情绪数据，格式为 {情绪类型: 值}
            
        返回:
            dict: 加权后的情绪影响
        """
        try:
            weighted_impact = {}
            total_weighted_sentiment = 0
            total_weight = 0
            
            # 应用权重
            for sentiment_type, sentiment_value in sentiment_data.items():
                if sentiment_type in self.current_weights:
                    weight = self.current_weights[sentiment_type]
                    weighted_value = sentiment_value * weight
                    
                    weighted_impact[sentiment_type] = {
                        'original': sentiment_value,
                        'weight': weight,
                        'weighted_value': weighted_value
                    }
                    
                    total_weighted_sentiment += weighted_value
                    total_weight += weight
            
            # 计算总体加权影响
            overall_impact = total_weighted_sentiment / total_weight if total_weight > 0 else 0
            
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_regime': self.current_regime.name,
                'weighted_components': weighted_impact,
                'overall_impact': overall_impact,
                'sentiment_field': self.sentiment_field
            }
            
        except Exception as e:
            self.logger.error(f"应用权重失败: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}
    
    def visualize_weights(self, save_path=None):
        """
        可视化当前权重
        
        参数:
            save_path: str, 可选，保存路径
        """
        try:
            # 准备数据
            sentiment_types = list(self.current_weights.keys())
            weights = [self.current_weights[t] for t in sentiment_types]
            base_weights = [self.base_weights[t] for t in sentiment_types]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制条形图
            x = np.arange(len(sentiment_types))
            width = 0.35
            
            ax.bar(x - width/2, base_weights, width, label='基础权重')
            ax.bar(x + width/2, weights, width, label='当前权重')
            
            # 添加标签
            ax.set_xlabel('情绪类型')
            ax.set_ylabel('权重值')
            ax.set_title(f'情绪权重分布 (市场状态: {self.current_regime.name})')
            ax.set_xticks(x)
            ax.set_xticklabels(sentiment_types, rotation=45, ha='right')
            ax.legend()
            
            # 添加能量场信息
            info_text = "\n".join([
                f"能量水平: {self.sentiment_field['energy_level']:.2f}",
                f"极性: {self.sentiment_field['polarity']:.2f}",
                f"相干性: {self.sentiment_field['coherence']:.2f}",
                f"共振频率: {self.sentiment_field['resonance_frequency']:.2f} Hz",
                f"熵: {self.sentiment_field['entropy']:.2f}"
            ])
            
            plt.figtext(0.15, 0.02, info_text, fontsize=10)
            
            plt.tight_layout()
            
            # 保存图表
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                self.logger.info(f"权重可视化已保存到: {save_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"可视化权重失败: {str(e)}")
            traceback.print_exc()
    
    def save_state(self, save_path):
        """
        保存系统状态
        
        参数:
            save_path: str, 保存路径
        
        返回:
            bool: 是否成功
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            state = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_weights': self.current_weights,
                'base_weights': self.base_weights,
                'current_regime': self.current_regime.name,
                'sentiment_field': self.sentiment_field,
                'learning_rate': self.learning_rate,
                'exploration_rate': self.exploration_rate,
                'prediction_performance': self.prediction_performance
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"系统状态已保存到: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_state(self, load_path):
        """
        加载系统状态
        
        参数:
            load_path: str, 加载路径
        
        返回:
            bool: 是否成功
        """
        try:
            if not os.path.exists(load_path):
                self.logger.error(f"状态文件不存在: {load_path}")
                return False
            
            with open(load_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 加载权重和参数
            self.current_weights = state['current_weights']
            self.base_weights = state['base_weights']
            self.current_regime = MarketRegime[state['current_regime']]
            self.sentiment_field = state['sentiment_field']
            self.learning_rate = state['learning_rate']
            self.exploration_rate = state['exploration_rate']
            self.prediction_performance = state['prediction_performance']
            
            self.logger.info(f"系统状态已从 {load_path} 加载")
            return True
            
        except Exception as e:
            self.logger.error(f"加载系统状态失败: {str(e)}")
            traceback.print_exc()
            return False

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建自适应情绪权重系统
    weight_system = AdaptiveSentimentWeightSystem()
    
    # 输出诊断信息
    print("\n" + "="*60)
    print("超神量子共生系统 - 自适应情绪权重系统")
    print("="*60 + "\n")
    
    print("情绪类型与基础权重:")
    for sentiment_type, weight in weight_system.base_weights.items():
        print(f"- {sentiment_type}: {weight:.2f}")
    
    print("\n市场状态调整因子示例:")
    for regime in [MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND, MarketRegime.QUANTUM_ANOMALY]:
        print(f"\n{regime.name}:")
        if regime.name in weight_system.adjustment_factors:
            for sentiment, factor in weight_system.adjustment_factors[regime.name].items():
                print(f"  - {sentiment}: x{factor:.2f}")
    
    print("\n情绪能量场当前状态:")
    for key, value in weight_system.sentiment_field.items():
        print(f"- {key}: {value}")
    
    print("\n系统准备就绪，可以开始动态调整情绪权重。")
    print("="*60 + "\n") 