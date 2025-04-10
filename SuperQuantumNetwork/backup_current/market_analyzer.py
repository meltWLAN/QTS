#!/usr/bin/env python3
"""
市场分析器 - 超神量子共生系统的市场分析组件
使用高维分析技术评估市场趋势和状态
"""

import os
import time
import json
import random
import logging
import numpy as np
from datetime import datetime, timedelta
import traceback

# 配置日志
logger = logging.getLogger("MarketAnalyzer")

class MarketAnalyzer:
    """市场分析器 - 分析市场趋势和模式"""
    
    def __init__(self, dimension_count=11):
        """初始化市场分析器"""
        self.logger = logging.getLogger("MarketAnalyzer")
        self.logger.info("初始化市场分析器...")
        
        # 基本配置
        self.dimension_count = dimension_count
        self.initialized = False
        self.active = False
        self.last_update = datetime.now()
        
        # 分析器状态
        self.analyzer_state = {
            "market_temperature": 0.5,  # 0-冷，1-热
            "volatility": 0.5,  # 0-低，1-高
            "trend_strength": 0.5,  # 0-弱，1-强
            "sentiment": 0.5,  # 0-悲观，1-乐观
            "liquidity": 0.5,  # 0-低，1-高
            "anomaly_index": 0.0,  # 异常指数，越高越异常
            "dimension_complexity": dimension_count,
            "last_analysis": None
        }
        
        # 高维分析矩阵
        self.analysis_matrix = np.zeros((dimension_count, dimension_count))
        
        # 历史分析结果
        self.analysis_history = []
        
        self.logger.info(f"市场分析器初始化完成，维度: {dimension_count}")
    
    def initialize(self, field_strength=0.75):
        """初始化分析器"""
        self.logger.info("初始化市场分析器...")
        
        # 初始化分析矩阵
        scale = field_strength * 0.5
        self.analysis_matrix = np.random.randn(self.dimension_count, self.dimension_count) * scale
        
        # 设置分析器状态
        self.analyzer_state["market_temperature"] = random.uniform(0.4, 0.6)
        self.analyzer_state["volatility"] = random.uniform(0.4, 0.6)
        self.analyzer_state["trend_strength"] = random.uniform(0.4, 0.6)
        self.analyzer_state["sentiment"] = random.uniform(0.4, 0.6)
        self.analyzer_state["liquidity"] = random.uniform(0.4, 0.6)
        
        self.initialized = True
        self.active = True
        self.logger.info("市场分析器初始化完成")
        
        return True
    
    def analyze_market(self, market_data, stock_data_dict):
        """分析整体市场状况
        
        参数:
            market_data (dict): 市场概况数据
            stock_data_dict (dict): 多只股票的数据字典
            
        返回:
            dict: 市场分析结果
        """
        if not self.active:
            self.logger.warning("市场分析器未激活，无法执行分析")
            return None
        
        self.logger.info(f"开始分析市场，包含{len(stock_data_dict)}只股票")
        
        try:
            # 提取市场基本数据
            indices = market_data.get('indices', {})
            market_status = market_data.get('market_status', '未知')
            up_stocks = market_data.get('up_stocks', 0)
            down_stocks = market_data.get('down_stocks', 0)
            flat_stocks = market_data.get('flat_stocks', 0)
            
            total_stocks = up_stocks + down_stocks + flat_stocks
            up_ratio = up_stocks / total_stocks if total_stocks > 0 else 0.5
            
            # 股票数据综合
            all_returns = []
            all_volumes = []
            all_volatilities = []
            
            for code, data in stock_data_dict.items():
                if len(data) >= 5:
                    # 收益率
                    returns = [(data[i]['close'] / data[i-1]['close'] - 1) for i in range(1, len(data))]
                    all_returns.extend(returns)
                    
                    # 成交量
                    volumes = [item['volume'] for item in data]
                    all_volumes.extend(volumes)
                    
                    # 波动率
                    if len(data) >= 10:
                        prices = [item['close'] for item in data[-10:]]
                        volatility = np.std(prices) / np.mean(prices)
                        all_volatilities.append(volatility)
            
            # 计算市场温度（热度）
            market_temp = up_ratio * 0.7 + 0.3 * (0.5 + 0.5 * np.mean(all_returns) / 0.01 if all_returns else 0.5)
            market_temp = max(0, min(1, market_temp))  # 限制在0-1之间
            
            # 计算波动性
            volatility = np.mean(all_volatilities) if all_volatilities else 0.5
            volatility = 0.3 + volatility * 7  # 调整到合适范围
            volatility = max(0, min(1, volatility))  # 限制在0-1之间
            
            # 计算趋势强度
            if len(all_returns) >= 5:
                recent_returns = all_returns[-5:]
                same_direction = sum(1 for i in range(1, len(recent_returns)) if recent_returns[i] * recent_returns[i-1] > 0)
                trend_strength = 0.5 + 0.1 * same_direction
            else:
                trend_strength = 0.5
            
            # 计算情绪指标
            sentiment = market_temp * 0.6 + (up_ratio - 0.5) * 0.8 + 0.5
            sentiment = max(0, min(1, sentiment))  # 限制在0-1之间
            
            # 计算流动性
            liquidity = np.mean(all_volumes) / 5000000 if all_volumes else 0.5
            liquidity = max(0, min(1, liquidity))  # 限制在0-1之间
            
            # 使用高维分析矩阵调整分析结果（模拟量子分析）
            matrix_influence = np.sum(np.abs(self.analysis_matrix)) / (self.dimension_count ** 2)
            
            # 根据矩阵影响调整各指标
            market_temp = market_temp * 0.8 + matrix_influence * 0.4 * random.uniform(0.8, 1.2)
            market_temp = max(0, min(1, market_temp))
            
            volatility = volatility * 0.8 + matrix_influence * 0.4 * random.uniform(0.8, 1.2)
            volatility = max(0, min(1, volatility))
            
            trend_strength = trend_strength * 0.8 + matrix_influence * 0.4 * random.uniform(0.8, 1.2)
            trend_strength = max(0, min(1, trend_strength))
            
            sentiment = sentiment * 0.8 + matrix_influence * 0.4 * random.uniform(0.8, 1.2)
            sentiment = max(0, min(1, sentiment))
            
            liquidity = liquidity * 0.8 + matrix_influence * 0.4 * random.uniform(0.8, 1.2)
            liquidity = max(0, min(1, liquidity))
            
            # 检测异常模式
            anomaly_index = 0.0
            if abs(market_temp - 0.5) > 0.3 and volatility > 0.7:
                anomaly_index += 0.3  # 极端市场加高波动
            
            if sentiment > 0.8 and trend_strength < 0.3:
                anomaly_index += 0.2  # 高情绪但趋势弱
            
            if sentiment < 0.2 and trend_strength > 0.7:
                anomaly_index += 0.2  # 低情绪但趋势强
            
            # 更新分析器状态
            self.analyzer_state["market_temperature"] = float(market_temp)
            self.analyzer_state["volatility"] = float(volatility)
            self.analyzer_state["trend_strength"] = float(trend_strength)
            self.analyzer_state["sentiment"] = float(sentiment)
            self.analyzer_state["liquidity"] = float(liquidity)
            self.analyzer_state["anomaly_index"] = float(anomaly_index)
            self.analyzer_state["last_analysis"] = datetime.now()
            
            # 确定市场状态
            if market_temp > 0.7:
                if volatility > 0.7:
                    market_phase = "过热"
                else:
                    market_phase = "牛市"
            elif market_temp < 0.3:
                if volatility > 0.7:
                    market_phase = "恐慌"
                else:
                    market_phase = "熊市"
            else:
                if volatility > 0.6:
                    market_phase = "震荡"
                else:
                    market_phase = "盘整"
            
            # 构建分析结果
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'indices': indices,
                'market_status': market_status,
                'analyzed_stocks_count': len(stock_data_dict),
                'up_ratio': float(up_ratio),
                'market_indicators': {
                    'temperature': float(market_temp),
                    'volatility': float(volatility),
                    'trend_strength': float(trend_strength),
                    'sentiment': float(sentiment),
                    'liquidity': float(liquidity),
                    'anomaly_index': float(anomaly_index)
                },
                'market_phase': market_phase,
                'quantum_influence': float(matrix_influence),
                'prediction': self._generate_market_prediction(market_temp, volatility, trend_strength, sentiment)
            }
            
            # 添加到历史分析
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'market_phase': market_phase,
                'temperature': float(market_temp),
                'volatility': float(volatility)
            })
            
            # 更新高维分析矩阵（模拟自适应）
            evolution = np.random.randn(self.dimension_count, self.dimension_count) * 0.05
            self.analysis_matrix = 0.95 * self.analysis_matrix + 0.05 * evolution
            
            self.logger.info(f"市场分析完成: 阶段={market_phase}, 温度={market_temp:.2f}, 波动={volatility:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"市场分析失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def analyze_stock(self, stock_data):
        """分析单只股票的技术指标和趋势
        
        参数:
            stock_data (list): 股票历史数据
            
        返回:
            dict: 股票分析结果
        """
        if not self.active or len(stock_data) < 10:
            self.logger.warning(f"无法分析股票，数据点不足 ({len(stock_data)})")
            return None
        
        code = stock_data[0]['code']
        self.logger.info(f"分析股票: {code}")
        
        try:
            # 提取价格和交易量数据
            closes = np.array([item['close'] for item in stock_data])
            opens = np.array([item['open'] for item in stock_data])
            highs = np.array([item['high'] for item in stock_data])
            lows = np.array([item['low'] for item in stock_data])
            volumes = np.array([item['volume'] for item in stock_data])
            dates = [item['date'] for item in stock_data]
            
            # 计算基本指标
            price_mean = np.mean(closes)
            price_std = np.std(closes)
            last_price = closes[-1]
            
            # 计算移动平均线
            ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else None
            ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else None
            ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
            ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else None
            
            # 计算相对强弱指标(RSI) - 简化版
            delta = np.diff(closes)
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            if len(gain) >= 14:
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50
            
            # 计算MACD
            if len(closes) >= 26:
                ema12 = np.mean(closes[-12:])  # 简化版，实际应使用指数移动平均
                ema26 = np.mean(closes[-26:])
                macd = ema12 - ema26
            else:
                macd = 0
                
            # 计算布林带
            if len(closes) >= 20:
                middle_band = ma20
                std_dev = np.std(closes[-20:])
                upper_band = middle_band + (std_dev * 2)
                lower_band = middle_band - (std_dev * 2)
                
                # 计算布林带宽度
                bandwidth = (upper_band - lower_band) / middle_band
            else:
                middle_band = price_mean
                upper_band = price_mean + price_std
                lower_band = price_mean - price_std
                bandwidth = 2 * price_std / price_mean
            
            # 计算成交量指标
            avg_volume = np.mean(volumes)
            vol_increase = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            
            # 计算波动率
            volatility = price_std / price_mean if price_mean > 0 else 0.0
            
            # 计算动量
            momentum = closes[-1] / closes[-5] - 1 if len(closes) >= 5 else 0.0
            
            # 根据高维分析矩阵调整技术指标
            matrix_influence = np.sum(np.abs(self.analysis_matrix)) / (self.dimension_count ** 2)
            
            # 计算趋势和反转信号
            trend_signals = []
            reversal_signals = []
            
            # 趋势信号
            if ma5 and ma20 and ma5 > ma20:
                trend_signals.append("短期均线位于长期均线上方，可能处于上升趋势")
            elif ma5 and ma20 and ma5 < ma20:
                trend_signals.append("短期均线位于长期均线下方，可能处于下降趋势")
            
            if rsi > 70:
                trend_signals.append("RSI大于70，显示强势")
            elif rsi < 30:
                trend_signals.append("RSI小于30，显示弱势")
            
            if macd > 0:
                trend_signals.append("MACD为正，可能处于上升趋势")
            elif macd < 0:
                trend_signals.append("MACD为负，可能处于下降趋势")
            
            # 反转信号
            if rsi > 75 and vol_increase < 0.7:
                reversal_signals.append("RSI过高且成交量下降，可能出现顶部反转")
            elif rsi < 25 and vol_increase > 1.3:
                reversal_signals.append("RSI过低且成交量上升，可能出现底部反转")
            
            if len(closes) >= 3 and closes[-1] < closes[-2] < closes[-3] and ma5 and ma5 > closes[-1]:
                reversal_signals.append("连续下跌并跌破短期均线，可能继续下跌")
            
            if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3] and ma5 and ma5 < closes[-1]:
                reversal_signals.append("连续上涨并突破短期均线，可能继续上涨")
            
            # 确定整体趋势
            if len(closes) >= 20:
                short_trend = ma5 / ma10 - 1 if ma5 and ma10 else 0
                medium_trend = ma10 / ma20 - 1 if ma10 and ma20 else 0
                
                if short_trend > 0.01 and medium_trend > 0.005:
                    trend = "上升"
                elif short_trend < -0.01 and medium_trend < -0.005:
                    trend = "下降"
                else:
                    trend = "横盘"
            else:
                trend = "数据不足"
            
            # 计算支撑位和阻力位
            support_levels = []
            resistance_levels = []
            
            if len(closes) >= 20:
                # 简化版支撑和阻力计算
                min_price = np.min(lows[-20:])
                support_levels.append(round(min_price * 0.99, 2))
                support_levels.append(round(min_price * 1.01, 2))
                
                max_price = np.max(highs[-20:])
                resistance_levels.append(round(max_price * 0.99, 2))
                resistance_levels.append(round(max_price * 1.01, 2))
                
                # 根据移动平均线添加支撑阻力
                if ma20:
                    if last_price > ma20:
                        support_levels.append(round(ma20, 2))
                    else:
                        resistance_levels.append(round(ma20, 2))
            
            # 强弱分析
            strength_score = 0.0
            
            # 基于均线的强弱分析
            if ma5 and ma10 and ma20:
                if ma5 > ma10 > ma20:
                    strength_score += 0.3
                elif ma5 < ma10 < ma20:
                    strength_score -= 0.3
                elif ma5 > ma10 and ma10 < ma20:
                    strength_score += 0.1  # 可能是反转初期
            
            # 基于RSI的强弱
            if rsi > 60:
                strength_score += 0.2
            elif rsi < 40:
                strength_score -= 0.2
            
            # 基于价格位置
            if upper_band and lower_band:
                price_position = (last_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
                strength_score += (price_position - 0.5) * 0.2
            
            # 基于成交量的强弱
            if vol_increase > 1.2:
                strength_score += 0.15
            elif vol_increase < 0.8:
                strength_score -= 0.15
            
            # 量子影响
            strength_score += (matrix_influence - 0.5) * 0.2
            
            # 强度值范围-1至1之间
            strength_score = max(-1, min(1, strength_score))
            
            # 根据得分确定强弱状态
            if strength_score > 0.5:
                strength_status = "强势"
            elif strength_score < -0.5:
                strength_status = "弱势"
            elif strength_score > 0.2:
                strength_status = "偏强"
            elif strength_score < -0.2:
                strength_status = "偏弱"
            else:
                strength_status = "中性"
            
            # 构建分析结果
            result = {
                'code': code,
                'name': stock_data[0].get('name', f"股票{code}"),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_price': float(last_price),
                'technical_indicators': {
                    'ma5': float(ma5) if ma5 is not None else None,
                    'ma10': float(ma10) if ma10 is not None else None,
                    'ma20': float(ma20) if ma20 is not None else None,
                    'ma60': float(ma60) if ma60 is not None else None,
                    'rsi': float(rsi),
                    'macd': float(macd),
                    'bollinger_bands': {
                        'upper': float(upper_band),
                        'middle': float(middle_band),
                        'lower': float(lower_band),
                        'bandwidth': float(bandwidth)
                    },
                    'volume': {
                        'average': float(avg_volume),
                        'increase_ratio': float(vol_increase)
                    },
                    'volatility': float(volatility),
                    'momentum': float(momentum)
                },
                'trend_analysis': {
                    'overall_trend': trend,
                    'trend_signals': trend_signals,
                    'reversal_signals': reversal_signals,
                    'strength_score': float(strength_score),
                    'strength_status': strength_status
                },
                'support_resistance': {
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels
                },
                'quantum_influence': float(matrix_influence)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"股票分析失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_market_prediction(self, temperature, volatility, trend_strength, sentiment):
        """生成市场预测"""
        if temperature > 0.7 and volatility > 0.6:
            if trend_strength > 0.6:
                prediction = "市场处于强势上涨阶段，但波动较大，需关注可能的短期回调"
            else:
                prediction = "市场热度较高但趋势不明确，可能面临变盘"
                
        elif temperature > 0.7 and volatility <= 0.6:
            prediction = "市场稳健上涨，热度较高，短期内可能继续上行"
            
        elif temperature < 0.3 and volatility > 0.6:
            prediction = "市场处于恐慌阶段，剧烈波动中，建议谨慎操作，等待企稳信号"
            
        elif temperature < 0.3 and volatility <= 0.6:
            prediction = "市场低迷，但波动较小，可能处于底部震荡阶段"
            
        else:  # 中性温度
            if volatility > 0.6:
                prediction = "市场震荡加剧，方向不明确，建议控制仓位，观望为主"
            else:
                prediction = "市场处于平稳整理阶段，可能在蓄势待发"
                
        # 根据情绪调整预测
        if sentiment > 0.7:
            prediction += "。市场情绪乐观，需警惕过度乐观带来的风险"
        elif sentiment < 0.3:
            prediction += "。市场情绪悲观，可能为逆势操作提供机会"
            
        return prediction
    
    def get_analyzer_state(self):
        """获取分析器状态"""
        return {
            "market_temperature": float(self.analyzer_state["market_temperature"]),
            "volatility": float(self.analyzer_state["volatility"]),
            "trend_strength": float(self.analyzer_state["trend_strength"]),
            "sentiment": float(self.analyzer_state["sentiment"]),
            "liquidity": float(self.analyzer_state["liquidity"]),
            "anomaly_index": float(self.analyzer_state["anomaly_index"]),
            "dimension_complexity": int(self.analyzer_state["dimension_complexity"]),
            "initialized": self.initialized,
            "active": self.active,
            "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        } 