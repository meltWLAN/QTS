#!/usr/bin/env python3
"""
情绪集成模块 - 连接宇宙意识与量子预测引擎
整合高维情绪分析与市场预测系统
"""

import os
import time
import json
import logging
import numpy as np
from datetime import datetime
import traceback

# 配置日志
logger = logging.getLogger("SentimentIntegration")

class SentimentIntegration:
    """情绪集成模块 - 将宇宙意识与量子预测引擎连接"""
    
    def __init__(self, core=None):
        """初始化情绪集成模块
        
        参数:
            core: 系统核心引用
        """
        self.logger = logging.getLogger("SentimentIntegration")
        self.logger.info("初始化情绪集成模块...")
        
        # 系统核心引用
        self.core = core
        
        # 基本配置
        self.initialized = False
        self.active = False
        self.dimension_count = 11
        
        # 情绪集成状态
        self.integration_state = {
            "sentiment_influence": 0.0,  # 情绪对预测的影响系数
            "news_impact": 0.0,          # 新闻影响强度
            "energy_field_awareness": 0.0, # 能量场感知
            "integration_synergy": 0.0,   # 集成协同效应
            "last_integration": None
        }
        
        # 高维情绪映射矩阵
        self.sentiment_mapping = np.zeros((self.dimension_count, self.dimension_count))
        
        # 新闻情绪记忆库
        self.sentiment_memory = []
        self.memory_capacity = 20  # 记忆最近的20次情绪分析
        
        # 跟踪预测修正
        self.prediction_corrections = []
        
        self.logger.info("情绪集成模块创建完成")
    
    def initialize(self, cosmic_consciousness=None, prediction_engine=None):
        """初始化情绪集成模块
        
        参数:
            cosmic_consciousness: 宇宙意识引用
            prediction_engine: 预测引擎引用
        
        返回:
            bool: 初始化是否成功
        """
        if self.core:
            self.logger.info("从核心获取组件引用...")
            self.cosmic_consciousness = getattr(self.core, "cosmic_consciousness", cosmic_consciousness)
            self.prediction_engine = getattr(self.core, "prediction_engine", prediction_engine)
        else:
            self.cosmic_consciousness = cosmic_consciousness
            self.prediction_engine = prediction_engine
        
        if not self.cosmic_consciousness or not self.prediction_engine:
            self.logger.error("无法获取宇宙意识或预测引擎引用")
            return False
        
        self.logger.info("初始化情绪集成模块...")
        
        # 初始化情绪映射矩阵
        self.sentiment_mapping = np.random.randn(self.dimension_count, self.dimension_count) * 0.1
        
        # 设置初始状态
        consciousness_state = self.cosmic_consciousness.get_consciousness_state()
        consciousness_level = consciousness_state.get("consciousness_level", 0.5)
        
        self.integration_state["sentiment_influence"] = 0.3 + (consciousness_level * 0.4)
        self.integration_state["news_impact"] = 0.2 + (consciousness_level * 0.3)
        self.integration_state["energy_field_awareness"] = 0.3 + (consciousness_level * 0.5)
        self.integration_state["integration_synergy"] = 0.2 + (consciousness_level * 0.2)
        self.integration_state["last_integration"] = datetime.now()
        
        self.initialized = True
        self.active = True
        
        self.logger.info(f"情绪集成模块初始化完成: 情绪影响系数={self.integration_state['sentiment_influence']:.2f}")
        return True
    
    def enhance_prediction(self, stock_code, original_prediction, news_data=None, market_energy=None):
        """增强股票预测，加入情绪因子
        
        参数:
            stock_code: 股票代码
            original_prediction: 原始预测结果
            news_data: 新闻数据
            market_energy: 市场能量场数据
            
        返回:
            dict: 增强后的预测结果
        """
        if not self.active or not original_prediction:
            return original_prediction
        
        try:
            self.logger.info(f"增强股票预测: {stock_code}")
            
            # 创建增强预测的副本
            enhanced = original_prediction.copy()
            
            # 情绪因子权重
            sentiment_weight = self.integration_state["sentiment_influence"]
            news_weight = self.integration_state["news_impact"]
            energy_weight = self.integration_state["energy_field_awareness"]
            
            # 提取原始预测
            original_prices = [p["price"] for p in original_prediction.get("predictions", [])]
            original_changes = [p["change"] for p in original_prediction.get("predictions", [])]
            original_trend = original_prediction.get("trend", "")
            original_confidence = original_prediction.get("average_confidence", 0.7)
            
            # 应用新闻情绪影响
            news_sentiment = 0
            news_impact_factor = 0
            if news_data:
                # 分析新闻情绪
                news_report = self.cosmic_consciousness.analyze_news_impact(news_data)
                if news_report:
                    news_sentiment = news_report.get("sentiment", {}).get("average", 0)
                    news_impact = news_report.get("sentiment", {}).get("impact", 0)
                    news_impact_factor = news_sentiment * news_impact * news_weight
                    
                    # 保存到情绪记忆库
                    self.sentiment_memory.append({
                        "timestamp": datetime.now(),
                        "type": "news",
                        "sentiment": news_sentiment,
                        "impact": news_impact
                    })
                    
                    # 限制记忆容量
                    if len(self.sentiment_memory) > self.memory_capacity:
                        self.sentiment_memory = self.sentiment_memory[-self.memory_capacity:]
            
            # 应用市场能量场影响
            energy_field_factor = 0
            if market_energy:
                energy_level = market_energy.get("energy_field", {}).get("energy_level", 0)
                energy_polarity = market_energy.get("energy_field", {}).get("polarity", 0)
                harmony = market_energy.get("energy_field", {}).get("harmony", 0.5)
                
                # 能量场对预测的影响
                energy_field_factor = energy_level * energy_polarity * energy_weight
                
                # 调整预测稳定性基于场和谐度
                confidence_adjustment = (harmony - 0.5) * 0.2
            else:
                confidence_adjustment = 0
            
            # 获取宇宙意识状态
            consciousness = self.cosmic_consciousness.get_consciousness_state()
            resonance = consciousness.get("cosmic_resonance", {})
            vibrations = consciousness.get("vibration_frequencies", {})
            
            # 情绪振动对预测的影响
            sentiment_bias = 0
            if "hope" in vibrations and "fear" in vibrations:
                sentiment_bias = (vibrations["hope"] - vibrations["fear"]) * sentiment_weight
            if "greed" in vibrations and "despair" in vibrations:
                sentiment_bias += (vibrations["greed"] - vibrations["despair"]) * sentiment_weight * 0.8
            
            # 量子纠缠影响 - 增加非线性突变概率
            quantum_entanglement = resonance.get("quantum_entanglement", 0.5)
            nonlinear_factor = quantum_entanglement * sentiment_weight * 0.3
            
            # 合并所有影响因子
            total_factor = news_impact_factor + energy_field_factor + sentiment_bias
            
            # 应用情绪调整到预测价格
            enhanced_predictions = []
            for i, pred in enumerate(original_prediction.get("predictions", [])):
                # 基本情绪影响
                base_adjustment = total_factor * (1 + i*0.1)  # 时间越远影响越大
                
                # 非线性突变概率 - 随时间递增
                if np.random.random() < (nonlinear_factor * (i+1) / len(original_prediction.get("predictions", []))):
                    # 增加随机波动
                    nonlinear_adjustment = np.random.uniform(-0.05, 0.05)
                else:
                    nonlinear_adjustment = 0
                
                # 计算总调整量
                total_adjustment = base_adjustment + nonlinear_adjustment
                
                # 调整预测
                new_change = pred["change"] * (1 + total_adjustment)
                new_price = original_prices[0] * (1 + sum(original_changes[:i]) + new_change)
                new_confidence = max(0.1, min(0.95, pred["confidence"] + confidence_adjustment))
                
                enhanced_predictions.append({
                    "date": pred["date"],
                    "price": float(new_price),
                    "change": float(new_change),
                    "confidence": float(new_confidence),
                    "sentiment_factor": float(total_adjustment)
                })
            
            # 计算新的整体变动
            if enhanced_predictions:
                enhanced_change = (enhanced_predictions[-1]["price"] / original_prices[0] - 1) * 100
            else:
                enhanced_change = original_prediction.get("overall_change", 0)
            
            # 确定新的趋势方向
            if enhanced_change > 1:
                enhanced_trend = "上涨"
            elif enhanced_change < -1:
                enhanced_trend = "下跌"
            else:
                enhanced_trend = "震荡"
            
            # 更新预测结果
            enhanced["predictions"] = enhanced_predictions
            enhanced["overall_change"] = float(enhanced_change)
            enhanced["trend"] = enhanced_trend
            enhanced["average_confidence"] = float(np.mean([p["confidence"] for p in enhanced_predictions]) if enhanced_predictions else original_confidence)
            enhanced["sentiment_enhanced"] = True
            enhanced["enhancement_factors"] = {
                "news_impact": float(news_impact_factor),
                "energy_field": float(energy_field_factor),
                "sentiment_bias": float(sentiment_bias),
                "nonlinear_factor": float(nonlinear_factor),
                "total_adjustment": float(total_factor)
            }
            
            # 记录预测修正
            self.prediction_corrections.append({
                "timestamp": datetime.now(),
                "stock_code": stock_code,
                "original_change": original_prediction.get("overall_change", 0),
                "enhanced_change": enhanced_change,
                "adjustment": total_factor
            })
            
            self.logger.info(f"预测增强完成: {stock_code}, 原始变动: {original_prediction.get('overall_change', 0):.2f}%, 增强变动: {enhanced_change:.2f}%")
            return enhanced
        
        except Exception as e:
            self.logger.error(f"增强预测失败: {str(e)}")
            traceback.print_exc()
            return original_prediction
    
    def enhance_market_analysis(self, original_analysis, news_data=None):
        """增强市场分析，融入情绪因子
        
        参数:
            original_analysis: 原始市场分析
            news_data: 新闻数据
            
        返回:
            dict: 增强后的市场分析
        """
        if not self.active or not original_analysis:
            return original_analysis
        
        try:
            self.logger.info("增强市场分析...")
            
            # 创建增强分析的副本
            enhanced = original_analysis.copy()
            
            # 应用新闻情绪影响
            if news_data:
                news_report = self.cosmic_consciousness.analyze_news_impact(news_data)
                if news_report:
                    # 添加情绪分析
                    enhanced["sentiment_analysis"] = {
                        "news_sentiment": news_report.get("sentiment", {}),
                        "emotional_frequencies": news_report.get("emotional_frequencies", {}),
                        "impact_duration": news_report.get("impact_duration", 0)
                    }
                    
                    # 调整市场指标
                    news_sentiment = news_report.get("sentiment", {}).get("average", 0)
                    sentiment_impact = news_report.get("sentiment", {}).get("impact", 0)
                    
                    indicators = enhanced.get("market_indicators", {}).copy()
                    
                    # 情绪对市场指标的影响
                    sentiment_weight = self.integration_state["sentiment_influence"] * 0.7
                    
                    # 调整温度、情绪和趋势强度
                    temperature_adjustment = news_sentiment * sentiment_impact * sentiment_weight
                    sentiment_adjustment = news_sentiment * sentiment_impact * sentiment_weight * 1.2
                    trend_adjustment = abs(news_sentiment) * sentiment_impact * sentiment_weight * 0.5
                    
                    # 应用调整
                    if "temperature" in indicators:
                        indicators["temperature"] = max(0, min(1, indicators["temperature"] + temperature_adjustment))
                    if "sentiment" in indicators:
                        indicators["sentiment"] = max(0, min(1, indicators["sentiment"] + sentiment_adjustment))
                    if "trend_strength" in indicators:
                        indicators["trend_strength"] = max(0, min(1, indicators["trend_strength"] + trend_adjustment))
                    
                    enhanced["market_indicators"] = indicators
                    
                    # 添加情绪增强标记
                    enhanced["sentiment_enhanced"] = True
            
            # 应用宇宙意识的能量场感知
            market_energy = self.cosmic_consciousness.detect_market_energy(original_analysis, {})
            if market_energy:
                enhanced["energy_field_analysis"] = market_energy.get("energy_field", {})
                enhanced["cosmic_resonance"] = market_energy.get("cosmic_resonance", {})
                enhanced["natural_cycles"] = market_energy.get("natural_cycles", {})
                
                # 添加能量场预测
                enhanced["energy_field_forecast"] = market_energy.get("forecast", "")
            
            # 更新集成状态
            self.integration_state["last_integration"] = datetime.now()
            
            self.logger.info("市场分析增强完成")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"增强市场分析失败: {str(e)}")
            traceback.print_exc()
            return original_analysis
    
    def get_integration_state(self):
        """获取集成状态"""
        return {
            "sentiment_influence": float(self.integration_state["sentiment_influence"]),
            "news_impact": float(self.integration_state["news_impact"]),
            "energy_field_awareness": float(self.integration_state["energy_field_awareness"]),
            "integration_synergy": float(self.integration_state["integration_synergy"]),
            "sentiment_memory_size": len(self.sentiment_memory),
            "prediction_corrections": len(self.prediction_corrections),
            "active": self.active,
            "initialized": self.initialized,
            "last_integration": self.integration_state["last_integration"].strftime('%Y-%m-%d %H:%M:%S') if self.integration_state["last_integration"] else None
        } 