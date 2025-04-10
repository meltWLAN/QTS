#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 投资组合控制器
实现超神级投资组合管理、资产配置优化和风险管理功能
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import random
import uuid
from PyQt5.QtCore import QObject, pyqtSignal

from quantum_symbiotic_network.quantum_prediction import get_predictor


class PortfolioController(QObject):
    """超神投资组合控制器，负责管理投资组合、资产配置与风险管理"""
    
    # 信号定义
    portfolio_updated = pyqtSignal(dict)
    allocation_updated = pyqtSignal(list)
    risk_metrics_updated = pyqtSignal(dict)
    
    def __init__(self):
        """初始化投资组合控制器"""
        super().__init__()
        self.logger = logging.getLogger("PortfolioController")
        
        # 投资组合基本属性
        self.portfolio_data = {}
        self.allocation_data = []
        self.risk_metrics = {}
        self.performance_data = {}
        self.optimization_history = []
        
        # 量子增强配置
        self.quantum_enhancement = {
            "enabled": True,
            "dimension_channels": 7,  # 多维度分析通道数
            "risk_sensitivity": 0.85,  # 风险敏感度
            "market_awareness": 0.92,  # 市场感知力
            "adaptation_rate": 0.78,  # 策略自适应速率
            "prediction_weight": 0.65  # 量子预测权重
        }
        
        # 高维市场状态感知
        self.market_state = {
            "volatility": 0.0,
            "trend": 0.0,
            "liquidity": 0.0,
            "sentiment": 0.0,
            "correlation": 0.0,
            "anomaly": 0.0,
            "dimension_stability": 0.0
        }
        
        # 连接量子预测器
        try:
            self.quantum_predictor = get_predictor()
            if self.quantum_predictor:
                self.logger.info("量子预测器连接成功")
        except Exception as e:
            self.logger.error(f"连接量子预测器失败: {str(e)}")
            self.quantum_predictor = None
            
        # 尝试连接共生核心
        try:
            from hyperunity_activation import get_symbiosis_core
            self.symbiosis_core = get_symbiosis_core()
            if self.symbiosis_core:
                self.logger.info("共生核心连接成功")
                self.symbiosis_core.register_module("portfolio_controller", self)
        except Exception as e:
            self.logger.error(f"连接共生核心失败: {str(e)}")
            self.symbiosis_core = None
            
        # 初始化模拟数据
        self._initialize_mock_data()
        
        # 启动自动优化线程
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._run_optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        self.logger.info("超神投资组合控制器初始化完成")
        
    def _initialize_mock_data(self):
        """初始化模拟数据"""
        # 投资组合数据
        self.portfolio_data = {
            "total_asset": 1000000.0,
            "available_cash": 500000.0,
            "market_value": 500000.0,
            "daily_profit": 50000.0,
            "daily_profit_pct": 0.05,
            "total_profit": 100000.0,
            "total_profit_pct": 0.1,
            "max_drawdown": 0.052,
            "sharpe": 2.35,
            "volatility": 0.158,
            "var": 25000.0,
            "quantum_score": 0.85  # 量子优化评分
        }
        
        # 资产配置数据
        self.allocation_data = [
            {'name': '金融', 'value': 0.25, 'color': (255, 0, 0), 'risk_score': 0.75, 'growth_potential': 0.65},
            {'name': '科技', 'value': 0.30, 'color': (0, 255, 0), 'risk_score': 0.85, 'growth_potential': 0.90},
            {'name': '医药', 'value': 0.15, 'color': (0, 0, 255), 'risk_score': 0.60, 'growth_potential': 0.75},
            {'name': '消费', 'value': 0.20, 'color': (255, 255, 0), 'risk_score': 0.65, 'growth_potential': 0.70},
            {'name': '其他', 'value': 0.10, 'color': (128, 128, 128), 'risk_score': 0.50, 'growth_potential': 0.60}
        ]
        
        # 风险指标
        self.risk_metrics = {
            "overall_risk": 0.65,
            "market_risk": 0.70,
            "specific_risk": 0.60,
            "liquidity_risk": 0.45,
            "correlation_risk": 0.55,
            "dimension_risk": 0.50,  # 维度风险
            "quantum_stability": 0.85,  # 量子稳定性
            "adaptive_hedge_ratio": 0.40  # 自适应对冲比率
        }
        
        # 绩效数据
        days = 100
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.sin(i/10)) for i in range(days)]
        benchmark_values = [1000000 * (1 + 0.0008 * i) for i in range(days)]
        
        self.performance_data = {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "annual_return": 0.158,
            "alpha": 0.052,
            "beta": 0.85,
            "sortino": 1.95,
            "win_rate": 0.652,
            "profit_loss_ratio": 2.5,
            "quantum_enhancement": 0.03,  # 量子增强收益
            "dimension_stability": 0.88  # 维度稳定性
        }
        
    def get_portfolio_data(self):
        """获取投资组合数据"""
        # 更新数据
        self._update_portfolio_data()
        return self.portfolio_data
    
    def get_allocation_data(self):
        """获取资产配置数据"""
        return self.allocation_data
    
    def get_risk_metrics(self):
        """获取风险指标"""
        return self.risk_metrics
    
    def get_performance_data(self):
        """获取绩效数据"""
        return self.performance_data
    
    def _update_portfolio_data(self):
        """更新投资组合数据"""
        try:
            # 如果有量子预测器，使用其市场情绪数据
            if self.quantum_predictor:
                insights = self.quantum_predictor.generate_market_insights({})
                if insights and 'market_sentiment' in insights:
                    sentiment = insights['market_sentiment'].get('score', 0)
                    
                    # 根据市场情绪调整收益
                    volatility = random.uniform(0.005, 0.015)
                    change = sentiment * volatility
                    
                    # 更新投资组合价值
                    self.portfolio_data['total_asset'] *= (1 + change)
                    self.portfolio_data['market_value'] *= (1 + change * 1.5)
                    
                    # 更新收益
                    self.portfolio_data['daily_profit'] = self.portfolio_data['total_asset'] * random.uniform(0.01, 0.05) * (1 + sentiment)
                    self.portfolio_data['daily_profit_pct'] = self.portfolio_data['daily_profit'] / self.portfolio_data['total_asset']
            else:
                # 简单随机变动
                volatility = random.uniform(0.005, 0.015)
                change = random.choice([-1, 1]) * volatility
                
                # 更新投资组合价值
                self.portfolio_data['total_asset'] *= (1 + change)
                self.portfolio_data['market_value'] *= (1 + change * 1.5)
                
                # 更新收益
                self.portfolio_data['daily_profit'] = self.portfolio_data['total_asset'] * random.uniform(0.01, 0.05)
                self.portfolio_data['daily_profit_pct'] = self.portfolio_data['daily_profit'] / self.portfolio_data['total_asset']
            
            # 更新可用资金
            self.portfolio_data['available_cash'] = self.portfolio_data['total_asset'] - self.portfolio_data['market_value']
            
            # 更新量子评分
            self.portfolio_data['quantum_score'] = min(0.99, self.portfolio_data['quantum_score'] + random.uniform(-0.03, 0.05))
            
            # 发出更新信号
            self.portfolio_updated.emit(self.portfolio_data)
            
        except Exception as e:
            self.logger.error(f"更新投资组合数据失败: {str(e)}")
    
    def _run_optimization_loop(self):
        """运行投资组合优化循环"""
        while self.optimization_active:
            try:
                # 更新市场状态
                self._update_market_state()
                
                # 优化资产配置
                self._optimize_asset_allocation()
                
                # 更新风险指标
                self._update_risk_metrics()
                
                # 更新绩效数据
                self._update_performance_data()
                
                # 记录优化历史
                self._record_optimization()
                
                # 若有共生核心，发送更新
                if self.symbiosis_core:
                    self.symbiosis_core.send_message({
                        "source": "portfolio_controller",
                        "type": "portfolio_update",
                        "data": {
                            "allocation": self.allocation_data,
                            "risk_metrics": self.risk_metrics,
                            "quantum_score": self.portfolio_data.get("quantum_score", 0)
                        }
                    })
                
            except Exception as e:
                self.logger.error(f"投资组合优化循环出错: {str(e)}")
            
            # 休眠一段时间
            time.sleep(random.uniform(30, 60))
    
    def _update_market_state(self):
        """更新高维市场状态"""
        try:
            # 更新基本状态
            self.market_state['volatility'] = random.uniform(0.1, 0.9)
            self.market_state['trend'] = random.uniform(-0.8, 0.8)
            self.market_state['liquidity'] = random.uniform(0.2, 0.95)
            self.market_state['sentiment'] = random.uniform(-0.7, 0.7)
            self.market_state['correlation'] = random.uniform(0.3, 0.9)
            
            # 如果有量子预测器，使用其数据
            if self.quantum_predictor:
                try:
                    insights = self.quantum_predictor.generate_market_insights({})
                    if insights and 'market_sentiment' in insights:
                        # 检查market_sentiment是否为字典类型
                        if isinstance(insights['market_sentiment'], dict):
                            self.market_state['sentiment'] = insights['market_sentiment'].get('score', 0)
                        else:
                            # 如果是浮点数或其他类型，直接使用该值
                            self.market_state['sentiment'] = float(insights['market_sentiment'])
                    
                    # 使用预测器的量子参数
                    self.market_state['dimension_stability'] = self.quantum_predictor.coherence
                    
                except Exception as e:
                    self.logger.error(f"获取量子预测数据失败: {str(e)}")
                    self.market_state['dimension_stability'] = random.uniform(0.5, 0.95)
            else:
                self.market_state['dimension_stability'] = random.uniform(0.5, 0.95)
            
            # 异常检测
            self.market_state['anomaly'] = random.uniform(0, 0.3)  # 低概率异常
            
            # 记录市场状态
            self.logger.debug(f"市场状态: 波动率={self.market_state['volatility']:.2f}, " + 
                             f"趋势={self.market_state['trend']:.2f}, " + 
                             f"情绪={self.market_state['sentiment']:.2f}")
                
        except Exception as e:
            self.logger.error(f"更新市场状态失败: {str(e)}")
    
    def _optimize_asset_allocation(self):
        """优化资产配置 - 量子增强版"""
        try:
            # 获取市场状态
            volatility = self.market_state['volatility']
            trend = self.market_state['trend']
            sentiment = self.market_state['sentiment']
            dimension_stability = self.market_state['dimension_stability']
            
            # 量子增强权重
            if self.quantum_enhancement['enabled']:
                enhancement_factor = self.quantum_enhancement['market_awareness'] * dimension_stability
            else:
                enhancement_factor = 0.5
            
            # 根据市场状态调整配置
            for sector in self.allocation_data:
                # 基础调整
                base_adjustment = random.uniform(-0.03, 0.03)
                
                # 基于风险的调整 (高波动时减少高风险资产)
                risk_adjustment = (0.5 - sector['risk_score']) * volatility * 0.1
                
                # 基于趋势的调整 (上升趋势时增加高成长性资产)
                trend_adjustment = sector['growth_potential'] * trend * 0.1
                
                # 基于情绪的调整
                sentiment_adjustment = sector['growth_potential'] * sentiment * 0.05
                
                # 量子维度调整 (使用量子维度稳定性作为信心因子)
                quantum_adjustment = (sector['growth_potential'] - 0.5) * dimension_stability * 0.1
                
                # 总调整
                total_adjustment = (base_adjustment + 
                                   risk_adjustment + 
                                   trend_adjustment + 
                                   sentiment_adjustment + 
                                   quantum_adjustment * enhancement_factor)
                
                # 应用调整
                sector['value'] = max(0.01, min(0.60, sector['value'] + total_adjustment))
            
            # 标准化配置比例总和为1
            total = sum(sector['value'] for sector in self.allocation_data)
            for sector in self.allocation_data:
                sector['value'] = sector['value'] / total
            
            # 发出更新信号
            self.allocation_updated.emit(self.allocation_data)
            
            self.logger.info(f"资产配置已优化 (量子增强系数: {enhancement_factor:.2f})")
            
        except Exception as e:
            self.logger.error(f"优化资产配置失败: {str(e)}")
    
    def _update_risk_metrics(self):
        """更新风险指标 - 多维度分析"""
        try:
            # 获取市场状态
            volatility = self.market_state['volatility']
            correlation = self.market_state['correlation']
            liquidity = self.market_state['liquidity']
            anomaly = self.market_state['anomaly']
            dimension_stability = self.market_state['dimension_stability']
            
            # 计算资产组合总风险
            portfolio_risk = 0
            for sector in self.allocation_data:
                portfolio_risk += sector['value'] * sector['risk_score']
            
            # 更新风险指标
            self.risk_metrics['market_risk'] = volatility * 0.8 + random.uniform(-0.1, 0.1)
            self.risk_metrics['specific_risk'] = portfolio_risk * 0.7 + random.uniform(-0.1, 0.1)
            self.risk_metrics['liquidity_risk'] = (1 - liquidity) * 0.8 + random.uniform(-0.05, 0.05)
            self.risk_metrics['correlation_risk'] = correlation * 0.6 + random.uniform(-0.1, 0.1)
            self.risk_metrics['dimension_risk'] = (1 - dimension_stability) * 0.7 + anomaly * 0.3
            
            # 量子稳定性与自适应对冲比率
            self.risk_metrics['quantum_stability'] = dimension_stability * 0.9 + random.uniform(-0.05, 0.05)
            self.risk_metrics['adaptive_hedge_ratio'] = max(0.1, min(0.9, portfolio_risk * (1 - dimension_stability) + 0.2))
            
            # 计算总体风险
            self.risk_metrics['overall_risk'] = (
                self.risk_metrics['market_risk'] * 0.3 +
                self.risk_metrics['specific_risk'] * 0.2 +
                self.risk_metrics['liquidity_risk'] * 0.15 +
                self.risk_metrics['correlation_risk'] * 0.15 +
                self.risk_metrics['dimension_risk'] * 0.2
            )
            
            # 标准化风险值
            for key in self.risk_metrics:
                self.risk_metrics[key] = max(0.01, min(0.99, self.risk_metrics[key]))
            
            # 发出更新信号
            self.risk_metrics_updated.emit(self.risk_metrics)
            
        except Exception as e:
            self.logger.error(f"更新风险指标失败: {str(e)}")
    
    def _update_performance_data(self):
        """更新绩效数据"""
        try:
            # 更新最近一天的收益
            days = len(self.performance_data.get('portfolio_values', []))
            if days > 0:
                last_value = self.performance_data['portfolio_values'][-1]
                
                # 获取市场状态
                trend = self.market_state['trend']
                sentiment = self.market_state['sentiment']
                volatility = self.market_state['volatility']
                
                # 量子增强系数
                if self.quantum_enhancement['enabled'] and self.quantum_predictor:
                    quantum_factor = self.quantum_enhancement['prediction_weight'] * self.market_state['dimension_stability']
                else:
                    quantum_factor = 0.5
                
                # 基本变动
                base_change = random.uniform(-0.02, 0.02)
                
                # 趋势影响
                trend_impact = trend * 0.01
                
                # 情绪影响
                sentiment_impact = sentiment * 0.005
                
                # 波动影响
                volatility_impact = (random.random() - 0.5) * volatility * 0.02
                
                # 量子增强影响
                quantum_impact = (random.random() - 0.3) * quantum_factor * 0.02
                
                # 总变动
                total_change = base_change + trend_impact + sentiment_impact + volatility_impact + quantum_impact
                
                # 计算新值
                new_value = last_value * (1 + total_change)
                
                # 更新历史数据
                self.performance_data['portfolio_values'].append(new_value)
                if len(self.performance_data['portfolio_values']) > 100:
                    self.performance_data['portfolio_values'] = self.performance_data['portfolio_values'][-100:]
                
                # 更新绩效指标
                self.performance_data['annual_return'] = max(0.01, min(0.3, self.performance_data['annual_return'] + random.uniform(-0.01, 0.015)))
                self.performance_data['quantum_enhancement'] = max(0.01, min(0.1, quantum_factor * 0.1))
                self.performance_data['dimension_stability'] = self.market_state['dimension_stability']
                
                # 更新基准比较
                last_benchmark = self.performance_data['benchmark_values'][-1]
                new_benchmark = last_benchmark * (1 + total_change * 0.7)  # 假设基准收益率较低
                self.performance_data['benchmark_values'].append(new_benchmark)
                if len(self.performance_data['benchmark_values']) > 100:
                    self.performance_data['benchmark_values'] = self.performance_data['benchmark_values'][-100:]
                
                # 更新日期
                if len(self.performance_data['dates']) > 0:
                    last_date = self.performance_data['dates'][-1]
                    new_date = last_date + timedelta(days=1)
                    self.performance_data['dates'].append(new_date)
                    if len(self.performance_data['dates']) > 100:
                        self.performance_data['dates'] = self.performance_data['dates'][-100:]
            
        except Exception as e:
            self.logger.error(f"更新绩效数据失败: {str(e)}")
    
    def _record_optimization(self):
        """记录优化历史"""
        try:
            # 创建优化记录
            record = {
                "timestamp": datetime.now().isoformat(),
                "allocation": {sector['name']: sector['value'] for sector in self.allocation_data},
                "risk_metrics": self.risk_metrics.copy(),
                "market_state": self.market_state.copy(),
                "portfolio_value": self.portfolio_data.get('total_asset', 0)
            }
            
            # 添加到历史记录
            self.optimization_history.append(record)
            
            # 限制历史记录数量
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
                
        except Exception as e:
            self.logger.error(f"记录优化历史失败: {str(e)}")
    
    def on_connect_symbiosis(self, symbiosis_core):
        """连接到共生核心"""
        try:
            self.symbiosis_core = symbiosis_core
            self.logger.info("投资组合控制器成功连接到共生核心")
            return True
        except Exception as e:
            self.logger.error(f"连接共生核心失败: {str(e)}")
            return False
    
    def on_disconnect_symbiosis(self):
        """断开与共生核心的连接"""
        self.symbiosis_core = None
        self.logger.info("投资组合控制器已断开与共生核心的连接")
        return True
    
    def on_symbiosis_message(self, message):
        """处理来自共生核心的消息"""
        try:
            message_type = message.get("type", "")
            source = message.get("source", "unknown")
            data = message.get("data", {})
            
            self.logger.debug(f"收到消息 [{message_type}] 来自 {source}")
            
            # 处理市场状态更新
            if message_type == "market_state_update":
                if "volatility" in data:
                    self.market_state['volatility'] = data['volatility']
                if "trend" in data:
                    self.market_state['trend'] = data['trend']
                if "sentiment" in data:
                    self.market_state['sentiment'] = data['sentiment']
                
                # 触发重新优化
                threading.Thread(target=self._optimize_asset_allocation).start()
                return True
            
            # 处理量子预测消息
            elif message_type == "quantum_prediction" and source == "quantum_predictor":
                if "quantum_state" in data and data["quantum_state"]:
                    # 更新维度稳定性
                    if "coherence" in data["quantum_state"]:
                        self.market_state['dimension_stability'] = data["quantum_state"]["coherence"]
                    
                    # 触发风险更新
                    threading.Thread(target=self._update_risk_metrics).start()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"处理共生消息失败: {str(e)}")
            return False
    
    def shutdown(self):
        """关闭投资组合控制器"""
        try:
            self.optimization_active = False
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=2.0)
            
            self.logger.info("投资组合控制器已安全关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭投资组合控制器时出错: {str(e)}")
            return False


# 全局投资组合控制器实例
_portfolio_controller = None

def get_portfolio_controller():
    """获取全局投资组合控制器实例"""
    global _portfolio_controller
    if _portfolio_controller is None:
        _portfolio_controller = PortfolioController()
    return _portfolio_controller 