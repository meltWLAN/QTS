#!/usr/bin/env python3
"""
量子集成模块 - 超神量子共生系统的集成组件
整合预测引擎和市场分析器，提供统一接口
"""

import os
import time
import json
import logging
import numpy as np
from datetime import datetime
import traceback

# 配置日志
logger = logging.getLogger("QuantumIntegration")

class QuantumIntegration:
    """量子集成模块 - 整合预测引擎和市场分析器的集成层"""
    
    def __init__(self, core=None):
        """初始化量子集成模块
        
        参数:
            core: 系统核心引用
        """
        self.logger = logging.getLogger("QuantumIntegration")
        self.logger.info("初始化量子集成模块...")
        
        # 系统核心引用
        self.core = core
        
        # 基本配置
        self.initialized = False
        self.active = False
        self.dimension_count = 11
        self.field_strength = 0.75
        
        # 集成状态
        self.integration_state = {
            "coherence": 0.0,
            "synergy_level": 0.0,
            "accuracy_boost": 0.0,
            "last_integration": None
        }
        
        self.logger.info("量子集成模块创建完成")
    
    def initialize(self, field_strength=0.75, dimension_count=11):
        """初始化集成模块
        
        参数:
            field_strength: 场强
            dimension_count: 维度数
        
        返回:
            bool: 初始化是否成功
        """
        self.logger.info(f"初始化集成模块: 场强={field_strength}, 维度={dimension_count}")
        
        # 保存参数
        self.field_strength = field_strength
        self.dimension_count = dimension_count
        
        # 计算初始状态
        self.integration_state["coherence"] = field_strength * 0.8
        self.integration_state["synergy_level"] = field_strength * 0.6
        self.integration_state["accuracy_boost"] = field_strength * 0.4
        self.integration_state["last_integration"] = datetime.now()
        
        self.initialized = True
        self.active = True
        
        self.logger.info(f"集成模块初始化完成: 相干性={self.integration_state['coherence']:.2f}")
        return True
    
    def analyze_and_predict(self, market_data, stock_data_dict, days_ahead=5):
        """整合市场分析和股票预测
        
        参数:
            market_data: 市场概况数据
            stock_data_dict: 多只股票数据字典
            days_ahead: 预测未来天数
            
        返回:
            dict: 整合的分析和预测结果
        """
        if not self.active or not self.core:
            self.logger.warning("集成模块未激活或核心引用不存在")
            return None
        
        self.logger.info(f"开始整合分析和预测: {len(stock_data_dict)}只股票")
        
        try:
            # 获取分析器和预测引擎
            analyzer = self.core.market_analyzer
            predictor = self.core.prediction_engine
            
            if not analyzer or not predictor:
                self.logger.error("无法获取分析器或预测引擎")
                return None
            
            # 执行市场分析
            market_analysis = analyzer.analyze_market(market_data, stock_data_dict)
            
            if not market_analysis:
                self.logger.error("市场分析失败")
                return None
            
            # 分析每只股票
            stock_analyses = {}
            for code, data in stock_data_dict.items():
                stock_analysis = analyzer.analyze_stock(data)
                if stock_analysis:
                    stock_analyses[code] = stock_analysis
            
            # 预测每只股票
            stock_predictions = {}
            for code, data in stock_data_dict.items():
                prediction = predictor.predict_stock(data, days_ahead=days_ahead)
                if prediction:
                    stock_predictions[code] = prediction
            
            # 整合分析和预测
            integrated_results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_analysis': market_analysis,
                'stock_analyses': stock_analyses,
                'stock_predictions': stock_predictions,
                'integration_metrics': {
                    'coherence': float(self.integration_state["coherence"]),
                    'synergy_level': float(self.integration_state["synergy_level"]),
                    'accuracy_boost': float(self.integration_state["accuracy_boost"]),
                    'field_strength': float(self.field_strength),
                    'dimension_count': int(self.dimension_count)
                },
                'overview': self._generate_integrated_overview(market_analysis, stock_analyses, stock_predictions)
            }
            
            # 更新集成状态
            self.integration_state["last_integration"] = datetime.now()
            
            # 保存结果到文件
            self._save_integrated_results(integrated_results)
            
            self.logger.info(f"分析和预测整合完成: {len(stock_analyses)}只股票分析, {len(stock_predictions)}只股票预测")
            return integrated_results
            
        except Exception as e:
            self.logger.error(f"整合分析和预测失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_integrated_overview(self, market_analysis, stock_analyses, stock_predictions):
        """生成整合概述"""
        if not market_analysis:
            return "无法生成概述：缺少市场分析数据"
        
        # 市场概述
        market_phase = market_analysis.get('market_phase', '未知')
        market_prediction = market_analysis.get('prediction', '')
        
        # 分析股票趋势分布
        trend_distribution = {"上升": 0, "下降": 0, "横盘": 0}
        for code, analysis in stock_analyses.items():
            trend = analysis.get('trend_analysis', {}).get('overall_trend', '')
            if trend in trend_distribution:
                trend_distribution[trend] += 1
        
        # 预测结果分布
        prediction_distribution = {"上涨": 0, "下跌": 0, "震荡": 0}
        for code, prediction in stock_predictions.items():
            trend = prediction.get('trend', '')
            if trend in prediction_distribution:
                prediction_distribution[trend] += 1
        
        # 寻找强势和弱势股票
        strong_stocks = []
        weak_stocks = []
        
        for code, analysis in stock_analyses.items():
            strength_status = analysis.get('trend_analysis', {}).get('strength_status', '')
            name = analysis.get('name', code)
            
            if strength_status in ["强势", "偏强"]:
                if code in stock_predictions:
                    change = stock_predictions[code].get('overall_change', 0)
                    if change > 0:
                        strong_stocks.append((name, change))
            
            elif strength_status in ["弱势", "偏弱"]:
                if code in stock_predictions:
                    change = stock_predictions[code].get('overall_change', 0)
                    if change < 0:
                        weak_stocks.append((name, change))
        
        # 排序
        strong_stocks.sort(key=lambda x: x[1], reverse=True)
        weak_stocks.sort(key=lambda x: x[1])
        
        # 生成概述文本
        overview = f"市场处于{market_phase}阶段。{market_prediction}\n\n"
        
        overview += f"股票趋势分布：上升 {trend_distribution['上升']}，下降 {trend_distribution['下降']}，横盘 {trend_distribution['横盘']}。\n"
        overview += f"预测结果分布：上涨 {prediction_distribution['上涨']}，下跌 {prediction_distribution['下跌']}，震荡 {prediction_distribution['震荡']}。\n\n"
        
        if strong_stocks:
            overview += "强势股票 (预计上涨):\n"
            for name, change in strong_stocks[:5]:  # 取前5名
                overview += f"- {name}: 预计变动 {change:.2f}%\n"
            overview += "\n"
        
        if weak_stocks:
            overview += "弱势股票 (预计下跌):\n"
            for name, change in weak_stocks[:5]:  # 取前5名
                overview += f"- {name}: 预计变动 {change:.2f}%\n"
        
        return overview
    
    def _save_integrated_results(self, results):
        """保存集成结果到文件"""
        try:
            # 创建报告目录
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # 保存为JSON
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_file = os.path.join(reports_dir, f"integrated_results_{timestamp}.json")
            
            # 将numpy值转换为Python原生类型
            results_json = json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x, indent=2)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(results_json)
                
            self.logger.info(f"集成结果已保存到: {json_file}")
            
        except Exception as e:
            self.logger.error(f"保存集成结果失败: {str(e)}")
    
    def generate_summary_report(self, results, output_file=None):
        """生成报告"""
        if not results:
            self.logger.warning("无法生成报告: 缺少结果数据")
            return None
        
        self.logger.info("生成分析和预测报告...")
        
        try:
            # 提取数据
            timestamp = results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            market_analysis = results.get('market_analysis', {})
            stock_analyses = results.get('stock_analyses', {})
            stock_predictions = results.get('stock_predictions', {})
            overview = results.get('overview', '')
            
            # 创建报告内容
            report = f"# 市场分析与预测报告\n\n"
            report += f"**生成时间:** {timestamp}\n\n"
            
            # 概述
            report += "## 概述\n\n"
            report += overview + "\n\n"
            
            # 市场分析
            report += "## 市场分析\n\n"
            
            if market_analysis:
                market_phase = market_analysis.get('market_phase', '未知')
                market_prediction = market_analysis.get('prediction', '')
                
                report += f"**市场阶段:** {market_phase}\n\n"
                report += f"**市场预测:** {market_prediction}\n\n"
                
                # 市场指标
                indicators = market_analysis.get('market_indicators', {})
                report += "### 市场指标\n\n"
                report += "| 指标 | 数值 |\n"
                report += "|------|------|\n"
                report += f"| 市场温度 | {indicators.get('temperature', 0):.2f} |\n"
                report += f"| 波动性 | {indicators.get('volatility', 0):.2f} |\n"
                report += f"| 趋势强度 | {indicators.get('trend_strength', 0):.2f} |\n"
                report += f"| 情绪指数 | {indicators.get('sentiment', 0):.2f} |\n"
                report += f"| 流动性 | {indicators.get('liquidity', 0):.2f} |\n"
                report += f"| 异常指数 | {indicators.get('anomaly_index', 0):.2f} |\n\n"
            
            # 个股分析和预测
            report += "## 个股分析和预测\n\n"
            
            # 按预测变动排序
            sorted_stocks = []
            for code in stock_predictions:
                if code in stock_analyses:
                    name = stock_analyses[code].get('name', code)
                    change = stock_predictions[code].get('overall_change', 0)
                    sorted_stocks.append((code, name, change))
            
            sorted_stocks.sort(key=lambda x: x[2], reverse=True)
            
            for code, name, change in sorted_stocks:
                analysis = stock_analyses.get(code, {})
                prediction = stock_predictions.get(code, {})
                
                report += f"### {name} ({code})\n\n"
                
                # 价格信息
                last_price = analysis.get('last_price', 0)
                report += f"**最新价格:** {last_price:.2f}\n\n"
                
                # 技术分析
                trend_analysis = analysis.get('trend_analysis', {})
                overall_trend = trend_analysis.get('overall_trend', '未知')
                strength_status = trend_analysis.get('strength_status', '未知')
                
                report += f"**整体趋势:** {overall_trend}\n"
                report += f"**强弱状态:** {strength_status}\n\n"
                
                # 技术指标
                indicators = analysis.get('technical_indicators', {})
                report += "#### 技术指标\n\n"
                report += "| 指标 | 数值 |\n"
                report += "|------|------|\n"
                
                if indicators.get('ma5') is not None:
                    report += f"| MA5 | {indicators.get('ma5', 0):.2f} |\n"
                if indicators.get('ma10') is not None:
                    report += f"| MA10 | {indicators.get('ma10', 0):.2f} |\n"
                if indicators.get('ma20') is not None:
                    report += f"| MA20 | {indicators.get('ma20', 0):.2f} |\n"
                
                report += f"| RSI | {indicators.get('rsi', 0):.2f} |\n"
                report += f"| MACD | {indicators.get('macd', 0):.2f} |\n"
                report += f"| 波动率 | {indicators.get('volatility', 0):.2f} |\n"
                report += f"| 动量 | {indicators.get('momentum', 0):.2f} |\n\n"
                
                # 趋势信号
                trend_signals = trend_analysis.get('trend_signals', [])
                if trend_signals:
                    report += "#### 趋势信号\n\n"
                    for signal in trend_signals:
                        report += f"- {signal}\n"
                    report += "\n"
                
                # 反转信号
                reversal_signals = trend_analysis.get('reversal_signals', [])
                if reversal_signals:
                    report += "#### 反转信号\n\n"
                    for signal in reversal_signals:
                        report += f"- {signal}\n"
                    report += "\n"
                
                # 支撑阻力位
                support_resistance = analysis.get('support_resistance', {})
                support_levels = support_resistance.get('support_levels', [])
                resistance_levels = support_resistance.get('resistance_levels', [])
                
                if support_levels or resistance_levels:
                    report += "#### 支撑阻力位\n\n"
                    if support_levels:
                        report += f"**支撑位:** {', '.join([str(level) for level in support_levels])}\n"
                    if resistance_levels:
                        report += f"**阻力位:** {', '.join([str(level) for level in resistance_levels])}\n"
                    report += "\n"
                
                # 预测结果
                report += "#### 预测结果\n\n"
                
                if prediction:
                    overall_change = prediction.get('overall_change', 0)
                    trend = prediction.get('trend', '未知')
                    confidence = prediction.get('average_confidence', 0)
                    
                    report += f"**预测趋势:** {trend}\n"
                    report += f"**预计变动:** {overall_change:.2f}%\n"
                    report += f"**平均信心度:** {confidence:.2f}\n\n"
                    
                    # 详细预测
                    predictions = prediction.get('predictions', [])
                    if predictions:
                        report += "| 日期 | 预测价格 | 变动百分比 | 信心度 |\n"
                        report += "|------|----------|------------|--------|\n"
                        
                        for p in predictions:
                            report += f"| {p.get('date')} | {p.get('price', 0):.2f} | {p.get('change', 0)*100:.2f}% | {p.get('confidence', 0):.2f} |\n"
                        
                        report += "\n"
                
                report += "---\n\n"
            
            # 如果指定输出文件，保存到文件
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"报告已保存到: {output_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {str(e)}")
            traceback.print_exc()
            return None 