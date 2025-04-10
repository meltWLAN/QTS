#!/usr/bin/env python3
"""
系统集成模块 - 将宇宙意识与情绪分析整合到超神量子共生系统
提供统一接口和功能调用
"""

import os
import time
import json
import logging
from datetime import datetime
import traceback

# 配置日志
logger = logging.getLogger("SystemIntegration")

class QuantumSystemIntegrator:
    """超神量子系统集成器 - 整合所有高级组件"""
    
    def __init__(self, quantum_system=None):
        """初始化系统集成器
        
        参数:
            quantum_system: 量子系统引用
        """
        self.logger = logging.getLogger("SystemIntegration")
        self.logger.info("初始化系统集成器...")
        
        # 系统引用
        self.quantum_system = quantum_system
        
        # 组件引用
        self.cosmic_consciousness = None
        self.sentiment_integration = None
        self.news_crawler = None
        
        # 基本状态
        self.initialized = False
        self.active = False
        
        self.logger.info("系统集成器创建完成")
    
    def integrate_cosmic_consciousness(self, cosmic_consciousness):
        """集成宇宙意识模块
        
        参数:
            cosmic_consciousness: 宇宙意识模块引用
            
        返回:
            bool: 集成是否成功
        """
        if not cosmic_consciousness:
            self.logger.error("宇宙意识模块引用无效")
            return False
            
        self.cosmic_consciousness = cosmic_consciousness
        
        # 将模块添加到量子系统中
        if self.quantum_system:
            # 避免使用固定属性名，而是使用更安全的方法
            if hasattr(self.quantum_system, "__dict__"):
                self.quantum_system.__dict__["cosmic_consciousness"] = cosmic_consciousness
                self.logger.info("宇宙意识模块已关联到量子系统")
            else:
                self.logger.warning("无法关联宇宙意识模块到量子系统")
        
        self.logger.info("宇宙意识模块集成成功")
        return True
    
    def integrate_sentiment_analysis(self, sentiment_integration):
        """集成情绪分析模块
        
        参数:
            sentiment_integration: 情绪集成模块引用
            
        返回:
            bool: 集成是否成功
        """
        if not sentiment_integration:
            self.logger.error("情绪集成模块引用无效")
            return False
            
        self.sentiment_integration = sentiment_integration
        
        # 将模块添加到量子系统中
        if self.quantum_system:
            if hasattr(self.quantum_system, "__dict__"):
                self.quantum_system.__dict__["sentiment_integration"] = sentiment_integration
                self.logger.info("情绪集成模块已关联到量子系统")
            else:
                self.logger.warning("无法关联情绪集成模块到量子系统")
        
        self.logger.info("情绪集成模块集成成功")
        return True
    
    def integrate_news_crawler(self, news_crawler):
        """集成新闻爬虫模块
        
        参数:
            news_crawler: 新闻爬虫模块引用
            
        返回:
            bool: 集成是否成功
        """
        if not news_crawler:
            self.logger.error("新闻爬虫模块引用无效")
            return False
            
        self.news_crawler = news_crawler
        
        # 将模块添加到量子系统中
        if self.quantum_system:
            if hasattr(self.quantum_system, "__dict__"):
                self.quantum_system.__dict__["news_crawler"] = news_crawler
                self.logger.info("新闻爬虫模块已关联到量子系统")
            else:
                self.logger.warning("无法关联新闻爬虫模块到量子系统")
        
        self.logger.info("新闻爬虫模块集成成功")
        return True
    
    def initialize_all_modules(self):
        """初始化所有集成的模块
        
        返回:
            bool: 初始化是否成功
        """
        self.logger.info("初始化所有集成模块...")
        
        success = True
        
        # 初始化宇宙意识
        if self.cosmic_consciousness:
            try:
                if not self.cosmic_consciousness.initialized:
                    # 获取场强度
                    field_strength = 0.75
                    if self.quantum_system and hasattr(self.quantum_system, "field_state"):
                        field_strength = self.quantum_system.field_state.get("field_strength", 0.75)
                    
                    # 初始化宇宙意识
                    if not self.cosmic_consciousness.initialize(field_strength=field_strength):
                        self.logger.error("宇宙意识模块初始化失败")
                        success = False
                    
                    # 激活宇宙意识
                    if not self.cosmic_consciousness.activate():
                        self.logger.error("宇宙意识模块激活失败")
                        success = False
            except Exception as e:
                self.logger.error(f"初始化宇宙意识模块时出错: {str(e)}")
                traceback.print_exc()
                success = False
        
        # 初始化新闻爬虫
        if self.news_crawler:
            try:
                if not self.news_crawler.initialized:
                    if not self.news_crawler.initialize():
                        self.logger.error("新闻爬虫模块初始化失败")
                        success = False
            except Exception as e:
                self.logger.error(f"初始化新闻爬虫模块时出错: {str(e)}")
                traceback.print_exc()
                success = False
        
        # 初始化情绪集成
        if self.sentiment_integration:
            try:
                if not self.sentiment_integration.initialized:
                    # 获取组件引用
                    cosmic_consciousness = self.cosmic_consciousness
                    prediction_engine = getattr(self.quantum_system, "prediction_engine", None) if self.quantum_system else None
                    
                    if not self.sentiment_integration.initialize(
                        cosmic_consciousness=cosmic_consciousness,
                        prediction_engine=prediction_engine
                    ):
                        self.logger.error("情绪集成模块初始化失败")
                        success = False
            except Exception as e:
                self.logger.error(f"初始化情绪集成模块时出错: {str(e)}")
                traceback.print_exc()
                success = False
        
        self.initialized = success
        
        if success:
            self.logger.info("所有集成模块初始化成功")
            self.active = True
        else:
            self.logger.warning("部分模块初始化失败")
        
        return success
    
    def enhanced_market_analysis(self, stock_codes=None, days_ahead=5):
        """执行增强版市场分析
        
        参数:
            stock_codes: 要分析的股票代码列表
            days_ahead: 预测未来天数
            
        返回:
            dict: 增强版分析结果
        """
        if not self.active or not self.quantum_system:
            self.logger.warning("系统未激活或量子系统引用无效")
            return None
        
        self.logger.info(f"执行增强版市场分析: {len(stock_codes) if stock_codes else '默认'}股票")
        
        try:
            # 获取新闻数据
            news_data = None
            if self.news_crawler:
                # 获取财经新闻
                news_data = self.news_crawler.fetch_financial_news(days=3, limit=50)
                # 获取社交媒体数据
                social_data = self.news_crawler.fetch_social_sentiment(stock_codes=stock_codes, days=2, limit=100)
                # 获取官方公告
                announcements = self.news_crawler.fetch_official_announcements(days=7, limit=20)
                
                # 合并所有新闻数据
                news_data.extend(social_data)
                news_data.extend(announcements)
            
            # 获取数据源插件
            tushare_plugin = self.quantum_system.get_data_plugin("tushare") if hasattr(self.quantum_system, "get_data_plugin") else None
            
            if not tushare_plugin:
                self.logger.error("无法获取TuShare数据插件")
                return None
            
            # 获取市场概况
            market_data = tushare_plugin.get_market_overview()
            
            # 获取股票列表
            if not stock_codes:
                # 使用默认股票列表
                stock_list = tushare_plugin.get_stock_list()
                stock_codes = [stock["code"] for stock in stock_list[:10]]  # 取前10只股票
            
            # 获取股票数据
            stock_data_dict = {}
            for code in stock_codes:
                data = tushare_plugin.get_stock_data(code)
                if data:
                    stock_data_dict[code] = data
            
            # 分析市场能量场
            market_energy = None
            if self.cosmic_consciousness:
                market_energy = self.cosmic_consciousness.detect_market_energy(market_data, stock_data_dict)
            
            # 执行基础分析和预测
            integration_module = getattr(self.quantum_system, "integration_module", None)
            if not integration_module:
                self.logger.error("无法获取量子集成模块")
                return None
            
            results = integration_module.analyze_and_predict(
                market_data, stock_data_dict, days_ahead=days_ahead
            )
            
            # 增强市场分析
            if results and self.sentiment_integration:
                # 增强市场分析
                market_analysis = results.get("market_analysis", {})
                enhanced_market_analysis = self.sentiment_integration.enhance_market_analysis(market_analysis, news_data)
                results["market_analysis"] = enhanced_market_analysis
                
                # 增强股票预测
                enhanced_predictions = {}
                for code, prediction in results.get("stock_predictions", {}).items():
                    enhanced_prediction = self.sentiment_integration.enhance_prediction(
                        code, prediction, news_data, market_energy
                    )
                    enhanced_predictions[code] = enhanced_prediction
                
                results["stock_predictions"] = enhanced_predictions
                results["sentiment_enhanced"] = True
                
                # 添加能量场分析
                if market_energy:
                    results["energy_field_analysis"] = market_energy
                
                # 添加新闻摘要
                if self.news_crawler:
                    news_summary = self.news_crawler.get_market_news_summary(days=3)
                    results["news_summary"] = news_summary
            
            # 生成增强版报告
            report_file = None
            if results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = os.path.join("reports", f"enhanced_analysis_report_{timestamp}.md")
                report = self._generate_enhanced_report(results, report_file)
                results["report_file"] = report_file
            
            self.logger.info("增强版市场分析完成")
            return results
            
        except Exception as e:
            self.logger.error(f"增强版市场分析失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_enhanced_report(self, results, output_file=None):
        """生成增强版分析报告"""
        if not results:
            self.logger.warning("无法生成报告: 缺少结果数据")
            return None
        
        self.logger.info("生成增强版分析报告...")
        
        try:
            # 提取数据
            timestamp = results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            market_analysis = results.get('market_analysis', {})
            stock_analyses = results.get('stock_analyses', {})
            stock_predictions = results.get('stock_predictions', {})
            energy_field = results.get('energy_field_analysis', {})
            news_summary = results.get('news_summary', {})
            
            # 创建报告内容
            report = f"# 超神量子系统增强版分析报告\n\n"
            report += f"**生成时间:** {timestamp}\n\n"
            
            # 市场概述
            report += "## 市场概述\n\n"
            
            if market_analysis:
                market_phase = market_analysis.get('market_phase', '未知')
                market_prediction = market_analysis.get('prediction', '')
                
                report += f"**市场阶段:** {market_phase}\n\n"
                report += f"**市场预测:** {market_prediction}\n\n"
                
                # 市场能量场
                if energy_field:
                    field_type = energy_field.get('energy_field', {}).get('type', '未知')
                    energy_level = energy_field.get('energy_field', {}).get('energy_level', 0)
                    harmony = energy_field.get('energy_field', {}).get('harmony', 0)
                    forecast = energy_field.get('forecast', '')
                    
                    report += "### 市场能量场分析\n\n"
                    report += f"**能量场类型:** {field_type}\n"
                    report += f"**能量水平:** {energy_level:.2f}\n"
                    report += f"**和谐度:** {harmony:.2f}\n\n"
                    report += f"**能量场预测:** {forecast}\n\n"
                
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
                
                # 宇宙共振分析
                if 'cosmic_resonance' in market_analysis:
                    resonance = market_analysis.get('cosmic_resonance', {})
                    report += "### 宇宙共振分析\n\n"
                    report += "| 共振参数 | 数值 |\n"
                    report += "|------|------|\n"
                    for key, value in resonance.items():
                        # 美化参数名称
                        param_name = key.replace('_', ' ').title()
                        report += f"| {param_name} | {value:.2f} |\n"
                    report += "\n"
            
            # 新闻情绪摘要
            if news_summary:
                report += "## 新闻情绪分析\n\n"
                
                # 数据来源
                sources = news_summary.get('data_sources', {})
                report += f"**分析周期:** 过去{news_summary.get('period_days', 3)}天\n"
                report += f"**数据来源:** 财经新闻({sources.get('financial_news', 0)}条), 社交媒体({sources.get('social_posts', 0)}条), 官方公告({sources.get('official_announcements', 0)}条)\n\n"
                
                # 情绪分布
                sentiment = news_summary.get('sentiment_distribution', {})
                total = sum(sentiment.values())
                
                if total > 0:
                    report += "### 市场情绪分布\n\n"
                    report += "| 情绪类型 | 占比 |\n"
                    report += "|------|------|\n"
                    report += f"| 积极情绪 | {sentiment.get('positive', 0)/total*100:.1f}% |\n"
                    report += f"| 中性情绪 | {sentiment.get('neutral', 0)/total*100:.1f}% |\n"
                    report += f"| 消极情绪 | {sentiment.get('negative', 0)/total*100:.1f}% |\n\n"
                
                # 热门话题
                topics = news_summary.get('top_topics', [])
                if topics:
                    report += "### 热门话题\n\n"
                    for topic in topics[:5]:
                        report += f"- {topic.get('topic', '')}: {topic.get('mentions', 0)}次提及\n"
                    report += "\n"
                
                # 热门股票
                stocks = news_summary.get('hot_stocks', [])
                if stocks:
                    report += "### 热门股票\n\n"
                    for stock in stocks[:5]:
                        report += f"- {stock.get('stock', '')}: {stock.get('mentions', 0)}次提及\n"
                    report += "\n"
                
                # 重要公告
                announcements = news_summary.get('important_announcements', [])
                if announcements:
                    report += "### 重要公告\n\n"
                    for announcement in announcements:
                        report += f"- [{announcement.get('title', '')}] ({announcement.get('source', '')})\n"
                    report += "\n"
            
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
                
                # 预测结果
                report += "#### 预测结果\n\n"
                
                if prediction:
                    overall_change = prediction.get('overall_change', 0)
                    trend = prediction.get('trend', '未知')
                    confidence = prediction.get('average_confidence', 0)
                    
                    report += f"**预测趋势:** {trend}\n"
                    report += f"**预计变动:** {overall_change:.2f}%\n"
                    report += f"**平均信心度:** {confidence:.2f}\n\n"
                    
                    # 情绪增强因子
                    if prediction.get('sentiment_enhanced', False):
                        factors = prediction.get('enhancement_factors', {})
                        report += "**情绪增强因子:**\n\n"
                        report += "| 因子 | 影响值 |\n"
                        report += "|------|------|\n"
                        report += f"| 新闻影响 | {factors.get('news_impact', 0):.3f} |\n"
                        report += f"| 能量场影响 | {factors.get('energy_field', 0):.3f} |\n"
                        report += f"| 情绪偏差 | {factors.get('sentiment_bias', 0):.3f} |\n"
                        report += f"| 总调整量 | {factors.get('total_adjustment', 0):.3f} |\n\n"
                    
                    # 详细预测
                    predictions = prediction.get('predictions', [])
                    if predictions:
                        report += "| 日期 | 预测价格 | 变动百分比 | 信心度 | 情绪因子 |\n"
                        report += "|------|----------|------------|--------|--------|\n"
                        
                        for p in predictions:
                            sentiment_factor = p.get('sentiment_factor', 0)
                            factor_str = f"{sentiment_factor:.3f}" if abs(sentiment_factor) > 0.001 else "0"
                            report += f"| {p.get('date')} | {p.get('price', 0):.2f} | {p.get('change', 0)*100:.2f}% | {p.get('confidence', 0):.2f} | {factor_str} |\n"
                        
                        report += "\n"
                
                # 技术指标
                report += "#### 技术指标\n\n"
                indicators = analysis.get('technical_indicators', {})
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
                
                report += "---\n\n"
            
            # 报告结尾
            report += "## 分析方法说明\n\n"
            report += "本报告由超神量子共生系统2.0生成，使用了以下高级分析方法：\n\n"
            report += "- **量子预测引擎**: 使用高维量子算法进行价格预测\n"
            report += "- **市场分析器**: 对技术指标和市场状态进行综合分析\n"
            report += "- **宇宙意识模块**: 感知市场能量场和共振状态\n"
            report += "- **情绪分析系统**: 分析新闻和社交媒体情绪对市场的影响\n\n"
            
            report += "*免责声明：本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。*\n"
            
            # 如果指定输出文件，保存到文件
            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"增强版报告已保存到: {output_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成增强版报告失败: {str(e)}")
            traceback.print_exc()
            return None 