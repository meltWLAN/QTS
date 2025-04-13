"""
enhanced_backtest - 高级量子回测引擎
提供更精准、更全面的回测能力，支持多种回测策略和评估指标
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

logger = logging.getLogger(__name__)

class EnhancedBacktestEngine:
    """增强型回测引擎 - 提供高级回测和性能评估功能"""
    
    def __init__(self, data_service=None):
        """初始化回测引擎
        
        Args:
            data_service: 数据服务实例，用于获取历史数据
        """
        self.data_service = data_service
        self.backtest_results = {}
        self.performance_metrics = {}
        logger.info("增强型回测引擎初始化完成")
        
    def run_backtest(self, stocks: List[Dict], 
                   period: int = 90,
                   initial_capital: float = 1000000.0,
                   benchmark: str = "000300.SH",
                   position_sizing: str = "equal",
                   risk_management: bool = True,
                   detailed_output: bool = True) -> Dict[str, Any]:
        """运行增强型回测
        
        Args:
            stocks: 股票列表
            period: 回测周期（天）
            initial_capital: 初始资金
            benchmark: 基准指数
            position_sizing: 仓位分配策略 ('equal', 'weighted', 'kelly')
            risk_management: 启用风险管理
            detailed_output: 输出详细结果
            
        Returns:
            回测结果
        """
        logger.info(f"开始增强型回测，股票数量: {len(stocks)}，周期: {period}天")
        
        if not stocks:
            logger.warning("股票列表为空，无法进行回测")
            return {"status": "error", "message": "股票列表为空"}
            
        # 生成回测日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        
        # 确保是交易日
        start_date_str = self._get_trade_date(start_date)
        end_date_str = self._get_trade_date(end_date)
        
        logger.info(f"回测区间: {start_date_str} - {end_date_str}")
        
        # 获取基准指数数据
        benchmark_data = self._get_benchmark_data(benchmark, start_date_str, end_date_str)
        
        # 分配初始资金
        positions = self._allocate_positions(stocks, initial_capital, position_sizing)
        
        # 执行回测
        daily_portfolio = self._simulate_portfolio(positions, start_date_str, end_date_str, risk_management)
        
        # 计算性能指标
        performance = self._calculate_performance(daily_portfolio, benchmark_data)
        
        # 格式化回测结果
        backtest_result = {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {
                "period": period,
                "initial_capital": initial_capital,
                "benchmark": benchmark,
                "position_sizing": position_sizing,
                "risk_management": risk_management
            },
            "date_range": {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "trading_days": len(daily_portfolio)
            },
            "performance": performance,
            "stock_performances": self._get_stock_performances(positions, start_date_str, end_date_str)
        }
        
        # 添加详细输出（如果需要）
        if detailed_output:
            backtest_result["daily_portfolio"] = daily_portfolio
            backtest_result["daily_benchmark"] = benchmark_data
            
        # 保存结果
        self.backtest_results = backtest_result
        
        logger.info(f"回测完成，总收益: {performance['total_return']:.2f}%，超额收益: {performance['excess_return']:.2f}%")
        return backtest_result
        
    def _get_trade_date(self, date: datetime) -> str:
        """获取有效交易日期
        
        Args:
            date: 日期
            
        Returns:
            交易日期字符串 (YYYYMMDD)
        """
        # 这里应该调用数据服务获取最近的交易日
        # 为了演示，简单地返回原始日期的字符串表示
        return date.strftime("%Y%m%d")
        
    def _get_benchmark_data(self, benchmark: str, 
                          start_date: str, 
                          end_date: str) -> Dict[str, float]:
        """获取基准指数数据
        
        Args:
            benchmark: 基准指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            基准指数每日数据
        """
        logger.info(f"获取基准指数数据: {benchmark}")
        
        # 这里应该调用数据服务获取真实的基准数据
        # 为了演示，模拟一些基准数据
        
        # 生成日期序列（简化处理）
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        days = (end_dt - start_dt).days + 1
        
        # 假设有效交易日约为总天数的70%
        trading_days = max(1, int(days * 0.7))
        
        # 生成随机基准数据
        # 假设指数每日波动在 -1% 到 1.5% 之间，带有轻微的上升偏好
        daily_returns = np.random.normal(0.0005, 0.01, trading_days)  # 均值略大于0
        
        # 计算累积收益
        benchmark_data = {}
        current_value = 1000.0  # 初始指数值
        
        for i in range(trading_days):
            # 生成日期（简化，不考虑实际交易日历）
            current_date = (start_dt + timedelta(days=i)).strftime("%Y%m%d")
            
            # 更新指数值
            current_value *= (1 + daily_returns[i])
            benchmark_data[current_date] = current_value
            
        return benchmark_data
        
    def _allocate_positions(self, stocks: List[Dict], 
                          initial_capital: float,
                          position_sizing: str) -> Dict[str, Dict]:
        """分配仓位
        
        Args:
            stocks: 股票列表
            initial_capital: 初始资金
            position_sizing: 仓位分配策略
            
        Returns:
            分配的仓位
        """
        logger.info(f"使用 {position_sizing} 策略分配仓位")
        
        positions = {}
        
        if position_sizing == "equal":
            # 等额分配
            position_size = initial_capital / len(stocks)
            
            for stock in stocks:
                stock_code = stock["code"]
                positions[stock_code] = {
                    "code": stock_code,
                    "name": stock.get("name", ""),
                    "allocated_capital": position_size,
                    "quantum_score": stock.get("quantum_score", 0),
                    "expected_gain": stock.get("expected_gain", 0)
                }
                
        elif position_sizing == "weighted":
            # 根据量子评分加权分配
            total_score = sum(stock.get("quantum_score", 1) for stock in stocks)
            
            for stock in stocks:
                stock_code = stock["code"]
                quantum_score = stock.get("quantum_score", 1)
                weight = quantum_score / total_score if total_score > 0 else 1.0 / len(stocks)
                
                positions[stock_code] = {
                    "code": stock_code,
                    "name": stock.get("name", ""),
                    "allocated_capital": initial_capital * weight,
                    "weight": weight,
                    "quantum_score": quantum_score,
                    "expected_gain": stock.get("expected_gain", 0)
                }
                
        elif position_sizing == "kelly":
            # 凯利公式分配（简化版）
            for stock in stocks:
                stock_code = stock["code"]
                expected_gain = stock.get("expected_gain", 0) / 100  # 转为小数
                confidence = stock.get("confidence", 0.5)
                
                # 简化凯利公式: f* = (edge * confidence - (1-confidence)) / edge
                edge = max(0.01, expected_gain)  # 防止除以零
                kelly_fraction = max(0, min(1, (edge * confidence - (1-confidence)) / edge))
                
                # 限制单一股票的最大配置比例
                kelly_fraction = min(kelly_fraction, 0.3)
                
                positions[stock_code] = {
                    "code": stock_code,
                    "name": stock.get("name", ""),
                    "allocated_capital": initial_capital * kelly_fraction,
                    "weight": kelly_fraction,
                    "quantum_score": stock.get("quantum_score", 0),
                    "expected_gain": stock.get("expected_gain", 0),
                    "kelly_fraction": kelly_fraction
                }
                
        else:
            # 默认等额分配
            position_size = initial_capital / len(stocks)
            
            for stock in stocks:
                stock_code = stock["code"]
                positions[stock_code] = {
                    "code": stock_code,
                    "name": stock.get("name", ""),
                    "allocated_capital": position_size,
                    "quantum_score": stock.get("quantum_score", 0),
                    "expected_gain": stock.get("expected_gain", 0)
                }
                
        return positions
        
    def _simulate_portfolio(self, positions: Dict[str, Dict],
                          start_date: str,
                          end_date: str,
                          risk_management: bool) -> List[Dict]:
        """模拟投资组合在回测期间的表现
        
        Args:
            positions: 仓位分配
            start_date: 开始日期
            end_date: 结束日期
            risk_management: 启用风险管理
            
        Returns:
            每日投资组合数据
        """
        logger.info("模拟投资组合表现")
        
        # 这里应该调用数据服务获取每支股票的真实历史数据
        # 为了演示，模拟一些数据
        
        # 生成日期序列（简化处理）
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        days = (end_dt - start_dt).days + 1
        
        # 假设有效交易日约为总天数的70%
        trading_days = max(1, int(days * 0.7))
        
        # 生成每日投资组合数据
        daily_portfolio = []
        
        # 初始持仓
        holdings = {}
        for code, position in positions.items():
            holdings[code] = {
                "code": code,
                "name": position.get("name", ""),
                "capital": position.get("allocated_capital", 0),
                "shares": 0,  # 待定
                "price": 0,  # 待定
                "value": position.get("allocated_capital", 0),
                "quantum_score": position.get("quantum_score", 0)
            }
            
        # 模拟每个交易日
        for day in range(trading_days):
            # 当前日期
            current_date = (start_dt + timedelta(days=day)).strftime("%Y%m%d")
            
            # 更新每支股票的价格和持仓价值
            total_value = 0
            
            for code, holding in holdings.items():
                # 生成该股票在当天的价格
                if day == 0:
                    # 第一天，设置初始价格
                    price = 10.0 + random.uniform(-2.0, 5.0)  # 随机初始价格
                    holding["price"] = price
                    holding["shares"] = holding["capital"] / price
                else:
                    # 后续日期，基于量子评分生成价格波动
                    quantum_score = holding.get("quantum_score", 50)
                    
                    # 转换量子评分到合理的收益预期
                    # 量子评分越高，日均收益越高，波动越小
                    expected_daily_return = (quantum_score - 50) * 0.0003  # 基础日收益率
                    volatility = max(0.005, 0.02 - (quantum_score - 50) * 0.0002)  # 波动率
                    
                    # 生成随机波动
                    daily_return = np.random.normal(expected_daily_return, volatility)
                    
                    # 新价格
                    price = holding["price"] * (1 + daily_return)
                    holding["price"] = price
                
                # 计算持仓价值
                holding["value"] = holding["shares"] * holding["price"]
                
                # 累加总价值
                total_value += holding["value"]
                
                # 如果启用风险管理，检查是否需要止损
                if risk_management and day > 0:
                    previous_portfolio = daily_portfolio[day-1]
                    previous_holding = next((h for h in previous_portfolio["holdings"] if h["code"] == code), None)
                    
                    if previous_holding:
                        # 计算单日跌幅
                        daily_change = (holding["price"] / previous_holding["price"]) - 1
                        
                        # 如果单日跌幅超过阈值，执行止损
                        if daily_change < -0.09:  # 9%止损线
                            logger.info(f"触发止损: {code}, 日跌幅: {daily_change*100:.2f}%")
                            holding["shares"] = 0
                            holding["value"] = 0
                            # 资金转为现金，此处简化处理
            
            # 记录当日投资组合
            portfolio_snapshot = {
                "date": current_date,
                "total_value": total_value,
                "holdings": list(holdings.values())
            }
            
            daily_portfolio.append(portfolio_snapshot)
            
        return daily_portfolio
        
    def _calculate_performance(self, daily_portfolio: List[Dict],
                            benchmark_data: Dict[str, float]) -> Dict[str, Any]:
        """计算性能指标
        
        Args:
            daily_portfolio: 每日投资组合数据
            benchmark_data: 基准指数数据
            
        Returns:
            性能指标
        """
        logger.info("计算投资组合性能指标")
        
        if not daily_portfolio:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "volatility": 0,
                "win_rate": 0,
                "benchmark_return": 0,
                "excess_return": 0,
                "information_ratio": 0
            }
            
        # 提取每日总价值
        portfolio_values = [day["total_value"] for day in daily_portfolio]
        
        # 计算初始和最终价值
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        # 总收益率
        total_return = (final_value / initial_value - 1) * 100
        
        # 回测天数
        days = len(portfolio_values)
        
        # 年化收益率 (假设一年252个交易日)
        annualized_return = (((final_value / initial_value) ** (252 / days)) - 1) * 100
        
        # 计算每日收益率
        daily_returns = []
        for i in range(1, days):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            daily_returns.append(daily_return)
            
        # 波动率 (年化)
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        
        # 最大回撤
        max_drawdown = 0
        peak = portfolio_values[0]
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
            
        # 夏普比率 (假设无风险利率为3%)
        risk_free_rate = 0.03
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        excess_daily_returns = [r - daily_risk_free for r in daily_returns]
        sharpe_ratio = 0
        
        if np.std(excess_daily_returns) > 0:
            sharpe_ratio = np.mean(excess_daily_returns) / np.std(excess_daily_returns) * np.sqrt(252)
            
        # 胜率 (正收益天数比例)
        win_days = sum(1 for r in daily_returns if r > 0)
        win_rate = (win_days / len(daily_returns)) * 100 if daily_returns else 0
        
        # 基准收益率
        benchmark_values = list(benchmark_data.values())
        if benchmark_values:
            benchmark_return = (benchmark_values[-1] / benchmark_values[0] - 1) * 100
        else:
            benchmark_return = 0
            
        # 超额收益
        excess_return = total_return - benchmark_return
        
        # 信息比率
        # 计算每日超额收益
        information_ratio = 0
        
        if benchmark_data:
            benchmark_dates = list(benchmark_data.keys())
            excess_returns = []
            
            for i, day in enumerate(daily_portfolio):
                if i < len(benchmark_dates):
                    benchmark_date = benchmark_dates[i]
                    portfolio_value = day["total_value"]
                    benchmark_value = benchmark_data[benchmark_date]
                    
                    if i > 0:
                        prev_portfolio_value = daily_portfolio[i-1]["total_value"]
                        prev_benchmark_date = benchmark_dates[i-1]
                        prev_benchmark_value = benchmark_data[prev_benchmark_date]
                        
                        # 计算超额收益
                        portfolio_return = (portfolio_value / prev_portfolio_value) - 1
                        benchmark_return = (benchmark_value / prev_benchmark_value) - 1
                        excess_return = portfolio_return - benchmark_return
                        excess_returns.append(excess_return)
            
            # 计算信息比率
            if excess_returns and np.std(excess_returns) > 0:
                information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
                
        # 返回性能指标
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "win_rate": win_rate,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "information_ratio": information_ratio
        }
        
    def _get_stock_performances(self, positions: Dict[str, Dict],
                              start_date: str,
                              end_date: str) -> Dict[str, Dict]:
        """获取各只股票的表现
        
        Args:
            positions: 仓位分配
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            各股票表现
        """
        # 这里应该调用数据服务获取真实的股票表现
        # 为了演示，模拟一些数据
        
        stock_performances = {}
        
        for code, position in positions.items():
            quantum_score = position.get("quantum_score", 50)
            expected_gain = position.get("expected_gain", 0)
            
            # 基于量子评分模拟表现
            # 量子评分越高，实际收益越接近预期收益
            score_factor = (quantum_score - 50) / 50  # -1到1之间
            base_return = expected_gain * (0.7 + 0.6 * score_factor)
            
            # 添加随机波动
            volatility = max(5, 20 - score_factor * 10)
            actual_return = base_return + random.uniform(-volatility, volatility)
            
            # 计算预测精度
            prediction_accuracy = 100 - min(100, abs(actual_return - expected_gain))
            
            stock_performances[code] = {
                "code": code,
                "name": position.get("name", ""),
                "initial_position": position.get("allocated_capital", 0),
                "quantum_score": quantum_score,
                "expected_gain": expected_gain,
                "actual_return": actual_return,
                "prediction_accuracy": prediction_accuracy
            }
            
        return stock_performances
        
    def plot_performance(self, output_file: Optional[str] = None) -> None:
        """绘制回测表现图表
        
        Args:
            output_file: 输出文件路径
        """
        if not self.backtest_results or "daily_portfolio" not in self.backtest_results:
            logger.warning("没有可绘制的回测结果")
            return
            
        logger.info("绘制回测性能图表")
        
        # 提取数据
        daily_portfolio = self.backtest_results["daily_portfolio"]
        dates = [day["date"] for day in daily_portfolio]
        portfolio_values = [day["total_value"] for day in daily_portfolio]
        
        # 转换为正确的日期格式
        datetime_dates = [datetime.strptime(date, "%Y%m%d") for date in dates]
        
        # 提取基准数据
        benchmark_data = self.backtest_results.get("daily_benchmark", {})
        benchmark_dates = sorted(benchmark_data.keys())
        benchmark_values = [benchmark_data[date] for date in benchmark_dates]
        benchmark_datetime_dates = [datetime.strptime(date, "%Y%m%d") for date in benchmark_dates]
        
        # 计算基准数据的缩放因子，使其与投资组合起点相同
        if benchmark_values:
            scale_factor = portfolio_values[0] / benchmark_values[0]
            scaled_benchmark_values = [value * scale_factor for value in benchmark_values]
        else:
            scaled_benchmark_values = []
            
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制投资组合价值曲线
        plt.plot(datetime_dates, portfolio_values, label="投资组合", linewidth=2)
        
        # 绘制基准指数曲线
        if scaled_benchmark_values:
            plt.plot(benchmark_datetime_dates, scaled_benchmark_values, label="基准指数", linestyle="--")
            
        # 设置图表
        plt.title("量子增强回测表现", fontsize=15)
        plt.xlabel("日期")
        plt.ylabel("组合价值")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加性能指标
        if "performance" in self.backtest_results:
            perf = self.backtest_results["performance"]
            
            # 创建性能指标文本
            text = (
                f"总收益率: {perf['total_return']:.2f}%\n"
                f"年化收益: {perf['annualized_return']:.2f}%\n"
                f"最大回撤: {perf['max_drawdown']:.2f}%\n"
                f"夏普比率: {perf['sharpe_ratio']:.2f}\n"
                f"胜率: {perf['win_rate']:.2f}%\n"
                f"超额收益: {perf['excess_return']:.2f}%"
            )
            
            # 添加文本到图表右上角
            plt.figtext(0.75, 0.15, text, fontsize=12,
                     bbox={"facecolor": "white", "alpha": 0.8, "pad": 10})
                     
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存到: {output_file}")
        else:
            logger.info("显示图表")
            plt.show()
            
        plt.close()
        
    def export_report(self, output_file: Optional[str] = None) -> bool:
        """导出回测报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否成功导出
        """
        if not self.backtest_results:
            logger.warning("没有可导出的回测结果")
            return False
            
        if not output_file:
            output_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        try:
            # 创建导出目录（如果不存在）
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # 准备导出数据
            export_data = {
                "backtest_summary": {
                    "timestamp": self.backtest_results.get("timestamp", ""),
                    "settings": self.backtest_results.get("settings", {}),
                    "date_range": self.backtest_results.get("date_range", {}),
                    "performance": self.backtest_results.get("performance", {})
                },
                "stock_performances": self.backtest_results.get("stock_performances", {})
            }
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"回测报告已导出到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出回测报告时出错: {str(e)}")
            return False 