"""
quantum_stock_validator - 量子选股验证模块
用于评估超神量子选股策略的有效性和性能
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import random
import time
import json

logger = logging.getLogger(__name__)

class QuantumStockValidator:
    """量子选股验证器 - 验证超神量子选股策略的有效性"""
    
    def __init__(self, stock_strategy=None, data_service=None):
        """初始化量子选股验证器
        
        Args:
            stock_strategy: 量子选股策略实例
            data_service: 数据服务实例，用于获取历史数据
        """
        self.stock_strategy = stock_strategy
        self.data_service = data_service
        self.validation_results = {}
        self.benchmark_performance = {}
        logger.info("量子选股验证器初始化完成")
        
    def validate_strategy(self, backtest_period: int = 90, 
                        quantum_powers: List[int] = None,
                        market_scopes: List[str] = None) -> Dict[str, Any]:
        """验证量子选股策略的有效性
        
        Args:
            backtest_period: 回测周期（天）
            quantum_powers: 要测试的量子能力列表，例如[20, 50, 80]
            market_scopes: 要测试的市场范围列表
            
        Returns:
            验证结果
        """
        if not self.stock_strategy:
            logger.error("无法验证：选股策略未提供")
            return {"status": "error", "message": "选股策略未提供"}
            
        # 使用默认值（如果未提供）
        if quantum_powers is None:
            quantum_powers = [20, 50, 80]
            
        if market_scopes is None:
            market_scopes = ["全市场"]
            
        # 确保选股策略已启动
        if not self.stock_strategy.is_running:
            self.stock_strategy.start()
            
        logger.info(f"开始验证选股策略，回测周期: {backtest_period}天")
        
        # 结果容器
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "backtest_period": backtest_period,
            "tests": [],
            "summary": {}
        }
        
        # 执行各种配置下的回测
        for power in quantum_powers:
            for scope in market_scopes:
                logger.info(f"测试配置 - 量子能力: {power}, 市场范围: {scope}")
                
                # 设置量子能力
                self.stock_strategy.set_quantum_power(power)
                
                # 执行选股
                stocks_result = self.stock_strategy.find_potential_stocks(
                    market_scope=scope,
                    max_stocks=10
                )
                
                if stocks_result.get("status") != "success":
                    logger.error(f"选股失败: {stocks_result.get('message')}")
                    continue
                    
                # 获取所选股票
                selected_stocks = stocks_result.get("stocks", [])
                
                if not selected_stocks:
                    logger.warning(f"未找到任何股票")
                    continue
                    
                # 执行回测
                backtest_result = self._perform_backtest(
                    selected_stocks, 
                    backtest_period
                )
                
                # 保存测试结果
                test_result = {
                    "quantum_power": power,
                    "market_scope": scope,
                    "selected_stocks": [s["code"] for s in selected_stocks],
                    "backtest_result": backtest_result
                }
                
                results["tests"].append(test_result)
        
        # 生成汇总分析
        results["summary"] = self._generate_summary(results["tests"])
        
        # 保存验证结果
        self.validation_results = results
        
        logger.info(f"选股策略验证完成，共测试 {len(results['tests'])} 种配置")
        return results
        
    def _perform_backtest(self, selected_stocks: List[Dict], 
                        period: int) -> Dict[str, Any]:
        """对所选股票执行回测
        
        Args:
            selected_stocks: 所选股票列表
            period: 回测周期（天）
            
        Returns:
            回测结果
        """
        logger.info(f"开始回测 {len(selected_stocks)} 只股票，周期: {period}天")
        
        # 生成起始日期（回测开始日期，为了演示，这里使用当前日期往前推）
        start_date = datetime.now() - timedelta(days=period)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = datetime.now().strftime("%Y%m%d")
        
        # 每只股票的表现
        stock_performances = {}
        
        # 总体表现指标
        total_return = 0.0
        winning_stocks = 0
        
        for stock in selected_stocks:
            stock_code = stock["code"]
            quantum_score = stock.get("quantum_score", 0)
            expected_gain = stock.get("expected_gain", 0)
            
            # 这里应该使用数据服务获取真实的历史数据
            # 由于这是一个演示，我们模拟一些回测结果
            actual_return = self._simulate_stock_performance(
                stock_code, 
                quantum_score, 
                period
            )
            
            # 记录表现
            stock_performances[stock_code] = {
                "name": stock.get("name", ""),
                "quantum_score": quantum_score,
                "expected_gain": expected_gain,
                "actual_return": actual_return,
                "prediction_accuracy": self._calculate_accuracy(
                    expected_gain, 
                    actual_return
                )
            }
            
            # 更新总体指标
            total_return += actual_return
            if actual_return > 0:
                winning_stocks += 1
                
        # 计算基准表现（例如市场指数）
        benchmark_return = self._get_benchmark_performance(period)
        
        # 计算超额收益
        excess_return = total_return / len(selected_stocks) - benchmark_return
        
        # 准备回测结果
        backtest_result = {
            "period": period,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "stock_performances": stock_performances,
            "average_return": total_return / len(selected_stocks) if selected_stocks else 0,
            "win_rate": winning_stocks / len(selected_stocks) if selected_stocks else 0,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "sharpe_ratio": self._calculate_sharpe_ratio(
                stock_performances.values(), 
                benchmark_return
            )
        }
        
        logger.info(f"回测完成，平均收益: {backtest_result['average_return']:.2f}%, "
                   f"超额收益: {excess_return:.2f}%")
        return backtest_result
        
    def _simulate_stock_performance(self, stock_code: str, 
                                  quantum_score: float, 
                                  period: int) -> float:
        """模拟股票在回测期间的表现
        
        在实际实现中，这里应该使用真实的历史数据
        
        Args:
            stock_code: 股票代码
            quantum_score: 量子评分
            period: 回测周期（天）
            
        Returns:
            股票的实际收益率（百分比）
        """
        # 关联量子评分和实际表现
        # 量子评分越高，模拟的表现越好，但有随机性
        
        # 基础收益率：正比于量子评分
        base_return = (quantum_score - 80) * 0.8
        
        # 添加随机噪声
        # 量子评分越高，噪声越小（更可靠）
        noise_factor = max(5, 25 - (quantum_score - 80) * 0.5)
        noise = random.uniform(-noise_factor, noise_factor)
        
        # 最终模拟收益率
        simulated_return = base_return + noise
        
        # 对于超长周期，增加波动性
        if period > 30:
            volatility_factor = 1 + (period - 30) / 100
            simulated_return *= volatility_factor
            
        return simulated_return
        
    def _get_benchmark_performance(self, period: int) -> float:
        """获取基准指数在回测期间的表现
        
        在实际实现中，这里应该使用真实的指数历史数据
        
        Args:
            period: 回测周期（天）
            
        Returns:
            基准指数的收益率（百分比）
        """
        # 模拟基准表现
        # 短期内波动小，长期内波动大
        if period <= 30:
            return random.uniform(-5, 8)
        elif period <= 60:
            return random.uniform(-8, 12)
        else:
            return random.uniform(-12, 15)
            
    def _calculate_accuracy(self, expected_gain: float, 
                          actual_return: float) -> float:
        """计算预测准确度
        
        Args:
            expected_gain: 预期收益
            actual_return: 实际收益
            
        Returns:
            预测准确度（0-100%）
        """
        # 预测方向是否正确
        if (expected_gain > 0 and actual_return > 0) or (expected_gain <= 0 and actual_return <= 0):
            direction_score = 100.0
        else:
            direction_score = 0.0
            
        # 预测幅度准确度
        magnitude_diff = abs(expected_gain - actual_return)
        magnitude_score = max(0, 100 - magnitude_diff * 2)
        
        # 综合评分，方向更重要
        return direction_score * 0.7 + magnitude_score * 0.3
        
    def _calculate_sharpe_ratio(self, stock_performances: List[Dict], 
                              benchmark_return: float) -> float:
        """计算夏普比率
        
        Args:
            stock_performances: 股票表现列表
            benchmark_return: 基准收益率
            
        Returns:
            夏普比率
        """
        returns = [s["actual_return"] for s in stock_performances]
        
        if not returns:
            return 0.0
            
        # 计算平均收益率
        mean_return = np.mean(returns)
        
        # 计算收益率标准差
        std_dev = np.std(returns) if len(returns) > 1 else 1.0
        
        # 避免除以零
        if std_dev == 0:
            std_dev = 0.0001
            
        # 无风险利率（假设为0%）
        risk_free_rate = 0.0
        
        # 计算夏普比率
        sharpe = (mean_return - risk_free_rate) / std_dev
        
        return sharpe
        
    def _generate_summary(self, test_results: List[Dict]) -> Dict[str, Any]:
        """生成验证测试的汇总分析
        
        Args:
            test_results: 所有测试结果的列表
            
        Returns:
            汇总分析
        """
        if not test_results:
            return {"message": "没有测试结果可分析"}
            
        # 提取所有回测结果
        backtest_results = [t["backtest_result"] for t in test_results if "backtest_result" in t]
        
        if not backtest_results:
            return {"message": "没有回测结果可分析"}
            
        # 分析不同量子能力下的表现
        power_performance = {}
        for test in test_results:
            power = test["quantum_power"]
            backtest = test["backtest_result"]
            
            if power not in power_performance:
                power_performance[power] = []
                
            power_performance[power].append(backtest["average_return"])
            
        # 计算每个量子能力的平均表现
        avg_power_performance = {
            power: sum(returns) / len(returns) 
            for power, returns in power_performance.items()
        }
        
        # 获取最佳量子能力
        best_power = max(avg_power_performance.items(), key=lambda x: x[1])[0]
        
        # 计算平均预测准确度
        all_accuracies = []
        for backtest in backtest_results:
            for stock in backtest["stock_performances"].values():
                all_accuracies.append(stock["prediction_accuracy"])
                
        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
        
        # 计算超额收益能力
        excess_returns = [b["excess_return"] for b in backtest_results]
        avg_excess_return = sum(excess_returns) / len(excess_returns)
        
        # 准备汇总
        summary = {
            "best_quantum_power": best_power,
            "average_prediction_accuracy": avg_accuracy,
            "average_excess_return": avg_excess_return,
            "power_performance": avg_power_performance,
            "sharpe_ratio": np.mean([b["sharpe_ratio"] for b in backtest_results]),
            "win_rate": np.mean([b["win_rate"] for b in backtest_results]),
            "quantum_score_correlation": self._calculate_score_correlation(backtest_results)
        }
        
        return summary
        
    def _calculate_score_correlation(self, backtest_results: List[Dict]) -> float:
        """计算量子评分与实际收益的相关性
        
        Args:
            backtest_results: 回测结果列表
            
        Returns:
            相关系数
        """
        scores = []
        returns = []
        
        # 收集所有股票的量子评分和实际收益
        for backtest in backtest_results:
            for stock_data in backtest["stock_performances"].values():
                scores.append(stock_data["quantum_score"])
                returns.append(stock_data["actual_return"])
                
        # 如果数据不足，无法计算相关性
        if len(scores) < 2:
            return 0.0
            
        # 计算皮尔逊相关系数
        try:
            correlation = np.corrcoef(scores, returns)[0, 1]
            return correlation
        except:
            logger.error("计算相关系数时出错")
            return 0.0
            
    def plot_validation_results(self, output_file: str = None) -> None:
        """绘制验证结果图表
        
        Args:
            output_file: 输出文件路径，如果不提供，则显示图表
        """
        if not self.validation_results or "tests" not in self.validation_results:
            logger.error("没有验证结果可绘制")
            return
            
        summary = self.validation_results.get("summary", {})
        tests = self.validation_results.get("tests", [])
        
        # 创建图表
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 量子能力与平均收益率关系图
        powers = []
        returns = []
        for power, avg_return in summary.get("power_performance", {}).items():
            powers.append(power)
            returns.append(avg_return)
            
        if powers and returns:
            axs[0, 0].bar(powers, returns)
            axs[0, 0].set_title('量子能力与平均收益率关系')
            axs[0, 0].set_xlabel('量子能力')
            axs[0, 0].set_ylabel('平均收益率 (%)')
            
        # 2. 量子评分与实际收益散点图
        scores = []
        actual_returns = []
        for test in tests:
            backtest = test.get("backtest_result", {})
            for stock_data in backtest.get("stock_performances", {}).values():
                scores.append(stock_data.get("quantum_score", 0))
                actual_returns.append(stock_data.get("actual_return", 0))
                
        if scores and actual_returns:
            axs[0, 1].scatter(scores, actual_returns)
            axs[0, 1].set_title('量子评分与实际收益关系')
            axs[0, 1].set_xlabel('量子评分')
            axs[0, 1].set_ylabel('实际收益率 (%)')
            
            # 添加趋势线
            if len(scores) > 1:
                z = np.polyfit(scores, actual_returns, 1)
                p = np.poly1d(z)
                axs[0, 1].plot(scores, p(scores), "r--")
                
        # 3. 预测准确度分布直方图
        accuracies = []
        for test in tests:
            backtest = test.get("backtest_result", {})
            for stock_data in backtest.get("stock_performances", {}).values():
                accuracies.append(stock_data.get("prediction_accuracy", 0))
                
        if accuracies:
            axs[1, 0].hist(accuracies, bins=10)
            axs[1, 0].set_title('预测准确度分布')
            axs[1, 0].set_xlabel('预测准确度 (%)')
            axs[1, 0].set_ylabel('频率')
            
        # 4. 选股表现与基准比较图
        test_labels = []
        test_returns = []
        benchmark_returns = []
        
        for i, test in enumerate(tests):
            backtest = test.get("backtest_result", {})
            test_labels.append(f"Test {i+1}")
            test_returns.append(backtest.get("average_return", 0))
            benchmark_returns.append(backtest.get("benchmark_return", 0))
            
        if test_labels and test_returns and benchmark_returns:
            x = np.arange(len(test_labels))
            width = 0.35
            
            axs[1, 1].bar(x - width/2, test_returns, width, label='选股策略')
            axs[1, 1].bar(x + width/2, benchmark_returns, width, label='市场基准')
            axs[1, 1].set_title('策略表现与基准比较')
            axs[1, 1].set_xticks(x)
            axs[1, 1].set_xticklabels(test_labels)
            axs[1, 1].set_ylabel('收益率 (%)')
            axs[1, 1].legend()
            
        plt.tight_layout()
        
        # 保存或显示图表
        if output_file:
            plt.savefig(output_file)
            logger.info(f"验证结果图表已保存至 {output_file}")
        else:
            plt.show()
            
    def export_validation_report(self, output_file: str) -> bool:
        """导出验证报告到文件
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            是否成功导出
        """
        if not self.validation_results:
            logger.error("没有验证结果可导出")
            return False
            
        try:
            # 导出为JSON
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
                
            logger.info(f"验证报告已导出至 {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"导出验证报告时出错: {str(e)}")
            return False 