#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子策略分析工具 - 命令行工具用于快速分析量子策略
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QuantumAnalyzer")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量子策略分析工具")
    
    parser.add_argument("strategy_id", 
                      help="要分析的策略ID")
    
    parser.add_argument("--quantum", action="store_true",
                      help="使用量子增强分析")
    
    parser.add_argument("--backtest-days", type=int, default=252,
                      help="回测天数，默认252个交易日（约一年）")
    
    parser.add_argument("--save", action="store_true",
                      help="保存分析结果")
    
    parser.add_argument("--compare", type=str,
                      help="与另一个策略进行比较")
    
    parser.add_argument("--detail", action="store_true",
                      help="显示详细分析")
    
    return parser.parse_args()

def load_strategy(strategy_id, days=252):
    """加载策略数据"""
    try:
        # 尝试导入回测管理器
        from quantum_core.backtest_manager import BacktestManager
        
        logger.info(f"加载策略 '{strategy_id}' 的数据")
        
        # 创建回测管理器
        backtest_manager = BacktestManager()
        
        # 加载回测结果
        result = backtest_manager.load_backtest_result(strategy_id)
        
        if result['status'] == 'success':
            data = result['data']
            
            # 如果指定了天数，截取最近的数据
            if 'equity_curve' in data:
                equity_curve = data['equity_curve']
                
                if isinstance(equity_curve, dict) and 'values' in equity_curve:
                    values = equity_curve['values']
                    if len(values) > days:
                        equity_curve['values'] = values[-days:]
                        if 'dates' in equity_curve:
                            equity_curve['dates'] = equity_curve['dates'][-days:]
                    data['equity_curve'] = equity_curve
                    
                elif isinstance(equity_curve, list):
                    if len(equity_curve) > days:
                        data['equity_curve'] = equity_curve[-days:]
                        
                elif isinstance(equity_curve, pd.Series):
                    if len(equity_curve) > days:
                        data['equity_curve'] = equity_curve.iloc[-days:]
            
            logger.info(f"成功加载策略 '{strategy_id}' 的数据")
            return data
        else:
            logger.error(f"无法加载策略 '{strategy_id}' 的数据: {result['message']}")
            return None
    except ImportError:
        logger.error("无法导入回测管理器，请确保量子核心已安装")
        return None
    except Exception as e:
        logger.error(f"加载策略时出错: {str(e)}")
        return None

def analyze_strategy(strategy_data, strategy_id, use_quantum=False):
    """分析策略性能"""
    try:
        # 导入策略评估器
        from quantum_core.strategy_evaluator import StrategyEvaluator
        
        logger.info(f"分析策略 '{strategy_id}'")
        
        if not strategy_data or 'equity_curve' not in strategy_data:
            logger.error(f"策略 '{strategy_id}' 缺少权益曲线数据")
            return None
        
        # 创建评估器
        evaluator = StrategyEvaluator(use_quantum_enhancement=use_quantum)
        evaluator.start()
        
        # 准备数据
        equity_curve = strategy_data['equity_curve']
        if isinstance(equity_curve, dict):
            equity_curve = pd.Series(equity_curve['values'], 
                                   index=equity_curve.get('dates'))
        elif isinstance(equity_curve, list):
            equity_curve = pd.Series(equity_curve)
        
        trades = strategy_data.get('trades', [])
        
        # 评估策略
        result = evaluator.evaluate_strategy(
            strategy_id=strategy_id,
            equity_curve=equity_curve,
            trades=trades,
            use_quantum=use_quantum
        )
        
        # 停止评估器
        evaluator.stop()
        
        logger.info(f"完成策略 '{strategy_id}' 的分析")
        return result
    except ImportError:
        logger.error("无法导入策略评估器，请确保量子核心已安装")
        return None
    except Exception as e:
        logger.error(f"分析策略时出错: {str(e)}")
        return None

def compare_strategies(strategy1_data, strategy1_id, strategy2_data, strategy2_id, use_quantum=False):
    """比较两个策略的性能"""
    try:
        # 导入策略评估器
        from quantum_core.strategy_evaluator import StrategyEvaluator
        
        logger.info(f"比较策略 '{strategy1_id}' 和 '{strategy2_id}'")
        
        if not strategy1_data or not strategy2_data:
            logger.error("缺少策略数据")
            return None
        
        # 创建评估器
        evaluator = StrategyEvaluator(use_quantum_enhancement=use_quantum)
        evaluator.start()
        
        # 准备策略1数据
        equity_curve1 = strategy1_data['equity_curve']
        if isinstance(equity_curve1, dict):
            equity_curve1 = pd.Series(equity_curve1['values'], 
                                    index=equity_curve1.get('dates'))
        elif isinstance(equity_curve1, list):
            equity_curve1 = pd.Series(equity_curve1)
        
        trades1 = strategy1_data.get('trades', [])
        
        # 准备策略2数据
        equity_curve2 = strategy2_data['equity_curve']
        if isinstance(equity_curve2, dict):
            equity_curve2 = pd.Series(equity_curve2['values'], 
                                    index=equity_curve2.get('dates'))
        elif isinstance(equity_curve2, list):
            equity_curve2 = pd.Series(equity_curve2)
        
        trades2 = strategy2_data.get('trades', [])
        
        # 评估策略1
        result1 = evaluator.evaluate_strategy(
            strategy_id=strategy1_id,
            equity_curve=equity_curve1,
            trades=trades1,
            use_quantum=use_quantum
        )
        
        # 评估策略2
        result2 = evaluator.evaluate_strategy(
            strategy_id=strategy2_id,
            equity_curve=equity_curve2,
            trades=trades2,
            use_quantum=use_quantum
        )
        
        # 获取相似度
        similarity = evaluator.get_strategy_similarity(strategy1_id, strategy2_id)
        
        # 停止评估器
        evaluator.stop()
        
        logger.info(f"完成策略比较")
        
        return {
            'strategy1': result1,
            'strategy2': result2,
            'similarity': similarity
        }
    except ImportError:
        logger.error("无法导入策略评估器，请确保量子核心已安装")
        return None
    except Exception as e:
        logger.error(f"比较策略时出错: {str(e)}")
        return None

def display_analysis_result(result, detail=False):
    """显示分析结果"""
    if not result or 'metrics' not in result:
        print("没有可显示的分析结果")
        return
    
    print("\n===== 策略分析结果 =====")
    print(f"策略ID: {result.get('strategy_id', 'unknown')}")
    print(f"量子增强: {'是' if result.get('quantum_enhanced', False) else '否'}")
    print("\n主要性能指标:")
    
    # 主要指标
    key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                 'max_drawdown', 'win_rate', 'sortino_ratio', 'volatility']
    
    metrics_data = []
    
    for metric in key_metrics:
        if metric in result['metrics'] and result['metrics'][metric] is not None:
            metric_value = result['metrics'][metric]
            
            # 处理新的字典格式
            if isinstance(metric_value, dict) and 'value' in metric_value:
                value = metric_value['value']
                is_quantum = metric_value.get('quantum_enhanced', False)
                
                # 格式化值
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
                    
                # 添加量子标记
                if is_quantum:
                    formatted_value += " (Q)"
            else:
                # 旧格式
                value = metric_value
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
            
            metrics_data.append([metric, formatted_value])
    
    print(tabulate(metrics_data, headers=['指标', '值'], tablefmt='grid'))
    
    # 详细信息
    if detail and 'metrics' in result:
        print("\n全部性能指标:")
        
        all_metrics = []
        for metric, value in result['metrics'].items():
            if metric not in key_metrics and value is not None:
                # 处理新的字典格式
                if isinstance(value, dict) and 'value' in value:
                    metric_value = value['value']
                    is_quantum = value.get('quantum_enhanced', False)
                    
                    # 格式化值
                    if isinstance(metric_value, (int, float)):
                        formatted_value = f"{metric_value:.4f}"
                    elif isinstance(metric_value, list):
                        formatted_value = f"[{len(metric_value)} values]"
                    else:
                        formatted_value = str(metric_value)
                        
                    # 添加量子标记
                    if is_quantum:
                        formatted_value += " (Q)"
                else:
                    # 旧格式
                    metric_value = value
                    if isinstance(metric_value, (int, float)):
                        formatted_value = f"{metric_value:.4f}"
                    elif isinstance(metric_value, list):
                        formatted_value = f"[{len(metric_value)} values]"
                    else:
                        formatted_value = str(metric_value)
                
                all_metrics.append([metric, formatted_value])
        
        if all_metrics:
            print(tabulate(all_metrics, headers=['指标', '值'], tablefmt='grid'))

def display_comparison_result(comparison, detail=False):
    """显示比较结果"""
    if not comparison or 'strategy1' not in comparison or 'strategy2' not in comparison:
        print("没有可显示的比较结果")
        return
    
    strategy1 = comparison['strategy1']
    strategy2 = comparison['strategy2']
    similarity = comparison.get('similarity', {})
    
    print("\n===== 策略比较结果 =====")
    print(f"策略1: {strategy1.get('strategy_id', 'unknown')}")
    print(f"策略2: {strategy2.get('strategy_id', 'unknown')}")
    
    # 主要指标比较
    key_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 
                 'max_drawdown', 'win_rate', 'sortino_ratio', 'volatility']
    
    comparison_data = []
    
    for metric in key_metrics:
        row = [metric]
        
        # 策略1的指标值
        if ('metrics' in strategy1 and 
            metric in strategy1['metrics'] and 
            strategy1['metrics'][metric] is not None):
            
            metric_value = strategy1['metrics'][metric]
            
            # 处理新的字典格式
            if isinstance(metric_value, dict) and 'value' in metric_value:
                value = metric_value['value']
                is_quantum = metric_value.get('quantum_enhanced', False)
                
                # 格式化值
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
                    
                # 添加量子标记
                if is_quantum:
                    formatted_value += " (Q)"
            else:
                # 旧格式
                value = metric_value
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
                    
            row.append(formatted_value)
        else:
            row.append("N/A")
            
        # 策略2的指标值
        if ('metrics' in strategy2 and 
            metric in strategy2['metrics'] and 
            strategy2['metrics'][metric] is not None):
            
            metric_value = strategy2['metrics'][metric]
            
            # 处理新的字典格式
            if isinstance(metric_value, dict) and 'value' in metric_value:
                value = metric_value['value']
                is_quantum = metric_value.get('quantum_enhanced', False)
                
                # 格式化值
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
                    
                # 添加量子标记
                if is_quantum:
                    formatted_value += " (Q)"
            else:
                # 旧格式
                value = metric_value
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                    if metric == 'total_return' or metric == 'annualized_return':
                        formatted_value = f"{value:.2%}"
                elif isinstance(value, list):
                    formatted_value = f"[{len(value)} values]"
                else:
                    formatted_value = str(value)
                    
            row.append(formatted_value)
        else:
            row.append("N/A")
        
        comparison_data.append(row)
    
    print("\n主要指标比较:")
    print(tabulate(comparison_data, 
                 headers=['指标', strategy1.get('strategy_id', '策略1'), 
                        strategy2.get('strategy_id', '策略2')], 
                 tablefmt='grid'))
    
    # 相似度信息
    if similarity and similarity.get('status') == 'success':
        print("\n策略相似度:")
        similarity_data = [
            ['余弦相似度', f"{similarity.get('cosine_similarity', 0):.4f}"],
            ['欧氏距离', f"{similarity.get('euclidean_distance', 0):.4f}"],
            ['相关系数', f"{similarity.get('correlation', 0):.4f}"],
            ['同聚类', '是' if similarity.get('same_cluster', False) else '否']
        ]
        print(tabulate(similarity_data, headers=['指标', '值'], tablefmt='grid'))
    
    # 详细比较
    if detail:
        print("\n详细指标比较 (仅显示差异显著的指标):")
        
        # 找出两个策略共有的指标
        common_metrics = set()
        if 'metrics' in strategy1 and 'metrics' in strategy2:
            common_metrics = set(strategy1['metrics'].keys()) & set(strategy2['metrics'].keys())
        
        # 过滤掉已显示的关键指标
        common_metrics = [m for m in common_metrics if m not in key_metrics]
        
        if common_metrics:
            detail_data = []
            
            for metric in common_metrics:
                value1 = strategy1['metrics'][metric]
                value2 = strategy2['metrics'][metric]
                
                # 提取值
                if isinstance(value1, dict) and 'value' in value1:
                    val1 = value1['value']
                else:
                    val1 = value1
                    
                if isinstance(value2, dict) and 'value' in value2:
                    val2 = value2['value']
                else:
                    val2 = value2
                
                # 检查是否为可比较的数值
                if (isinstance(val1, (int, float)) and 
                    isinstance(val2, (int, float))):
                    
                    # 计算差异百分比
                    if val1 != 0:
                        diff_pct = abs((val2 - val1) / val1)
                    elif val2 != 0:
                        diff_pct = 1.0  # 从0变为非0的差异设为100%
                    else:
                        diff_pct = 0.0  # 两个0没有差异
                    
                    # 只显示差异超过10%的指标
                    if diff_pct >= 0.1:
                        # 格式化值
                        formatted_val1 = f"{val1:.4f}"
                        formatted_val2 = f"{val2:.4f}"
                        
                        # 添加差异标记
                        if val2 > val1:
                            diff_mark = f"+{diff_pct:.1%}"
                        else:
                            diff_mark = f"-{diff_pct:.1%}"
                            
                        detail_data.append([metric, formatted_val1, formatted_val2, diff_mark])
            
            if detail_data:
                print(tabulate(detail_data, 
                             headers=['指标', strategy1.get('strategy_id', '策略1'), 
                                    strategy2.get('strategy_id', '策略2'), '差异'], 
                             tablefmt='grid'))
            else:
                print("(未发现差异显著的指标)")

def save_analysis_result(result, strategy_id):
    """保存分析结果"""
    if not result:
        logger.error("没有可保存的分析结果")
        return False
    
    try:
        # 确保目录存在
        output_dir = "analysis_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/{strategy_id}_analysis_{timestamp}.json"
        
        # 转换numpy类型
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        # 保存为JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        logger.info(f"分析结果已保存至 {filename}")
        print(f"\n分析结果已保存至 {filename}")
        return True
    except Exception as e:
        logger.error(f"保存分析结果失败: {str(e)}")
        return False

def main():
    """主函数"""
    args = parse_arguments()
    
    # 加载策略数据
    strategy_data = load_strategy(args.strategy_id, args.backtest_days)
    
    if not strategy_data:
        print(f"无法加载策略 '{args.strategy_id}' 的数据")
        return 1
    
    # 比较模式
    if args.compare:
        # 加载对比策略
        compare_strategy_data = load_strategy(args.compare, args.backtest_days)
        
        if not compare_strategy_data:
            print(f"无法加载对比策略 '{args.compare}' 的数据")
            return 1
        
        # 比较策略
        comparison = compare_strategies(
            strategy_data, args.strategy_id,
            compare_strategy_data, args.compare,
            args.quantum
        )
        
        if comparison:
            # 显示比较结果
            display_comparison_result(comparison, args.detail)
            
            # 保存结果
            if args.save:
                save_analysis_result(comparison, f"{args.strategy_id}_vs_{args.compare}")
        else:
            print("策略比较失败")
            return 1
    else:
        # 单策略分析
        result = analyze_strategy(strategy_data, args.strategy_id, args.quantum)
        
        if result:
            # 显示分析结果
            display_analysis_result(result, args.detail)
            
            # 保存结果
            if args.save:
                save_analysis_result(result, args.strategy_id)
        else:
            print("策略分析失败")
            return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n未处理的异常: {str(e)}")
        logger.exception("未处理的异常")
        sys.exit(1) 