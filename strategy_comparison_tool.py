#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略对比分析工具 - 对比多个量子交易策略的性能
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StrategyComparison")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="策略对比分析工具")
    
    parser.add_argument("--strategies", type=str, required=True, 
                      help="要对比的策略ID，用逗号分隔")
    
    parser.add_argument("--output", type=str, default="comparison_report.html",
                      help="输出报告文件名")
    
    parser.add_argument("--metrics", type=str, 
                      help="要对比的指标，用逗号分隔")
    
    parser.add_argument("--period", type=str, default="all",
                      help="回测周期 (all, 1y, 6m, 3m, 1m)")
    
    parser.add_argument("--quantum", action="store_true",
                      help="使用量子增强评估")
    
    parser.add_argument("--dimension-analysis", action="store_true",
                      help="执行多维分析")
    
    parser.add_argument("--plot", action="store_true",
                      help="生成图表")
    
    return parser.parse_args()

def load_strategies(strategy_ids):
    """加载策略回测结果"""
    from quantum_core.backtest_manager import BacktestManager
    
    logger.info(f"加载策略数据: {strategy_ids}")
    
    backtest_manager = BacktestManager()
    strategies_data = {}
    
    for strategy_id in strategy_ids:
        try:
            # 加载回测结果
            result = backtest_manager.load_backtest_result(strategy_id)
            
            if result['status'] == 'success':
                strategies_data[strategy_id] = result['data']
                logger.info(f"成功加载策略 '{strategy_id}' 的数据")
            else:
                logger.warning(f"无法加载策略 '{strategy_id}' 的数据: {result['message']}")
        except Exception as e:
            logger.error(f"加载策略 '{strategy_id}' 时出错: {str(e)}")
    
    return strategies_data

def evaluate_strategies(strategies_data, metrics=None, use_quantum=False):
    """评估多个策略"""
    from quantum_core.strategy_evaluator import StrategyEvaluator
    
    logger.info(f"开始评估 {len(strategies_data)} 个策略")
    
    # 创建评估器并启动
    evaluator = StrategyEvaluator(use_quantum_enhancement=use_quantum)
    evaluator.start()
    
    # 评估每个策略
    evaluation_results = {}
    
    for strategy_id, data in strategies_data.items():
        try:
            if 'equity_curve' not in data:
                logger.warning(f"策略 '{strategy_id}' 缺少权益曲线数据，跳过评估")
                continue
                
            # 转换权益曲线为Series对象
            if isinstance(data['equity_curve'], list):
                equity_curve = pd.Series(data['equity_curve'])
            elif isinstance(data['equity_curve'], dict):
                equity_curve = pd.Series(data['equity_curve']['values'], 
                                       index=data['equity_curve'].get('dates'))
            else:
                equity_curve = data['equity_curve']
                
            # 提取交易记录
            trades = data.get('trades', [])
            
            # 执行评估
            result = evaluator.evaluate_strategy(
                strategy_id=strategy_id,
                equity_curve=equity_curve,
                trades=trades,
                metrics=metrics,
                use_quantum=use_quantum
            )
            
            evaluation_results[strategy_id] = result
            logger.info(f"完成策略 '{strategy_id}' 的评估")
        except Exception as e:
            logger.error(f"评估策略 '{strategy_id}' 时出错: {str(e)}")
    
    # 如果需要，执行多维分析
    dimension_result = None
    if len(evaluation_results) >= 2:
        try:
            dimension_result = evaluator.perform_dimension_analysis(list(evaluation_results.keys()))
            logger.info("完成多维分析")
        except Exception as e:
            logger.error(f"执行多维分析时出错: {str(e)}")
    
    # 停止评估器
    evaluator.stop()
    
    return {
        'evaluations': evaluation_results,
        'dimension_analysis': dimension_result
    }

def generate_comparison_plots(strategies_data, evaluation_results):
    """生成对比图表"""
    logger.info("生成对比图表")
    
    plot_data = {}
    
    # 1. 权益曲线对比
    plt.figure(figsize=(12, 6))
    
    for strategy_id, data in strategies_data.items():
        if 'equity_curve' in data:
            equity_curve = data['equity_curve']
            if isinstance(equity_curve, dict):
                plt.plot(equity_curve.get('dates', range(len(equity_curve['values']))), 
                       equity_curve['values'], label=strategy_id)
            elif isinstance(equity_curve, list):
                plt.plot(range(len(equity_curve)), equity_curve, label=strategy_id)
            else:
                plt.plot(equity_curve.index, equity_curve.values, label=strategy_id)
    
    plt.title("策略权益曲线对比")
    plt.xlabel("日期")
    plt.ylabel("权益")
    plt.legend()
    plt.grid(True)
    
    plot_path = "strategy_equity_comparison.png"
    plt.savefig(plot_path)
    plot_data['equity_curve'] = plot_path
    
    # 2. 性能指标对比 - 雷达图
    if evaluation_results and len(evaluation_results['evaluations']) > 0:
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'sortino_ratio']
        available_metrics = set()
        
        # 找出所有可用的指标
        for strategy_id, eval_result in evaluation_results['evaluations'].items():
            if 'metrics' in eval_result:
                available_metrics.update(eval_result['metrics'].keys())
        
        # 使用可用的指标
        metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
        
        if metrics_to_plot:
            # 准备数据
            N = len(metrics_to_plot)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # 闭合多边形
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            for strategy_id, eval_result in evaluation_results['evaluations'].items():
                values = []
                
                for metric in metrics_to_plot:
                    if metric in eval_result['metrics']:
                        metric_value = eval_result['metrics'][metric]
                        # 处理新的字典格式
                        if isinstance(metric_value, dict) and 'value' in metric_value:
                            value = metric_value['value']
                        else:
                            value = metric_value
                            
                        # 对于max_drawdown，我们需要反转值（因为越小越好）
                        if metric == 'max_drawdown':
                            value = 1.0 - abs(value)
                        
                        values.append(value)
                    else:
                        values.append(0)
                
                # 归一化
                max_vals = []
                for i, metric in enumerate(metrics_to_plot):
                    max_val = max([abs(eval_result['metrics'].get(metric, {'value': 0}).get('value', 0) 
                               if isinstance(eval_result['metrics'].get(metric, 0), dict) 
                               else eval_result['metrics'].get(metric, 0)) 
                               for eval_result in evaluation_results['evaluations'].values()])
                    if max_val == 0:
                        max_val = 1
                    max_vals.append(max_val)
                
                norm_values = [v / max_vals[i] if max_vals[i] != 0 else 0 
                             for i, v in enumerate(values)]
                
                # 闭合多边形
                values += values[:1]
                norm_values += norm_values[:1]
                
                # 绘制
                ax.plot(angles, norm_values, linewidth=2, label=strategy_id)
                ax.fill(angles, norm_values, alpha=0.1)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot)
            
            plt.title("策略性能雷达图")
            plt.legend(loc='upper right')
            
            radar_plot_path = "strategy_radar_comparison.png"
            plt.savefig(radar_plot_path)
            plot_data['radar'] = radar_plot_path
    
    # 3. 如果有多维分析结果，绘制PCA图
    if (evaluation_results and 'dimension_analysis' in evaluation_results and 
        evaluation_results['dimension_analysis'] and 
        'pca' in evaluation_results['dimension_analysis']):
        
        pca_data = evaluation_results['dimension_analysis']['pca']
        if pca_data and 'components' in pca_data:
            components = pca_data['components']
            
            if len(components) > 0 and len(components[0]) >= 2:
                plt.figure(figsize=(10, 8))
                
                # 获取策略ID
                strategies = evaluation_results['dimension_analysis'].get('strategies', [])
                
                # 绘制散点图
                for i, strategy_id in enumerate(strategies):
                    if i < len(components):
                        plt.scatter(components[i][0], components[i][1], s=100, label=strategy_id)
                
                plt.title("策略PCA分析")
                plt.xlabel("主成分1")
                plt.ylabel("主成分2")
                plt.grid(True)
                plt.legend()
                
                pca_plot_path = "strategy_pca_comparison.png"
                plt.savefig(pca_plot_path)
                plot_data['pca'] = pca_plot_path
    
    return plot_data

def generate_html_report(strategies_data, evaluation_results, plot_data, output_file):
    """生成HTML报告"""
    logger.info(f"生成HTML报告: {output_file}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>策略对比分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .card {{ border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>策略对比分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="card">
                <h2>策略概述</h2>
                <table>
                    <tr>
                        <th>策略ID</th>
                        <th>数据点数</th>
                        <th>起始日期</th>
                        <th>结束日期</th>
                        <th>交易次数</th>
                    </tr>
    """
    
    # 添加策略概述表格
    for strategy_id, data in strategies_data.items():
        equity_curve = data.get('equity_curve', [])
        trades = data.get('trades', [])
        
        # 获取起止日期
        start_date = "N/A"
        end_date = "N/A"
        
        if isinstance(equity_curve, dict) and 'dates' in equity_curve:
            dates = equity_curve['dates']
            if dates:
                start_date = dates[0]
                end_date = dates[-1]
        elif isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            start_date = equity_curve.index[0]
            end_date = equity_curve.index[-1]
        
        # 数据点数
        if isinstance(equity_curve, dict) and 'values' in equity_curve:
            data_points = len(equity_curve['values'])
        elif isinstance(equity_curve, list):
            data_points = len(equity_curve)
        elif isinstance(equity_curve, pd.Series):
            data_points = len(equity_curve)
        else:
            data_points = "N/A"
        
        html_content += f"""
                    <tr>
                        <td>{strategy_id}</td>
                        <td>{data_points}</td>
                        <td>{start_date}</td>
                        <td>{end_date}</td>
                        <td>{len(trades)}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
    """
    
    # 添加权益曲线图
    if 'equity_curve' in plot_data:
        html_content += f"""
            <div class="card">
                <h2>权益曲线对比</h2>
                <div class="plot">
                    <img src="{plot_data['equity_curve']}" alt="权益曲线对比">
                </div>
            </div>
        """
    
    # 添加性能指标对比表格
    if evaluation_results and 'evaluations' in evaluation_results:
        html_content += """
            <div class="card">
                <h2>性能指标对比</h2>
                <table>
                    <tr>
                        <th>指标</th>
        """
        
        # 添加策略ID作为表头
        for strategy_id in evaluation_results['evaluations'].keys():
            html_content += f"<th>{strategy_id}</th>"
        
        html_content += """
                    </tr>
        """
        
        # 收集所有可能的指标
        all_metrics = set()
        for eval_result in evaluation_results['evaluations'].values():
            if 'metrics' in eval_result:
                all_metrics.update(eval_result['metrics'].keys())
        
        # 添加指标行
        for metric in sorted(all_metrics):
            html_content += f"""
                    <tr>
                        <td>{metric}</td>
            """
            
            for strategy_id, eval_result in evaluation_results['evaluations'].items():
                if 'metrics' in eval_result and metric in eval_result['metrics']:
                    metric_value = eval_result['metrics'][metric]
                    
                    # 处理新的字典格式
                    if isinstance(metric_value, dict) and 'value' in metric_value:
                        value = metric_value['value']
                        is_quantum = metric_value.get('quantum_enhanced', False)
                        
                        # 格式化值
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:.4f}"
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
                        elif isinstance(value, list):
                            formatted_value = f"[{len(value)} values]"
                        else:
                            formatted_value = str(value)
                    
                    html_content += f"<td>{formatted_value}</td>"
                else:
                    html_content += "<td>N/A</td>"
            
            html_content += """
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    # 添加雷达图
    if 'radar' in plot_data:
        html_content += f"""
            <div class="card">
                <h2>策略性能雷达图</h2>
                <div class="plot">
                    <img src="{plot_data['radar']}" alt="策略性能雷达图">
                </div>
            </div>
        """
    
    # 添加PCA分析图
    if 'pca' in plot_data:
        html_content += f"""
            <div class="card">
                <h2>策略PCA分析</h2>
                <div class="plot">
                    <img src="{plot_data['pca']}" alt="策略PCA分析">
                </div>
            </div>
        """
    
    # 添加维度重要性
    if (evaluation_results and 'dimension_analysis' in evaluation_results and 
        evaluation_results['dimension_analysis'] and 
        'pca' in evaluation_results['dimension_analysis']):
        
        pca_data = evaluation_results['dimension_analysis']['pca']
        if pca_data and 'feature_importances' in pca_data and 'feature_names' in pca_data:
            importances = pca_data['feature_importances']
            feature_names = pca_data['feature_names']
            
            if importances and feature_names:
                # 按重要性排序
                importance_tuples = sorted(zip(feature_names, importances), 
                                         key=lambda x: x[1], reverse=True)
                
                html_content += """
                <div class="card">
                    <h2>指标维度重要性</h2>
                    <table>
                        <tr>
                            <th>排名</th>
                            <th>指标</th>
                            <th>重要性分数</th>
                        </tr>
                """
                
                for i, (feature, importance) in enumerate(importance_tuples):
                    html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{feature}</td>
                            <td>{importance:.4f}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
    
    # 添加策略相似度矩阵
    if evaluation_results and 'evaluations' in evaluation_results:
        strategy_ids = list(evaluation_results['evaluations'].keys())
        
        if len(strategy_ids) > 1:
            html_content += """
                <div class="card">
                    <h2>策略相似度矩阵</h2>
                    <table>
                        <tr>
                            <th>策略ID</th>
            """
            
            # 添加策略ID作为表头
            for strategy_id in strategy_ids:
                html_content += f"<th>{strategy_id}</th>"
            
            html_content += """
                        </tr>
            """
            
            # 创建评估器
            from quantum_core.strategy_evaluator import StrategyEvaluator
            evaluator = StrategyEvaluator()
            evaluator.start()
            evaluator.results_cache = evaluation_results['evaluations']
            
            # 如果有聚类结果，更新策略聚类信息
            if ('dimension_analysis' in evaluation_results and 
                evaluation_results['dimension_analysis'] and 
                'clusters' in evaluation_results['dimension_analysis']):
                
                clusters = evaluation_results['dimension_analysis']['clusters']
                if clusters and 'labels' in clusters:
                    labels = clusters['labels']
                    for i, strategy_id in enumerate(evaluation_results['dimension_analysis'].get('strategies', [])):
                        if i < len(labels):
                            evaluator.strategy_clusters[strategy_id] = labels[i]
            
            # 计算每对策略之间的相似度
            for i, strategy_id1 in enumerate(strategy_ids):
                html_content += f"""
                        <tr>
                            <td>{strategy_id1}</td>
                """
                
                for j, strategy_id2 in enumerate(strategy_ids):
                    if i == j:
                        # 对角线元素是1.0
                        html_content += "<td>1.0000</td>"
                    else:
                        # 计算相似度
                        similarity = evaluator.get_strategy_similarity(strategy_id1, strategy_id2)
                        
                        if similarity['status'] == 'success':
                            cosine_similarity = similarity.get('cosine_similarity', 0)
                            html_content += f"<td>{cosine_similarity:.4f}</td>"
                        else:
                            html_content += "<td>N/A</td>"
                
                html_content += """
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
            
            evaluator.stop()
    
    # 添加页脚
    html_content += """
            <div class="footer">
                <p>此报告由超神量子核心策略对比分析工具生成</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 写入文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"报告已保存至 {output_file}")
        return True
    except Exception as e:
        logger.error(f"保存报告失败: {str(e)}")
        return False

def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析策略ID
    strategy_ids = [s.strip() for s in args.strategies.split(',')]
    
    # 解析指标
    metrics = None
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
    
    # 加载策略数据
    strategies_data = load_strategies(strategy_ids)
    
    if not strategies_data:
        logger.error("未能加载任何策略数据，终止分析")
        return 1
    
    # 评估策略
    evaluation_results = evaluate_strategies(strategies_data, metrics, args.quantum)
    
    # 生成图表
    plot_data = {}
    if args.plot:
        plot_data = generate_comparison_plots(strategies_data, evaluation_results)
    
    # 生成HTML报告
    generate_html_report(strategies_data, evaluation_results, plot_data, args.output)
    
    logger.info("分析完成")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"未处理的异常: {str(e)}", exc_info=True)
        sys.exit(1) 