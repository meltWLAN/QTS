#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面超神分析启动器
执行完整的超神分析流程并在GUI中展示结果
"""

import os
import sys
import time
import json
import logging
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SuperGodDesktopAnalysis")

def setup_arg_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="超神量子共生网络交易系统")
    parser.add_argument("--token", type=str, help="TuShare API令牌（可选，系统已内置默认token）")
    parser.add_argument("--stocks", type=str, help="分析的股票代码列表，逗号分隔")
    parser.add_argument("--count", type=int, default=50, help="随机选择分析的股票数量")
    parser.add_argument("--days", type=int, default=10, help="预测的天数")
    parser.add_argument("--output-dir", type=str, default="results", help="输出结果目录")
    return parser

# 添加从run_enhanced_system.py导入失败时的备用函数
def setup_supergod_modules(config, token=None):
    """初始化超神模块
    
    Args:
        config: 配置参数
        token: TuShare API令牌（可选，内部已使用固定token）
        
    Returns:
        tuple: (predictor, cosmic_engine, consciousness)
    """
    try:
        from quantum_symbiotic_network.quantum_prediction import get_predictor
        from quantum_symbiotic_network.cosmic_resonance import get_engine
        from quantum_symbiotic_network.quantum_consciousness import get_consciousness
        
        # 使用内置token
        fixed_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        logger.info("⚡ 初始化量子预测模块...")
        predictor = get_predictor(tushare_token=fixed_token)
        logger.info("✅ 量子预测模块初始化成功")
        
        logger.info("⚡ 初始化宇宙共振引擎...")
        cosmic_engine = get_engine(config)
        logger.info("✅ 宇宙共振引擎初始化成功")
        
        logger.info("⚡ 初始化量子意识...")
        consciousness = get_consciousness(config)
        logger.info("✅ 量子意识初始化成功")
        
        # 启动引擎
        cosmic_engine.start()
        consciousness.start()
        
        # 设置量子参数
        predictor.set_quantum_params(
            coherence=0.92,
            superposition=0.88,
            entanglement=0.85
        )
        
        return predictor, cosmic_engine, consciousness
    except Exception as e:
        logger.error(f"初始化超神模块失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_stocks(predictor, cosmic_engine, consciousness_engine, stock_list=None, stock_count=50, prediction_days=10):
    """分析股票并生成预测和洞察
    
    Args:
        predictor: 量子预测器
        cosmic_engine: 宇宙共振引擎
        consciousness_engine: 量子意识引擎
        stock_list: 指定的股票列表，如果为None则随机选择
        stock_count: 随机选择的股票数量
        prediction_days: 预测的天数
        
    Returns:
        tuple: (market_insights, stock_predictions)
    """
    try:
        # 获取市场指数状态
        logger.info("获取市场指数数据...")
        market_indexes = predictor.get_market_indexes()
        
        # 分析市场共振
        logger.info("分析市场共振...")
        resonance = cosmic_engine.analyze_market_resonance(market_indexes)
        
        # 分析市场意识
        logger.info("分析市场意识状态...")
        consciousness_state = consciousness_engine.analyze_market_consciousness(market_indexes)
        
        # 生成市场洞察
        logger.info("生成市场洞察...")
        market_insights = predictor.generate_market_insights(market_indexes)
        
        # 接收宇宙指引
        logger.info("接收宇宙指引...")
        cosmic_guidance = consciousness_engine.receive_cosmic_guidance()
        
        # 整合市场洞察
        insights = {
            'timestamp': datetime.now().isoformat(),
            'market_resonance': resonance,
            'consciousness': consciousness_state,
            'market_insights': market_insights,
            'cosmic_guidance': cosmic_guidance
        }
        
        # 如果没有指定股票列表，则获取并随机选择
        if not stock_list:
            try:
                from quantum_symbiotic_network.data_sources.enhanced_data_source import EnhancedDataSource
                data_source = EnhancedDataSource(token=predictor.tushare_token)
                all_stocks = data_source.get_stock_list()
                if all_stocks:
                    from random import sample
                    sample_size = min(stock_count, len(all_stocks))
                    stock_list = [stock['ts_code'] for stock in sample(all_stocks, sample_size)]
                else:
                    logger.warning("无法获取股票列表，使用默认股票")
                    stock_list = ['000001.SZ', '600000.SH', '600519.SH', '000651.SZ', '000333.SZ']
            except Exception as e:
                logger.error(f"获取股票列表失败: {str(e)}")
                stock_list = ['000001.SZ', '600000.SH', '600519.SH', '000651.SZ', '000333.SZ']
        
        # 分析每只股票
        logger.info(f"开始分析 {len(stock_list)} 只股票...")
        stock_predictions = {}
        
        for i, stock_code in enumerate(stock_list):
            logger.info(f"[{i+1}/{len(stock_list)}] 分析股票 {stock_code}...")
            
            try:
                # 使用超神预测
                prediction = predictor.predict(stock_code, days=prediction_days, use_tushare=True)
                
                if prediction:
                    stock_predictions[stock_code] = prediction
                    
                    # 输出关键预测
                    if 'predictions' in prediction and len(prediction['predictions']) > 0:
                        start_price = prediction['predictions'][0]
                        end_price = prediction['predictions'][-1]
                        change_pct = (end_price - start_price) / start_price * 100
                        
                        logger.info(f"股票 {stock_code} 预测: {start_price:.2f} -> {end_price:.2f} ({change_pct:.2f}%), 置信度: {prediction.get('confidence', 0):.1f}%")
                
                # 短暂暂停，避免API限制
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"分析股票 {stock_code} 失败: {str(e)}")
        
        logger.info(f"股票分析完成，共 {len(stock_predictions)} 只股票")
        return insights, stock_predictions
        
    except Exception as e:
        logger.error(f"分析股票时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {}

def save_results(insights, predictions, insights_file, predictions_file):
    """保存分析结果
    
    Args:
        insights: 市场洞察
        predictions: 股票预测
        insights_file: 洞察文件路径
        predictions_file: 预测文件路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(insights_file), exist_ok=True)
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        
        # 保存市场洞察
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)
        
        # 保存股票预测
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存:")
        logger.info(f"- 市场洞察: {insights_file}")
        logger.info(f"- 股票预测: {predictions_file}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")

def print_results(insights, predictions):
    """打印分析结果摘要
    
    Args:
        insights: 市场洞察
        predictions: 股票预测
    """
    try:
        print("\n" + "=" * 80)
        print("                   超神量子分析结果摘要                   ")
        print("=" * 80)
        
        # 打印市场洞察摘要
        market_resonance = insights.get('market_resonance', {})
        consciousness = insights.get('consciousness', {})
        market_insights = insights.get('market_insights', {})
        cosmic_guidance = insights.get('cosmic_guidance', {})
        
        print(f"\n📊 市场共振状态:")
        print(f"  - 共振评分: {market_resonance.get('resonance_score', 0)}")
        print(f"  - 共振级别: {market_resonance.get('resonance_level', '未知')}")
        
        print(f"\n🧠 量子意识状态:")
        print(f"  - 意识清晰度: {consciousness.get('clarity', 0)}")
        print(f"  - 清晰度级别: {consciousness.get('clarity_level', '未知')}")
        
        print(f"\n💡 市场洞察:")
        if isinstance(market_insights, dict):
            for key, value in market_insights.items():
                if isinstance(value, dict):
                    print(f"  - {key}:")
                    for k, v in value.items():
                        print(f"    + {k}: {v}")
                else:
                    print(f"  - {key}: {value}")
        
        print(f"\n🌌 宇宙指引:")
        if isinstance(cosmic_guidance, dict):
            for key, value in cosmic_guidance.items():
                print(f"  - {key}: {value}")
        elif isinstance(cosmic_guidance, str):
            print(f"  {cosmic_guidance}")
        
        # 打印股票预测摘要
        print("\n📈 股票预测摘要:")
        
        # 按预期收益率排序
        sorted_predictions = []
        for code, pred in predictions.items():
            if 'predictions' in pred and len(pred['predictions']) > 0:
                start_price = pred['predictions'][0]
                end_price = pred['predictions'][-1]
                change_pct = (end_price - start_price) / start_price * 100
                sorted_predictions.append((code, change_pct, pred))
        
        # 按预期收益率排序
        sorted_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 打印前10个最佳预测
        top_count = min(10, len(sorted_predictions))
        if top_count > 0:
            print(f"\n🔝 收益率最高的 {top_count} 只股票:")
            for i, (code, change_pct, pred) in enumerate(sorted_predictions[:top_count]):
                print(f"  {i+1}. {code}: {change_pct:.2f}% (置信度: {pred.get('confidence', 0):.1f}%)")
        
        # 打印后5个最差预测
        bottom_count = min(5, len(sorted_predictions))
        if bottom_count > 0:
            print(f"\n⚠️ 需注意的 {bottom_count} 只股票:")
            for i, (code, change_pct, pred) in enumerate(sorted_predictions[-bottom_count:]):
                print(f"  {i+1}. {code}: {change_pct:.2f}% (置信度: {pred.get('confidence', 0):.1f}%)")
        
        print("\n" + "=" * 80)
        print("提示: 完整结果已保存到JSON文件，将在GUI中显示更详细的分析")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"打印结果时出错: {str(e)}")

def run_supergod_analysis(token=None, stocks=None, count=50, days=10, output_dir="results"):
    """运行超神分析"""
    try:
        # 使用固定token
        fixed_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        # 尝试从run_enhanced_system导入函数
        setup_modules = None
        try:
            from run_enhanced_system import initialize_supergod_modules
            setup_modules = initialize_supergod_modules
            logger.info("成功从run_enhanced_system导入initialize_supergod_modules")
        except ImportError:
            setup_modules = setup_supergod_modules
            logger.info("使用内部实现的setup_supergod_modules")
        
        logger.info("🌟 启动超神量子分析流程 🌟")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化超神模块
        logger.info("⚡ 初始化超神量子模块...")
        quantum_predictor, cosmic_engine, consciousness_engine = setup_modules({}, fixed_token)
        
        if not quantum_predictor or not cosmic_engine or not consciousness_engine:
            logger.error("初始化超神模块失败，无法继续分析")
            return None
        
        # 解析股票列表
        stock_list = []
        if stocks:
            stock_list = [s.strip() for s in stocks.split(",")]
            logger.info(f"将分析指定的 {len(stock_list)} 只股票")
        
        # 获取市场洞察和股票预测
        logger.info("🔮 执行超神量子分析...")
        insights, predictions = analyze_stocks(
            quantum_predictor, 
            cosmic_engine, 
            consciousness_engine,
            stock_list=stock_list,
            stock_count=count,
            prediction_days=days
        )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        insights_file = os.path.join(output_dir, f"supergod_insights_{timestamp}.json")
        predictions_file = os.path.join(output_dir, f"supergod_predictions_{timestamp}.json")
        
        save_results(insights, predictions, insights_file, predictions_file)
        logger.info(f"✅ 分析结果已保存到 {insights_file} 和 {predictions_file}")
        
        # 打印结果摘要
        print_results(insights, predictions)
        
        return {
            "insights_file": insights_file,
            "predictions_file": predictions_file,
            "insights": insights,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"运行超神分析时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def launch_gui_with_results(analysis_results):
    """加载超神分析结果并启动GUI"""
    logger.info("🖥️ 启动超神桌面系统并加载分析结果...")
    
    # 在环境变量中传递结果文件路径
    os.environ["SUPERGOD_INSIGHTS_FILE"] = analysis_results["insights_file"]
    os.environ["SUPERGOD_PREDICTIONS_FILE"] = analysis_results["predictions_file"]
    os.environ["SUPERGOD_ANALYSIS_COMPLETE"] = "1"
    
    try:
        # 直接启动GUI
        import launch_supergod
        # 启动带有超神模式的GUI
        launch_supergod.launch_super_desktop(True)
    except Exception as e:
        logger.error(f"启动GUI时出错: {str(e)}")
        # 回退到命令行方式启动
        os.system("python launch_supergod.py --supergod-mode")

def main():
    """主函数"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 显示启动信息
    print("\n" + "=" * 80)
    print("                超神量子共生网络交易系统 - 桌面超神分析")
    print("=" * 80 + "\n")
    
    # 使用固定的token，不再需要命令行参数
    fixed_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    try:
        # 运行超神分析
        analysis_results = run_supergod_analysis(
            token=fixed_token,
            stocks=args.stocks,
            count=args.count,
            days=args.days,
            output_dir=args.output_dir
        )
        
        if not analysis_results:
            logger.error("分析结果为空，无法继续")
            return 1
        
        # 启动GUI并加载结果
        launch_gui_with_results(analysis_results)
        
        return 0
    except KeyboardInterrupt:
        print("\n用户中断，退出程序")
        return 1
    except Exception as e:
        logger.error(f"运行超神分析时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 