#!/usr/bin/env python3
"""
超神量子预测修复版 - 简单预测界面
修复问题：股票预测和市场洞察没有显示
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
import logging
import json
from datetime import datetime, timedelta
import time

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示
except:
    pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("SuperPredictionFix")

# 确保quantum_symbiotic_network目录可导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def predict_stock(stock_code, days=10):
    """预测股票走势
    
    Args:
        stock_code: 股票代码
        days: 预测天数
        
    Returns:
        dict: 预测结果
    """
    try:
        logger.info(f"开始预测股票 {stock_code} 未来 {days} 天走势")
        
        # 导入预测模块
        from quantum_symbiotic_network.quantum_prediction import get_predictor
        
        # 获取预测器并设置最佳参数
        predictor = get_predictor()
        predictor.set_quantum_params(
            coherence=0.95,
            superposition=0.92,
            entanglement=0.90
        )
        
        # 测试股票代码格式转换
        try:
            from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
            data_source = TushareDataSource()
            
            # 获取股票列表
            stocks = data_source.get_stock_list()
            
            # 查找匹配的股票
            matched_stock = None
            for stock in stocks:
                if stock['code'] == stock_code or stock['ts_code'].startswith(stock_code):
                    matched_stock = stock
                    break
            
            if matched_stock:
                logger.info(f"找到匹配的股票: {matched_stock['ts_code']} - {matched_stock['name']}")
                stock_code_to_use = matched_stock['ts_code']
                stock_name = matched_stock['name']
            else:
                logger.warning(f"未找到匹配的股票，尝试自动转换格式")
                if stock_code.startswith('6'):
                    stock_code_to_use = f"{stock_code}.SH"
                else:
                    stock_code_to_use = f"{stock_code}.SZ"
                stock_name = f"股票{stock_code}"
                logger.info(f"转换后的股票代码: {stock_code_to_use}")
        except Exception as e:
            logger.error(f"股票代码转换失败: {str(e)}")
            stock_code_to_use = stock_code
            stock_name = f"股票{stock_code}"
        
        # 获取历史数据
        try:
            stock_data = data_source.get_stock_data(stock_code_to_use)
            if 'history' in stock_data and 'prices' in stock_data['history']:
                historical_prices = stock_data['history']['prices']
                historical_dates = stock_data['history']['dates']
                logger.info(f"成功获取 {len(historical_prices)} 天的历史数据")
            else:
                historical_prices = []
                historical_dates = []
        except:
            historical_prices = []
            historical_dates = []
        
        # 执行预测
        start_time = time.time()
        prediction = predictor.predict(stock_code_to_use, days=days, use_tushare=True)
        elapsed_time = time.time() - start_time
        logger.info(f"预测完成，耗时 {elapsed_time:.2f} 秒")
        
        # 添加股票名称
        prediction['stock_name'] = stock_name
        prediction['stock_code'] = stock_code_to_use
        
        # 添加历史数据
        prediction['historical_prices'] = historical_prices[-30:] if historical_prices else []
        prediction['historical_dates'] = historical_dates[-30:] if historical_dates else []
        
        return prediction
    
    except Exception as e:
        logger.error(f"预测过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_prediction(prediction, output_file=None):
    """绘制预测结果
    
    Args:
        prediction: 预测结果
        output_file: 输出文件路径，如果为None则显示图表
    """
    if not prediction:
        logger.error("没有预测结果可供绘制")
        return
    
    try:
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 设置标题
        stock_name = prediction.get('stock_name', '')
        stock_code = prediction.get('stock_code', '')
        title = f"超神量子预测: {stock_name}({stock_code}) - {datetime.now().strftime('%Y-%m-%d')}"
        plt.suptitle(title, fontsize=16)
        
        # 创建主图表
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        
        # 历史数据
        historical_prices = prediction.get('historical_prices', [])
        historical_dates = prediction.get('historical_dates', [])
        
        if historical_prices:
            # 转换日期格式
            x_hist = list(range(len(historical_prices)))
            ax1.plot(x_hist, historical_prices, color='gray', linewidth=1, label='历史价格')
            
            # 设置x轴刻度
            if historical_dates:
                num_ticks = min(5, len(historical_dates))
                step = len(historical_dates) // num_ticks
                tick_indices = list(range(0, len(historical_dates), step))
                ax1.set_xticks(tick_indices)
                ax1.set_xticklabels([historical_dates[i] for i in tick_indices], rotation=45)
        
        # 预测数据
        pred_prices = prediction.get('predictions', [])
        pred_dates = prediction.get('dates', [])
        
        if pred_prices:
            # 计算预测开始点
            start_x = len(historical_prices) - 1 if historical_prices else 0
            x_pred = list(range(start_x, start_x + len(pred_prices) + 1))
            
            # 连接最后一个历史点和预测
            y_pred = [historical_prices[-1] if historical_prices else pred_prices[0]] + pred_prices
            
            # 画预测线
            ax1.plot(x_pred, y_pred, color='red', linewidth=2, marker='o', label='预测价格')
            
            # 设置预测区域底色
            ax1.axvspan(start_x, start_x + len(pred_prices), alpha=0.1, color='red')
            
            # 标注预测区域
            ax1.text(start_x + len(pred_prices)/2, min(y_pred), "预测区间", 
                   bbox=dict(facecolor='red', alpha=0.2), horizontalalignment='center')
        
        # 设置网格
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title("价格预测")
        ax1.set_ylabel("价格")
        ax1.legend()
        
        # 涨跌幅文本框 - 右上角
        ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
        ax2.axis('off')
        
        if pred_prices:
            # 计算涨跌幅
            start_price = pred_prices[0]
            end_price = pred_prices[-1]
            change_pct = (end_price - start_price) / start_price * 100
            
            # 设置文本框内容
            text_content = f"预测涨跌幅: {change_pct:.2f}%\n"
            text_content += f"起始价: {start_price:.2f}\n"
            text_content += f"结束价: {end_price:.2f}\n"
            text_content += f"置信度: {prediction.get('confidence', 0):.1f}%\n"
            
            # 设置文本颜色
            text_color = "red" if change_pct > 0 else "green"
            
            # 添加文本框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props, color=text_color)
        
        # 市场洞察 - 右中部
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
        ax3.axis('off')
        
        if 'market_insights' in prediction:
            insights = prediction['market_insights']
            
            # 趋势信息
            if 'trend' in insights:
                trend = insights['trend']
                trend_type = trend.get('type', '未知')
                trend_direction = trend.get('direction', '未知')
                trend_strength = trend.get('strength', 0)
            else:
                trend_type = '未知'
                trend_direction = '未知'
                trend_strength = 0
            
            # 波动性信息
            if 'volatility' in insights:
                volatility = insights['volatility']
                vol_level = volatility.get('level', 0)
                vol_eval = volatility.get('evaluation', '未知')
            else:
                vol_level = 0
                vol_eval = '未知'
            
            # 买入/卖出时机
            entry_point = "未知"
            exit_point = "未知"
            if 'timing' in insights:
                timing = insights['timing']
                if 'entry' in timing:
                    entry = timing['entry']
                    entry_point = f"第{entry.get('day', 0)}天 (¥{entry.get('price', 0):.2f})"
                if 'exit' in timing:
                    exit = timing['exit']
                    exit_point = f"第{exit.get('day', 0)}天 (¥{exit.get('price', 0):.2f})"
            
            # 量子分析
            q_stability = 0
            q_states = 0
            if 'quantum_analysis' in insights:
                q_analysis = insights['quantum_analysis']
                q_stability = q_analysis.get('quantum_stability', 0)
                q_states = q_analysis.get('superposition_states', 0)
            
            # 组装市场洞察文本
            insights_text = f"市场洞察:\n"
            insights_text += f"趋势类型: {trend_type}\n"
            insights_text += f"趋势方向: {trend_direction}\n"
            insights_text += f"波动评估: {vol_eval}\n"
            insights_text += f"建议买入: {entry_point}\n"
            insights_text += f"建议卖出: {exit_point}\n"
            insights_text += f"量子稳定度: {q_stability:.2f}\n"
            
            # 添加文本框
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            ax3.text(0.05, 0.95, insights_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # 预测详情表格 - 下方
        ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3, rowspan=1)
        ax4.axis('off')
        
        if pred_prices and pred_dates:
            # 创建表格数据
            table_data = []
            for i, (date, price) in enumerate(zip(pred_dates, pred_prices)):
                # 计算当日涨跌幅
                prev_price = pred_prices[i-1] if i > 0 else (historical_prices[-1] if historical_prices else price)
                day_change = (price - prev_price) / prev_price * 100
                
                # 添加到表格
                table_data.append([f"第{i+1}天", date, f"{price:.2f}", f"{day_change:+.2f}%"])
            
            # 创建表格
            table = ax4.table(cellText=table_data, 
                             colLabels=["预测天数", "日期", "预测价格", "当日涨跌幅"],
                             loc='center', cellLoc='center')
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            
            # 设置单元格颜色
            for i in range(len(pred_prices)):
                # 根据当日涨跌设置颜色
                day_change = float(table_data[i][3].strip('%+'))
                if day_change > 0:
                    table[(i+1, 3)].set_facecolor('#ffcccc')
                elif day_change < 0:
                    table[(i+1, 3)].set_facecolor('#ccffcc')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存或显示
        if output_file:
            plt.savefig(output_file, dpi=150)
            logger.info(f"预测图表已保存到 {output_file}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"绘制预测图表时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def save_prediction_json(prediction, output_file):
    """保存预测结果为JSON文件
    
    Args:
        prediction: 预测结果
        output_file: 输出文件路径
    """
    if not prediction:
        logger.error("没有预测结果可保存")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2)
        logger.info(f"预测结果已保存到 {output_file}")
    except Exception as e:
        logger.error(f"保存预测结果时出错: {str(e)}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="超神量子预测修复版")
    parser.add_argument("stock_code", help="股票代码，如: 600000")
    parser.add_argument("--days", type=int, default=10, help="预测天数")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 执行预测
    prediction = predict_stock(args.stock_code, args.days)
    
    if prediction:
        # 保存为JSON
        json_file = os.path.join(args.output, f"prediction_{args.stock_code}_{timestamp}.json")
        save_prediction_json(prediction, json_file)
        
        # 绘制图表
        image_file = os.path.join(args.output, f"prediction_{args.stock_code}_{timestamp}.png")
        plot_prediction(prediction, image_file)
        
        # 显示图表
        plot_prediction(prediction)
        
        print(f"\n✨ 预测完成! ✨")
        print(f"- 预测结果已保存到: {json_file}")
        print(f"- 预测图表已保存到: {image_file}")
        
        return 0
    else:
        print("\n❌ 预测失败，请检查日志获取详细信息")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 