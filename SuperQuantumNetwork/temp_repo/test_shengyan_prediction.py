#!/usr/bin/env python3
"""
测试圣元环保(300867)的量子预测功能
"""

import sys
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ShengYuanPrediction")

def main():
    """测试圣元环保的量子预测功能"""
    try:
        logger.info("开始测试圣元环保(300867)的量子预测功能...")
        
        # 导入量子增强器
        from quantum_stock_enhancer import enhancer
        
        # 设置测试股票 - 圣元环保
        test_stock_code = "300867.SZ"  # 圣元环保
        test_stock_name = "圣元环保"
        test_price = 22.0  # 假设当前价格
        
        logger.info(f"正在为 {test_stock_name}({test_stock_code}) 生成未来10天价格预测...")
        
        # 调用预测函数
        prediction_result = enhancer.predict_stock_future_prices(
            stock_code=test_stock_code,
            stock_name=test_stock_name,
            current_price=test_price,
            days=10
        )
        
        # 验证预测结果
        if prediction_result and prediction_result.get('success', False):
            logger.info("预测成功！")
            
            # 打印量子分析参数
            quantum_analysis = prediction_result.get('quantum_analysis', {})
            print("\n量子分析参数：")
            print(f"量子纠缠强度: {quantum_analysis.get('entanglement', 0)*100:.1f}%")
            print(f"量子相干性: {quantum_analysis.get('coherence', 0)*100:.1f}%")
            print(f"量子共振度: {quantum_analysis.get('resonance', 0)*100:.1f}%")
            print(f"量子维度: {quantum_analysis.get('dimensions', 0)}维")
            
            # 打印预测结果
            predictions = prediction_result.get('predictions', [])
            print("\n价格预测：")
            print(f"{'日期':<12} {'预测价格':>10} {'涨跌幅':>10} {'置信度':>10}")
            print("-" * 45)
            
            for pred in predictions:
                date = pred.get('date', '')
                price = pred.get('price', 0)
                change = pred.get('change', 0)
                confidence = pred.get('confidence', 0) * 100
                print(f"{date:<12} {price:>10.2f} {change:>+10.2f}% {confidence:>10.1f}%")
            
            # 打印趋势分析
            trends = prediction_result.get('trends', {})
            print("\n趋势分析：")
            print(f"方向: {trends.get('direction', '未知')}")
            print(f"强度: {trends.get('strength', 0)*100:.1f}%")
            print(f"置信度: {trends.get('confidence', 0)*100:.1f}%")
            
            # 打印关键点位
            critical_points = prediction_result.get('critical_points', [])
            if critical_points:
                print("\n关键点位：")
                for point in critical_points:
                    date = point.get('date', '')
                    desc = point.get('description', '')
                    importance = point.get('importance', 0) * 100
                    print(f"{date}: {desc} (重要性: {importance:.1f}%)")
            
            # 将结果保存到文件
            with open('shengyan_prediction_result.json', 'w', encoding='utf-8') as f:
                json.dump(prediction_result, f, ensure_ascii=False, indent=2)
            logger.info("预测结果已保存到 shengyan_prediction_result.json")
        else:
            logger.error("预测失败！")
            if prediction_result:
                print(f"错误信息: {prediction_result.get('message', '未知错误')}")
            
    except ImportError:
        logger.error("无法导入量子增强器，请确保 quantum_stock_enhancer.py 已正确安装")
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 