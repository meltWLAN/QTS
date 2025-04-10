#!/usr/bin/env python3
"""
超神量子预测测试脚本 - 用于排查预测问题
"""

import os
import sys
import logging
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("TestQuantumPrediction")

# 确保quantum_symbiotic_network目录可导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prediction(stock_code):
    """测试股票预测功能
    
    Args:
        stock_code: 股票代码
    """
    try:
        logger.info(f"开始测试股票 {stock_code} 的预测功能")
        
        # 导入预测模块
        from quantum_symbiotic_network.quantum_prediction import get_predictor
        
        # 获取预测器
        logger.info("初始化量子预测器...")
        predictor = get_predictor()
        
        if predictor is None:
            logger.error("预测器初始化失败！")
            return
        
        # 输出预测器状态
        logger.info(f"预测器状态: coherence={predictor.coherence}, superposition={predictor.superposition}, entanglement={predictor.entanglement}")
        logger.info(f"TuShare API状态: {'已连接' if predictor.pro is not None else '未连接'}")
        
        # 测试获取市场指数
        logger.info("获取市场指数数据...")
        market_indexes = predictor.get_market_indexes()
        logger.info(f"获取到 {len(market_indexes)} 个市场指数")
        
        for code, index_data in market_indexes.items():
            logger.info(f"指数 {code} ({index_data['name']}): 最新价格={index_data['last_close']}, 变化率={index_data['change']}%")
        
        # 测试股票代码格式转换
        try:
            from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
            data_source = TushareDataSource()
            
            logger.info(f"测试股票代码 {stock_code} 格式转换...")
            
            # 获取股票列表
            stocks = data_source.get_stock_list()
            logger.info(f"获取到 {len(stocks)} 只股票")
            
            # 查找匹配的股票
            matched_stock = None
            for stock in stocks:
                if stock['code'] == stock_code or stock['ts_code'].startswith(stock_code):
                    matched_stock = stock
                    break
            
            if matched_stock:
                logger.info(f"找到匹配的股票: {matched_stock['ts_code']} - {matched_stock['name']}")
                stock_code_to_use = matched_stock['ts_code']
            else:
                logger.warning(f"未找到匹配的股票，尝试自动转换格式")
                if stock_code.startswith('6'):
                    stock_code_to_use = f"{stock_code}.SH"
                else:
                    stock_code_to_use = f"{stock_code}.SZ"
                logger.info(f"转换后的股票代码: {stock_code_to_use}")
        except Exception as e:
            logger.error(f"股票代码转换失败: {str(e)}")
            stock_code_to_use = stock_code
            logger.info(f"使用原始股票代码: {stock_code_to_use}")
        
        # 进行预测
        logger.info(f"开始预测股票 {stock_code_to_use} 未来10天走势...")
        start_time = time.time()
        prediction = predictor.predict(stock_code_to_use, days=10, use_tushare=True)
        elapsed_time = time.time() - start_time
        logger.info(f"预测完成，耗时 {elapsed_time:.2f} 秒")
        
        # 打印预测结果
        if prediction:
            logger.info("预测结果:")
            
            # 日期和价格预测
            if 'dates' in prediction and 'predictions' in prediction:
                for i, (date, price) in enumerate(zip(prediction['dates'], prediction['predictions'])):
                    logger.info(f"  {date}: {price}")
                
                # 计算涨跌幅
                start_price = prediction['predictions'][0]
                end_price = prediction['predictions'][-1]
                change_pct = (end_price - start_price) / start_price * 100
                logger.info(f"10天预期涨跌幅: {change_pct:.2f}%")
                
                # 置信度
                logger.info(f"预测置信度: {prediction.get('confidence', 0):.2f}%")
                
                # 市场洞察
                if 'market_insights' in prediction:
                    insights = prediction['market_insights']
                    logger.info("市场洞察:")
                    for key, value in insights.items():
                        if isinstance(value, dict):
                            logger.info(f"  {key}:")
                            for k, v in value.items():
                                logger.info(f"    {k}: {v}")
                        else:
                            logger.info(f"  {key}: {value}")
            else:
                logger.error("预测结果格式不正确，缺少日期或预测值")
        else:
            logger.error("预测失败，未返回有效结果")
        
        return prediction
    
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 从命令行获取股票代码
    import argparse
    parser = argparse.ArgumentParser(description="测试超神量子预测功能")
    parser.add_argument("stock_code", help="股票代码，如: 600000")
    args = parser.parse_args()
    
    # 运行测试
    test_prediction(args.stock_code) 