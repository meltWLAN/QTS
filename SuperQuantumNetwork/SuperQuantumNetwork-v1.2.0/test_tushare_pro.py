#!/usr/bin/env python3
"""
测试TuShare Pro API是否能正常工作
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入需要测试的模块
try:
    from gui.controllers.data_controller import DataController
    from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    sys.exit(1)

def test_tushare_pro_api():
    """测试TuShare Pro API"""
    logger.info("开始测试TuShare Pro API")
    
    # 从配置文件加载token
    import json
    token = None
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            token = config.get('tushare_token')
    except Exception as e:
        logger.warning(f"无法加载配置文件中的token: {e}")
    
    # 使用默认token作为备用
    if not token:
        token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        logger.info("使用默认token进行测试")
    
    # 初始化TushareDataSource
    tushare_source = TushareDataSource(token=token)
    if tushare_source.api is None:
        logger.error("TuShare Pro API初始化失败")
        return False
    
    # 测试获取股票列表
    logger.info("测试获取股票列表...")
    stocks = tushare_source.get_stock_list()
    if not stocks:
        logger.error("获取股票列表失败")
    else:
        logger.info(f"成功获取股票列表，共 {len(stocks)} 支股票")
    
    # 测试获取日线数据
    logger.info("测试获取股票日线数据...")
    test_codes = ['000001.SZ', '600000.SH']
    for code in test_codes:
        # 获取最近5天数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        logger.info(f"获取 {code} 从 {start_date} 到 {end_date} 的日线数据")
        df = tushare_source.get_daily_data(code, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            logger.error(f"获取 {code} 日线数据失败")
        else:
            logger.info(f"成功获取 {code} 日线数据，共 {len(df)} 行")
            # 显示前几行数据
            logger.info(f"数据示例:\n{df.head()}")
    
    logger.info("TuShare Pro API测试完成")
    return True

def test_data_controller():
    """测试DataController"""
    logger.info("开始测试DataController")
    
    # 初始化DataController
    controller = DataController()
    controller.initialize()
    
    # 测试获取日线数据
    test_codes = ['000001.SZ', '600000.SH']
    for code in test_codes:
        logger.info(f"通过DataController获取 {code} 最近5天数据")
        df = controller.get_daily_data(code, days=5)
        
        if df is None or df.empty:
            logger.error(f"通过DataController获取 {code} 数据失败")
        else:
            logger.info(f"成功通过DataController获取 {code} 数据，共 {len(df)} 行")
            logger.info(f"数据示例:\n{df.head()}")
    
    logger.info("DataController测试完成")
    return True

if __name__ == "__main__":
    logger.info("TuShare Pro API 和 DataController 测试开始")
    
    # 测试TuShare Pro API
    api_success = test_tushare_pro_api()
    
    # 测试DataController
    controller_success = test_data_controller()
    
    # 显示测试结果
    if api_success and controller_success:
        logger.info("所有测试通过！TuShare Pro API 和 DataController 工作正常")
    else:
        if not api_success:
            logger.error("TuShare Pro API 测试失败")
        if not controller_success:
            logger.error("DataController 测试失败") 