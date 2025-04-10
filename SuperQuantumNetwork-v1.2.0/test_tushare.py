#!/usr/bin/env python3
"""
测试Tushare数据获取功能
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TushareTest")

def test_tushare_connection():
    """测试Tushare连接"""
    try:
        import tushare as ts
        logger.info(f"成功导入Tushare，版本: {ts.__version__}")
        
        # 使用公共接口获取数据（不需要token）
        logger.info("测试公共接口...")
        df_public = ts.get_hs300s()
        if df_public is not None and not df_public.empty:
            logger.info(f"公共接口测试成功，获取到 {len(df_public)} 条数据")
            print(df_public.head(3))
        else:
            logger.error("公共接口测试失败，未能获取数据")
        
        # 测试Pro接口
        logger.info("测试Pro接口...")
        token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"  # 使用默认token
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 测试获取交易日历
        logger.info("获取交易日历...")
        df_cal = pro.trade_cal(exchange='SSE', start_date='20250101', end_date='20250401')
        if df_cal is not None and not df_cal.empty:
            logger.info(f"交易日历获取成功，共 {len(df_cal)} 条数据")
            print(df_cal.head(3))
        else:
            logger.error("交易日历获取失败")
        
        # 测试获取指数行情
        logger.info("获取指数行情...")
        df_index = pro.index_daily(ts_code='000001.SH', start_date='20250301', end_date='20250401')
        if df_index is not None and not df_index.empty:
            logger.info(f"指数行情获取成功，共 {len(df_index)} 条数据")
            print(df_index.head(3))
        else:
            logger.error("指数行情获取失败")
            
        # 测试使用备用方法获取数据
        logger.info("测试备用方法...")
        df_stock = ts.get_hist_data('000001', start='2024-01-01', end='2024-04-01')
        if df_stock is not None and not df_stock.empty:
            logger.info(f"备用方法测试成功，获取到 {len(df_stock)} 条数据")
            print(df_stock.head(3))
        else:
            logger.error("备用方法测试失败")
            
        return True
    except Exception as e:
        logger.error(f"Tushare测试出错: {str(e)}")
        return False

def test_local_data_generation():
    """测试本地数据生成功能"""
    logger.info("生成本地模拟数据...")
    
    try:
        # 创建模拟的市场指数数据
        dates = pd.date_range(end=datetime.now(), periods=30)
        
        # 沪深300指数模拟数据
        np.random.seed(42)  # 设置随机种子以确保可重复性
        
        # 基础价格和变化
        base_price = 3500
        daily_changes = np.random.normal(0.0002, 0.015, len(dates))
        
        # 添加轻微趋势
        trend = np.linspace(0, 0.03, len(dates))
        daily_changes = daily_changes + trend
        
        # 计算价格序列
        prices = base_price * np.cumprod(1 + daily_changes)
        
        # 创建成交量
        volumes = np.random.normal(2000000, 500000, len(dates))
        volumes = np.abs(volumes)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - np.random.uniform(0, 0.005, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'close': prices,
            'volume': volumes,
            'code': '000300.SH'
        })
        
        logger.info(f"成功生成模拟数据，共 {len(df)} 条记录")
        print(df.head(3))
        
        # 保存到本地CSV文件
        df.to_csv('data/local_market_data.csv', index=False)
        logger.info("模拟数据已保存到 data/local_market_data.csv")
        
        return True
    except Exception as e:
        logger.error(f"生成本地数据出错: {str(e)}")
        return False

def test_akshare():
    """测试AKShare数据源"""
    try:
        import akshare as ak
        logger.info(f"成功导入AKShare，版本: {ak.__version__}")
        
        # 测试获取A股指数数据
        logger.info("获取上证指数数据...")
        df_sh = ak.stock_zh_index_daily(symbol="sh000001")
        if df_sh is not None and not df_sh.empty:
            logger.info(f"上证指数数据获取成功，共 {len(df_sh)} 条数据")
            print(df_sh.head(3))
        else:
            logger.error("上证指数数据获取失败")
        
        # 测试获取A股个股数据
        logger.info("获取平安银行数据...")
        df_stock = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", end_date="20240401")
        if df_stock is not None and not df_stock.empty:
            logger.info(f"个股数据获取成功，共 {len(df_stock)} 条数据")
            print(df_stock.head(3))
        else:
            logger.error("个股数据获取失败")
            
        return True
    except ImportError:
        logger.error("未安装AKShare库，请使用 pip install akshare 安装")
        return False
    except Exception as e:
        logger.error(f"AKShare测试出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("超神量子共生系统 - 数据源测试")
    print("=" * 50)
    
    # 确保数据目录存在
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
        logger.info("创建数据目录: data")
    
    # 测试Tushare
    print("\n[1] 测试Tushare数据源")
    print("-" * 30)
    tushare_result = test_tushare_connection()
    
    # 测试本地数据生成
    print("\n[2] 测试本地数据生成")
    print("-" * 30)
    local_result = test_local_data_generation()
    
    # 测试AKShare
    print("\n[3] 测试AKShare数据源")
    print("-" * 30)
    akshare_result = test_akshare()
    
    # 输出总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"Tushare: {'✓ 成功' if tushare_result else '✗ 失败'}")
    print(f"本地数据: {'✓ 成功' if local_result else '✗ 失败'}")
    print(f"AKShare: {'✓ 成功' if akshare_result else '✗ 失败'}")
    print("=" * 50)
    
    # 建议最佳数据源
    print("\n推荐使用的数据源:")
    if akshare_result:
        print("- AKShare (推荐): 稳定可靠，无需令牌")
    elif tushare_result:
        print("- Tushare: 需要有效的API令牌")
    else:
        print("- 本地模拟数据: 当外部数据源都不可用时的备选方案")
    
    print("\n完成测试。")

if __name__ == "__main__":
    main() 