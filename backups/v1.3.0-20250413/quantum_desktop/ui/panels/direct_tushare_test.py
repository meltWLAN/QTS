"""
直接测试Tushare API连接
"""
import tushare as ts
import pandas as pd
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TushareTest')

def test_tushare_connection(token):
    """测试Tushare API连接"""
    logger.info(f"开始测试Tushare API连接，Token: {token[:10]}...")
    
    try:
        # 设置token
        ts.set_token(token)
        pro = ts.pro_api()
        logger.info("成功创建Tushare pro API对象")
        
        # 测试连接 - 获取一只股票基本信息
        logger.info("测试获取股票信息...")
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market', limit=5)
        
        if df is not None and not df.empty:
            logger.info(f"连接成功！获取到 {len(df)} 条股票信息：")
            print(df)
            return True, df
        else:
            logger.error("连接成功但未获取到数据")
            return False, None
            
    except Exception as e:
        logger.error(f"连接失败：{str(e)}")
        return False, str(e)

def test_daily_data(pro, trade_date=None):
    """测试获取日线数据"""
    try:
        # 获取交易日期
        if trade_date is None:
            today = datetime.now()
            # 尝试最近10个交易日
            for i in range(1, 11):
                test_date = (today - timedelta(days=i)).strftime('%Y%m%d')
                logger.info(f"尝试交易日期: {test_date}")
                
                # 获取上证指数数据测试日期是否有效
                df_test = pro.index_daily(ts_code='000001.SH', trade_date=test_date)
                if df_test is not None and not df_test.empty:
                    trade_date = test_date
                    logger.info(f"找到有效交易日: {trade_date}")
                    break
            
            if trade_date is None:
                logger.error("未找到有效交易日")
                return False, "未找到有效交易日"
        
        # 获取几只主要股票的日线数据
        stock_list = ['600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000333.SZ']
        ts_codes = ','.join(stock_list)
        
        logger.info(f"获取日线数据，交易日: {trade_date}，股票: {ts_codes}")
        df_daily = pro.daily(ts_code=ts_codes, trade_date=trade_date)
        
        if df_daily is not None and not df_daily.empty:
            logger.info(f"成功获取日线数据: {len(df_daily)} 条记录")
            print(df_daily)
            return True, df_daily
        else:
            logger.error("未获取到日线数据")
            return False, None
            
    except Exception as e:
        logger.error(f"获取日线数据失败: {str(e)}")
        return False, str(e)

def test_stock_basic(pro, limit=10):
    """测试获取股票基本信息"""
    try:
        logger.info(f"获取股票基本信息，限制 {limit} 条")
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date', limit=limit)
        
        if df is not None and not df.empty:
            logger.info(f"成功获取基本信息: {len(df)} 条记录")
            print(df)
            return True, df
        else:
            logger.error("未获取到股票基本信息")
            return False, None
            
    except Exception as e:
        logger.error(f"获取股票基本信息失败: {str(e)}")
        return False, str(e)

def test_daily_basic(pro, trade_date, ts_codes):
    """测试获取每日指标"""
    try:
        logger.info(f"获取每日指标，交易日: {trade_date}，股票: {ts_codes}")
        df = pro.daily_basic(ts_code=ts_codes, trade_date=trade_date)
        
        if df is not None and not df.empty:
            logger.info(f"成功获取每日指标: {len(df)} 条记录")
            print(df)
            return True, df
        else:
            logger.error("未获取到每日指标")
            return False, None
            
    except Exception as e:
        logger.error(f"获取每日指标失败: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    # 输入Tushare API Token
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    # 测试基本连接
    success, result = test_tushare_connection(token)
    
    if success:
        # 创建API对象
        pro = ts.pro_api()
        
        # 测试获取交易日期和日线数据
        success_daily, result_daily = test_daily_data(pro)
        
        # 测试获取股票基本信息
        success_basic, result_basic = test_stock_basic(pro)
        
        # 如果成功获取了日线数据，继续测试每日指标
        if success_daily:
            trade_date = result_daily.iloc[0]['trade_date']
            ts_codes = ','.join(result_daily['ts_code'].unique())
            test_daily_basic(pro, trade_date, ts_codes) 