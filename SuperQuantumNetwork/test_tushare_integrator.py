#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子系统 - Tushare数据接口集成测试
"""

import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import tushare as ts
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tushare_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TushareIntegrator:
    """Tushare数据接口集成器"""
    
    def __init__(self, token=None):
        """初始化Tushare集成器
        
        Args:
            token: Tushare API令牌
        """
        self.token = token or "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        self.init_success = False
        self.pro = None
        self.data_cache = {}
        self._initialize()
    
    def _initialize(self):
        """初始化Tushare API"""
        try:
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            self.init_success = True
            logger.info("Tushare API初始化成功")
        except Exception as e:
            logger.error(f"Tushare API初始化失败: {str(e)}")
    
    def test_connection(self):
        """测试Tushare连接"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法测试连接")
            return False
        
        try:
            # 尝试获取交易日历进行测试
            df = self.pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20230110')
            if df is not None and not df.empty:
                logger.info(f"Tushare连接测试成功，获取到 {len(df)} 条交易日历数据")
                return True
            else:
                logger.warning("Tushare连接测试失败，未获取到数据")
                return False
        except Exception as e:
            logger.error(f"Tushare连接测试失败: {str(e)}")
            return False
    
    def get_stock_data(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        """获取股票日线数据"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取股票数据")
            return None
        
        try:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = self.pro.daily(**params)
            if df is not None and not df.empty:
                logger.info(f"获取到 {len(df)} 条股票日线数据")
                return df
            else:
                logger.warning("未获取到股票日线数据")
                return None
        except Exception as e:
            logger.error(f"获取股票日线数据失败: {str(e)}")
            return None
    
    def get_index_data(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        """获取指数日线数据"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取指数数据")
            return None
        
        try:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = self.pro.index_daily(**params)
            if df is not None and not df.empty:
                logger.info(f"获取到 {len(df)} 条指数日线数据")
                return df
            else:
                logger.warning("未获取到指数日线数据")
                return None
        except Exception as e:
            logger.error(f"获取指数日线数据失败: {str(e)}")
            return None
    
    def get_stock_basic(self):
        """获取股票基本信息"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取股票基本信息")
            return None
        
        try:
            df = self.pro.stock_basic(exchange='', list_status='L')
            if df is not None and not df.empty:
                logger.info(f"获取到 {len(df)} 只股票的基本信息")
                return df
            else:
                logger.warning("未获取到股票基本信息")
                return None
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {str(e)}")
            return None
    
    def get_daily_basic(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        """获取每日指标数据"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取每日指标数据")
            return None
        
        try:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            df = self.pro.daily_basic(**params)
            if df is not None and not df.empty:
                logger.info(f"获取到 {len(df)} 条每日指标数据")
                return df
            else:
                logger.warning("未获取到每日指标数据")
                return None
        except Exception as e:
            logger.error(f"获取每日指标数据失败: {str(e)}")
            return None
    
    def get_financial_data(self, ts_code, report_type='1', start_date=None, end_date=None):
        """获取财务数据（利润表、资产负债表、现金流量表）"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取财务数据")
            return None
        
        try:
            # 利润表
            income_df = self.pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date, report_type=report_type)
            if income_df is not None and not income_df.empty:
                logger.info(f"获取到 {ts_code} 的 {len(income_df)} 条利润表数据")
            else:
                logger.warning(f"未获取到 {ts_code} 的利润表数据")
            
            # 资产负债表
            balance_df = self.pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date, report_type=report_type)
            if balance_df is not None and not balance_df.empty:
                logger.info(f"获取到 {ts_code} 的 {len(balance_df)} 条资产负债表数据")
            else:
                logger.warning(f"未获取到 {ts_code} 的资产负债表数据")
            
            # 现金流量表
            cashflow_df = self.pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date, report_type=report_type)
            if cashflow_df is not None and not cashflow_df.empty:
                logger.info(f"获取到 {ts_code} 的 {len(cashflow_df)} 条现金流量表数据")
            else:
                logger.warning(f"未获取到 {ts_code} 的现金流量表数据")
            
            return {
                'income': income_df,
                'balance': balance_df,
                'cashflow': cashflow_df
            }
        except Exception as e:
            logger.error(f"获取财务数据失败: {str(e)}")
            return None
    
    def get_index_weight(self, index_code, trade_date=None):
        """获取指数成分和权重"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取指数成分和权重")
            return None
        
        try:
            if not trade_date:
                trade_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')  # 默认取最近一个月的数据
            
            df = self.pro.index_weight(index_code=index_code, trade_date=trade_date)
            if df is not None and not df.empty:
                logger.info(f"获取到 {index_code} 在 {trade_date} 的 {len(df)} 条成分和权重数据")
                return df
            else:
                logger.warning(f"未获取到 {index_code} 在 {trade_date} 的成分和权重数据")
                return None
        except Exception as e:
            logger.error(f"获取指数成分和权重失败: {str(e)}")
            return None
    
    def get_market_data(self, category='stock', date=None):
        """获取市场各类数据"""
        if not self.init_success:
            logger.error("Tushare未初始化，无法获取市场数据")
            return None
        
        if not date:
            date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')  # 默认取最近一周的数据
        
        result = {}
        
        try:
            # 根据不同类型获取不同数据
            if category == 'stock':
                # 股票行情
                df_daily = self.pro.daily(trade_date=date)
                if df_daily is not None and not df_daily.empty:
                    logger.info(f"获取到 {len(df_daily)} 条股票日线数据")
                    result['daily'] = df_daily
                
                # 每日指标
                df_basic = self.pro.daily_basic(trade_date=date)
                if df_basic is not None and not df_basic.empty:
                    logger.info(f"获取到 {len(df_basic)} 条每日指标数据")
                    result['daily_basic'] = df_basic
                
                # 涨跌停价格
                df_limit = self.pro.stk_limit(trade_date=date)
                if df_limit is not None and not df_limit.empty:
                    logger.info(f"获取到 {len(df_limit)} 条涨跌停价格数据")
                    result['stk_limit'] = df_limit
                
            elif category == 'index':
                # 指数行情
                df_index = self.pro.index_daily(trade_date=date)
                if df_index is not None and not df_index.empty:
                    logger.info(f"获取到 {len(df_index)} 条指数日线数据")
                    result['index_daily'] = df_index
                
                # 大盘指数每日指标
                df_index_basic = self.pro.index_dailybasic(trade_date=date)
                if df_index_basic is not None and not df_index_basic.empty:
                    logger.info(f"获取到 {len(df_index_basic)} 条大盘指数每日指标数据")
                    result['index_dailybasic'] = df_index_basic
                
            elif category == 'fund':
                # 公募基金净值
                df_fund = self.pro.fund_nav(end_date=date)
                if df_fund is not None and not df_fund.empty:
                    logger.info(f"获取到 {len(df_fund)} 条公募基金净值数据")
                    result['fund_nav'] = df_fund
                
                # 场内基金日线行情
                df_fund_daily = self.pro.fund_daily(trade_date=date)
                if df_fund_daily is not None and not df_fund_daily.empty:
                    logger.info(f"获取到 {len(df_fund_daily)} 条场内基金日线行情数据")
                    result['fund_daily'] = df_fund_daily
                
            elif category == 'future':
                # 期货日线行情
                df_future = self.pro.fut_daily(trade_date=date)
                if df_future is not None and not df_future.empty:
                    logger.info(f"获取到 {len(df_future)} 条期货日线行情数据")
                    result['fut_daily'] = df_future
                
            elif category == 'fx':
                # 外汇日线行情
                df_fx = self.pro.fx_daily(trade_date=date)
                if df_fx is not None and not df_fx.empty:
                    logger.info(f"获取到 {len(df_fx)} 条外汇日线行情数据")
                    result['fx_daily'] = df_fx
                
            return result
            
        except Exception as e:
            logger.error(f"获取市场 {category} 数据失败: {str(e)}")
            return None
    
    def save_data_to_csv(self, data, filename, directory='data'):
        """将数据保存为CSV文件"""
        try:
            # 创建目录
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # 保存数据
            file_path = os.path.join(directory, filename)
            
            if isinstance(data, dict):
                # 如果是字典，保存多个文件
                for key, df in data.items():
                    if df is not None and not df.empty:
                        key_file_path = os.path.join(directory, f"{key}_{filename}")
                        df.to_csv(key_file_path, index=False)
                        logger.info(f"数据已保存到 {key_file_path}")
            else:
                # 直接保存单个DataFrame
                if data is not None and not data.empty:
                    data.to_csv(file_path, index=False)
                    logger.info(f"数据已保存到 {file_path}")
                else:
                    logger.warning("数据为空，未保存文件")
        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")


def main():
    """主函数"""
    logger.info("=== 超神量子系统 - Tushare数据接口集成测试 ===")
    
    # 初始化Tushare集成器
    tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    integrator = TushareIntegrator(token=tushare_token)
    
    # 测试连接
    if not integrator.test_connection():
        logger.error("Tushare连接测试失败，程序退出")
        return 1
    
    logger.info("开始测试各类数据接口...")
    
    # 测试股票基本信息
    stock_basic = integrator.get_stock_basic()
    if stock_basic is not None:
        integrator.save_data_to_csv(stock_basic, "stock_basic.csv")
    
    # 测试指数数据
    index_codes = ['000001.SH', '399001.SZ', '000300.SH']
    for index_code in index_codes:
        index_data = integrator.get_index_data(ts_code=index_code, start_date='20230101', end_date='20230331')
        if index_data is not None:
            integrator.save_data_to_csv(index_data, f"{index_code}_daily.csv")
    
    # 测试成分股数据
    sample_stocks = ['600519.SH', '000858.SZ', '601318.SH']
    for stock in sample_stocks:
        stock_data = integrator.get_stock_data(ts_code=stock, start_date='20230101', end_date='20230331')
        if stock_data is not None:
            integrator.save_data_to_csv(stock_data, f"{stock}_daily.csv")
    
    # 测试财务数据
    financial_data = integrator.get_financial_data('600519.SH', start_date='20220101', end_date='20221231')
    if financial_data is not None:
        integrator.save_data_to_csv(financial_data, "600519_financial.csv")
    
    # 测试指数成分和权重
    index_weight = integrator.get_index_weight('000300.SH')
    if index_weight is not None:
        integrator.save_data_to_csv(index_weight, "000300_weight.csv")
    
    # 测试市场数据
    categories = ['stock', 'index', 'fund', 'future', 'fx']
    for category in categories:
        logger.info(f"测试 {category} 类数据...")
        market_data = integrator.get_market_data(category=category)
        if market_data is not None:
            integrator.save_data_to_csv(market_data, f"{category}_market.csv")
        time.sleep(1)  # 防止请求过于频繁
    
    logger.info("=== Tushare数据接口集成测试完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 