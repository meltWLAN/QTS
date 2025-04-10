#!/usr/bin/env python3
"""
超神量子回测系统 - Tushare数据源调用示例
使用已有的Tushare数据源和新的日志系统进行回测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入日志系统
from src.utils.logger import setup_backtest_logger

# 导入已有的Tushare数据源
# 根据系统中的位置可能需要调整导入路径
try:
    from SuperQuantumNetwork.quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
    print("成功导入量子共生网络中的TushareDataSource")
except ImportError:
    try:
        from SuperQuantumNetwork.tushare_data_connector import TushareDataConnector as TushareDataSource
        print("成功导入TushareDataConnector作为数据源")
    except ImportError:
        # 如果两者都导入失败，创建一个简单的Mock类用于演示
        print("无法导入已有的Tushare数据源，将使用模拟数据")
        
        class TushareDataSource:
            """模拟的Tushare数据源类，用于演示目的"""
            
            def __init__(self, token=None):
                self.token = token
                self.logger = logging.getLogger("MockTushare")
                self.logger.info(f"模拟TushareDataSource初始化，使用Token: {token[:8]}...")
                
            def get_daily_data(self, code, start_date=None, end_date=None):
                """获取日线数据"""
                # 生成模拟数据
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
                if end_date is None:
                    end_date = datetime.now().strftime('%Y%m%d')
                
                # 生成日期范围
                start = datetime.strptime(start_date, '%Y%m%d')
                end = datetime.strptime(end_date, '%Y%m%d')
                date_range = [start + timedelta(days=x) for x in range((end-start).days + 1)]
                date_range = [d for d in date_range if d.weekday() < 5]  # 只保留工作日
                
                data = []
                price = 100.0  # 起始价格
                
                for date in date_range:
                    # 生成随机价格变动
                    change = np.random.normal(0, 1) / 100
                    price = price * (1 + change)
                    
                    # 生成当天的OHLCV数据
                    open_price = price * (1 + np.random.normal(0, 0.005))
                    high_price = price * (1 + abs(np.random.normal(0, 0.01)))
                    low_price = price * (1 - abs(np.random.normal(0, 0.01)))
                    close_price = price
                    volume = np.random.randint(1000, 10000)
                    
                    data.append({
                        'date': date.strftime('%Y%m%d'),
                        'code': code,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'amount': volume * close_price
                    })
                
                return pd.DataFrame(data)
            
            def get_stock_list(self):
                """获取股票列表"""
                stocks = [
                    {'ts_code': '000001.SZ', 'name': '平安银行', 'industry': '银行'},
                    {'ts_code': '000002.SZ', 'name': '万科A', 'industry': '房地产'},
                    {'ts_code': '000063.SZ', 'name': '中兴通讯', 'industry': '通信设备'},
                    {'ts_code': '000568.SZ', 'name': '泸州老窖', 'industry': '白酒'},
                    {'ts_code': '000651.SZ', 'name': '格力电器', 'industry': '家电'},
                    {'ts_code': '000858.SZ', 'name': '五粮液', 'industry': '白酒'},
                    {'ts_code': '002415.SZ', 'name': '海康威视', 'industry': '电子'},
                    {'ts_code': '600030.SH', 'name': '中信证券', 'industry': '证券'},
                    {'ts_code': '600036.SH', 'name': '招商银行', 'industry': '银行'},
                    {'ts_code': '600276.SH', 'name': '恒瑞医药', 'industry': '医药'},
                    {'ts_code': '600519.SH', 'name': '贵州茅台', 'industry': '白酒'},
                    {'ts_code': '601318.SH', 'name': '中国平安', 'industry': '保险'},
                    {'ts_code': '601398.SH', 'name': '工商银行', 'industry': '银行'}
                ]
                return pd.DataFrame(stocks)


def run_simple_backtest():
    """运行一个简单的回测示例"""
    # 设置回测专用日志记录器
    logger = setup_backtest_logger(strategy_name="简单均线策略")
    
    # 记录回测的基本信息
    logger.info("=============================================")
    logger.info("超神量子回测系统 - Tushare数据调用示例")
    logger.info("简单均线策略")
    
    # 使用已有的Token（在多处代码中找到的Token）
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    # 回测参数
    start_date = "20220101"  # 开始日期
    end_date = "20221231"    # 结束日期
    initial_capital = 1000000.0  # 初始资金
    
    # 主要回测标的
    stocks = [
        "600519.SH",  # 贵州茅台
        "000858.SZ",  # 五粮液
        "600036.SH",  # 招商银行
        "601318.SH",  # 中国平安
        "000651.SZ",  # 格力电器
    ]
    
    logger.info(f"回测区间: {start_date} - {end_date}")
    logger.info(f"回测标的: {len(stocks)} 只股票")
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info("=============================================")
    
    # 初始化数据源
    try:
        logger.info(f"初始化Tushare数据源，使用Token: {token[:8]}...")
        data_source = TushareDataSource(token=token)
        
        # 获取股票列表（测试数据源是否正常工作）
        try:
            stock_list = data_source.get_stock_list()
            logger.info(f"成功获取股票列表，共 {len(stock_list)} 只股票")
        except Exception as e:
            logger.error(f"获取股票列表失败: {str(e)}")
    except Exception as e:
        logger.error(f"初始化Tushare数据源失败: {str(e)}")
        logger.info("将使用模拟数据进行演示")
        data_source = TushareDataSource(token=token)
    
    # 回测循环
    portfolio = {stock: 0 for stock in stocks}  # 持仓
    cash = initial_capital  # 现金
    trades = []  # 交易记录
    
    # 遍历每个股票
    for stock in stocks:
        logger.info(f"正在获取 {stock} 的历史数据...")
        
        try:
            # 获取股票数据
            df = data_source.get_daily_data(stock, start_date=start_date, end_date=end_date)
            
            if df is None or len(df) == 0:
                logger.warning(f"获取 {stock} 数据失败或数据为空，跳过该股票")
                continue
                
            logger.info(f"成功获取 {stock} 数据，共 {len(df)} 条记录")
            
            # 计算技术指标 - 简单的移动平均线策略
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # 生成交易信号
            df['signal'] = 0
            df.loc[df['MA5'] > df['MA20'], 'signal'] = 1  # 买入信号
            df.loc[df['MA5'] < df['MA20'], 'signal'] = -1  # 卖出信号
            
            # 移除NaN值
            df = df.dropna()
            
            # 回测
            position = 0
            for i in range(1, len(df)):
                prev_signal = df.iloc[i-1]['signal']
                curr_signal = df.iloc[i]['signal']
                price = df.iloc[i]['close']
                date = df.iloc[i]['date']
                
                # 信号变化时交易
                if prev_signal != curr_signal:
                    # 买入信号
                    if curr_signal == 1 and position == 0:
                        # 计算买入股数（简化为全仓买入）
                        shares = int(cash * 0.2 / price)  # 使用20%资金买入
                        
                        if shares > 0:
                            cost = shares * price
                            cash -= cost
                            portfolio[stock] += shares
                            position = 1
                            
                            trades.append({
                                'date': date,
                                'stock': stock,
                                'action': 'BUY',
                                'price': price,
                                'shares': shares,
                                'cost': cost,
                                'cash': cash
                            })
                            
                            logger.info(f"买入 {stock}: 日期={date}, 价格={price:.2f}, 数量={shares}, 花费={cost:.2f}, 剩余现金={cash:.2f}")
                    
                    # 卖出信号
                    elif curr_signal == -1 and position == 1:
                        shares = portfolio[stock]
                        
                        if shares > 0:
                            revenue = shares * price
                            cash += revenue
                            portfolio[stock] = 0
                            position = 0
                            
                            trades.append({
                                'date': date,
                                'stock': stock,
                                'action': 'SELL',
                                'price': price,
                                'shares': shares,
                                'revenue': revenue,
                                'cash': cash
                            })
                            
                            logger.info(f"卖出 {stock}: 日期={date}, 价格={price:.2f}, 数量={shares}, 收入={revenue:.2f}, 现金={cash:.2f}")
        
        except Exception as e:
            logger.error(f"处理 {stock} 时发生错误: {str(e)}")
    
    # 计算最终持仓价值
    portfolio_value = cash
    for stock, shares in portfolio.items():
        if shares > 0:
            try:
                # 获取最后一个交易日的收盘价
                df = data_source.get_daily_data(stock, start_date=end_date, end_date=end_date)
                if df is not None and len(df) > 0:
                    last_price = df.iloc[-1]['close']
                    stock_value = shares * last_price
                    portfolio_value += stock_value
                    logger.info(f"最终持仓 {stock}: {shares} 股, 价值 {stock_value:.2f}")
            except Exception as e:
                logger.error(f"计算 {stock} 最终价值时出错: {str(e)}")
    
    # 计算回测结果
    profit = portfolio_value - initial_capital
    profit_pct = profit / initial_capital * 100
    
    # 记录回测结果
    logger.info("=============================================")
    logger.info("回测完成，结果如下:")
    logger.info(f"初始资金: {initial_capital:,.2f}")
    logger.info(f"最终资金: {portfolio_value:,.2f}")
    logger.info(f"总收益: {profit:,.2f}")
    logger.info(f"收益率: {profit_pct:.2f}%")
    logger.info(f"交易次数: {len(trades)}")
    logger.info("=============================================")
    
    # 返回回测结果
    return {
        'initial_capital': initial_capital,
        'final_capital': portfolio_value,
        'profit': profit,
        'profit_pct': profit_pct,
        'trades': trades
    }


if __name__ == "__main__":
    # 运行回测
    results = run_simple_backtest()
    
    # 输出总结
    print(f"\n回测完成！总收益率: {results['profit_pct']:.2f}%")
    print(f"查看日志文件获取详细信息。") 