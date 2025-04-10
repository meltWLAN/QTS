#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试TushareDataHandler加载指数数据的过程
"""

import logging
from datetime import datetime
import pandas as pd
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'SuperQuantumNetwork'))

# 从项目导入数据处理模块
try:
    from SuperQuantumNetwork.data import TushareDataHandler
except ImportError:
    try:
        from data import TushareDataHandler
    except ImportError:
        print("无法导入TushareDataHandler，请确保在正确的目录中运行脚本")
        sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    # 定义要测试的指数列表
    indices = [
        '000001.SH',  # 上证指数
        '399001.SZ',  # 深证成指
        '000300.SH',  # 沪深300
    ]
    
    # 使用与测试脚本相同的时间范围和token
    start_date = '20230101'
    end_date = '20230105'
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    
    logger.info("== 测试1: 直接使用Tushare API获取指数数据 ==")
    
    import tushare as ts
    # 设置token和初始化API
    ts.set_token(token)
    pro = ts.pro_api()
    
    for index_code in indices:
        try:
            logger.info(f"尝试直接使用Tushare API获取指数 {index_code} 数据...")
            df = pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                logger.info(f"成功获取指数 {index_code} 数据，共 {len(df)} 条记录")
                logger.info(f"数据样例: \n{df.head()}")
            else:
                logger.warning(f"获取指数 {index_code} 数据失败，返回为空")
        except Exception as e:
            logger.error(f"获取指数 {index_code} 数据时发生错误: {str(e)}")
    
    logger.info("\n== 测试2: 使用TushareDataHandler加载指数数据 ==")
    
    try:
        # 创建TushareDataHandler实例并记录详细过程
        logger.info("初始化TushareDataHandler...")
        data_handler = TushareDataHandler(
            symbols=indices,
            start_date=start_date,
            end_date=end_date,
            token=token
        )
        
        logger.info(f"数据加载完成，成功加载的标的列表: {data_handler.symbols}")
        
        # 检查是否所有指数都成功加载
        for index_code in indices:
            if index_code in data_handler.symbols:
                hist_data = data_handler.get_historical_data(index_code, 10)
                logger.info(f"指数 {index_code} 加载成功，数据条数: {len(hist_data)}")
                if hist_data and len(hist_data) > 0:
                    logger.info(f"最新数据: {hist_data[-1]}")
                else:
                    logger.warning(f"指数 {index_code} 数据为空")
            else:
                logger.warning(f"指数 {index_code} 数据加载失败，未在成功加载列表中找到")
        
    except Exception as e:
        logger.error(f"TushareDataHandler测试过程中发生错误: {str(e)}", exc_info=True)
    
    logger.info("\n== 测试3: 测试数据处理器与策略集成 ==")
    
    try:
        try:
            from SuperQuantumNetwork.quantum_burst_strategy_enhanced import QuantumBurstStrategyEnhanced
        except ImportError:
            try:
                from quantum_burst_strategy_enhanced import QuantumBurstStrategyEnhanced
            except ImportError:
                logger.error("无法导入QuantumBurstStrategyEnhanced，跳过测试3")
                return
        
        # 创建包含股票和指数的测试数据集
        test_symbols = indices + ['000001.SZ', '600000.SH']  # 添加两只股票
        
        logger.info(f"初始化数据处理器，包含 {len(test_symbols)} 个标的...")
        data_handler = TushareDataHandler(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            token=token
        )
        
        logger.info("初始化量子爆发策略...")
        strategy = QuantumBurstStrategyEnhanced()
        strategy.data_handler = data_handler
        strategy.market_index_code = '000001.SH'  # 设置市场指数代码
        
        # 模拟策略的update_market_state方法调用
        logger.info("调用strategy.update_market_state()方法...")
        # 获取最后一个交易日作为当前日期
        current_date = sorted(data_handler.dates)[-1] if data_handler.dates else datetime.now()
        
        # 创建一个简单的事件对象来传递给update_market_state
        from event import MarketDataEvent, EventType
        event = MarketDataEvent(
            type=EventType.MARKET_DATA,
            timestamp=current_date,
            data={}  # 空数据，因为策略会从data_handler中获取数据
        )
        
        logger.info(f"当前日期: {current_date}, 可用股票数量: {len(data_handler.symbols)}")
        logger.info(f"调用策略update_market_state方法，收集详细日志...")
        
        # 设置更详细的日志级别
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        logging.getLogger("SuperQuantumNetwork").setLevel(logging.DEBUG)
        
        # 调用strategy的update_market_state方法
        strategy.update_market_state(event)
        
        # 检查市场指数数据是否获取成功
        if hasattr(strategy, 'market_index_data') and strategy.market_index_data is not None:
            logger.info(f"成功获取市场指数数据，使用的指数代码: {strategy.market_index_code}")
            logger.info(f"市场指数数据点数: {len(strategy.market_index_data)}")
            if len(strategy.market_index_data) > 0:
                logger.info(f"第一条数据: {strategy.market_index_data.iloc[0].to_dict() if isinstance(strategy.market_index_data, pd.DataFrame) else strategy.market_index_data[0]}")
        else:
            logger.warning("未能获取市场指数数据，可能使用了替代的综合指数")
        
    except Exception as e:
        logger.error(f"策略集成测试过程中发生错误: {str(e)}", exc_info=True)
    
if __name__ == "__main__":
    main() 