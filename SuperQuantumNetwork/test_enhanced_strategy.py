#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子爆发增强策略回测脚本 - 极度优化参数版
"""

import logging
import sys
import os
from datetime import datetime

# 确保可以导入量子爆发策略模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SuperQuantumNetwork.quantum_burst_strategy_enhanced import QuantumBurstStrategyEnhanced
from core.backtest_engine import BacktestEngine
from data_source.data_source import DataSource
from utils.config import SYMBOLS

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 回测用股票列表
SYMBOLS = [
    # 金融
    '601398.SH', '600036.SH', '601318.SH', '600030.SH', '601688.SH',
    # 消费
    '600519.SH', '603288.SH', '600276.SH', '600887.SH', '600690.SH',
    # 科技
    '600703.SH', '002415.SZ', '002230.SZ', '600745.SH', '601012.SH',
    # 医药
    '300750.SZ', '002594.SZ',
    # ETF和指数
    '600905.SH', 
    # 能源
    '601985.SH', '601857.SH', '600028.SH', '601899.SH',
    # 材料
    '600019.SH', '600188.SH',
    # 房地产
    '600048.SH',
    # 其他行业
    '001979.SZ', '601111.SH', '600029.SH', '600585.SH', '601888.SH',
    # 增加更多中小盘股票以提高多样性
    '002475.SZ', '300014.SZ', '300124.SZ', '002241.SZ', '300274.SZ',
    '002714.SZ', '300015.SZ', '002572.SZ', '300059.SZ', '002773.SZ',
    '300760.SZ', '002410.SZ', '300122.SZ', '002640.SZ', '300413.SZ'
]

def main():
    # 回测参数
    start_date = "2021-01-01"  # 开始日期
    end_date = "2021-06-30"    # 结束日期
    initial_capital = 1000000.0  # 初始资金

    # 创建数据源
    data_source = DataSource()

    # 创建策略实例，进一步调整参数使策略更积极
    strategy = QuantumBurstStrategyEnhanced(
        symbols=SYMBOLS,
        lookback_period=20,         # 缩短回溯期，使策略对短期变化更敏感
        max_positions=10,           # 增加最大持仓数，更分散风险
        price_change_threshold=0.02,  # 大幅降低价格变化阈值
        volume_change_threshold=0.8,  # 降低成交量变化阈值
        signal_threshold=0.3,        # 极大降低信号阈值，更容易触发交易
        stop_loss=0.05,              # 设置较小的止损以降低单笔损失
        take_profit=0.10,            # 保持合理的止盈水平
        position_sizing=0.1,         # 每个头寸使用10%的资金
        # 调整量子权重，强化短期动量和波动因素
        quantum_weights={
            'trend_strength': 0.15,  # 降低趋势强度权重
            'momentum': 0.25,        # 增加动量权重
            'volatility': 0.20,      # 增加波动率权重
            'volume_structure': 0.20, # 增加成交量结构权重
            'reversal_pattern': 0.10, # 保持反转模式权重
            'support_resistance': 0.05, # 降低支撑阻力权重
            'quantum_oscillator': 0.05  # 降低量子振荡器权重
        }
    )

    # 创建回测引擎
    backtest = BacktestEngine(
        strategy=strategy,
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )

    # 运行回测
    results = backtest.run()

    # 记录回测结果
    logger.info("回测完成，结果如下:")
    logger.info(f"最终权益: {results['metrics'].get('final_equity', 'N/A')}")
    logger.info(f"总收益率: {results['metrics'].get('total_return', 'N/A'):.2f}%")
    logger.info(f"年化收益: {results['metrics'].get('annual_return', 'N/A'):.2f}%")
    logger.info(f"最大回撤: {results['metrics'].get('max_drawdown', 'N/A'):.2f}%")
    
    # 获取交易次数
    trades_count = 0
    if 'trades' in results:
        trades_count = len(results['trades'])
    elif 'trades_count' in results.get('metrics', {}):
        trades_count = results['metrics']['trades_count']
    
    logger.info(f"交易次数: {trades_count}")
    
    if trades_count > 0:
        logger.info(f"胜率: {results['metrics'].get('win_rate', 'N/A'):.2f}%")
        logger.info(f"盈亏比: {results['metrics'].get('profit_loss_ratio', 'N/A'):.2f}")
        logger.info(f"夏普比率: {results['metrics'].get('sharpe_ratio', 'N/A'):.2f}")
        
        # 输出交易明细
        logger.info("交易明细:")
        if 'trades' in results:
            for trade in results['trades']:
                logger.info(f"股票: {trade.get('symbol')}, "
                           f"买入时间: {trade.get('entry_time')}, "
                           f"买入价格: {trade.get('entry_price'):.2f}, "
                           f"卖出时间: {trade.get('exit_time')}, "
                           f"卖出价格: {trade.get('exit_price'):.2f}, "
                           f"收益: {trade.get('profit'):.2f}, "
                           f"收益率: {trade.get('return'):.2f}%")
        else:
            logger.info("无详细交易记录")
    else:
        logger.info("无交易记录")


if __name__ == "__main__":
    main() 