# 输出回测结果
logger.info("回测完成，结果如下:")
try:
    if 'final_equity' in results:
        logger.info(f"最终权益: {results['final_equity']:.2f}")
    else:
        logger.info("最终权益: 未生成交易，维持初始资金")
        
    if 'total_return' in results:
        logger.info(f"总收益率: {results['total_return']:.2%}")
    else:
        logger.info("总收益率: 0.00%")
        
    if 'annual_return' in results:
        logger.info(f"年化收益率: {results['annual_return']:.2%}")
    else:
        logger.info("年化收益率: 0.00%")
        
    if 'max_drawdown' in results:
        logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    else:
        logger.info("最大回撤: 0.00%")
        
    if 'win_rate' in results:
        logger.info(f"胜率: {results['win_rate']:.2%}")
    else:
        logger.info("胜率: 无交易记录")
        
    if 'sharpe_ratio' in results:
        logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
    else:
        logger.info("夏普比率: 无法计算")
except Exception as e:
    logger.error(f"回测过程中发生错误: {str(e)}")
    logger.error(traceback.format_exc()) 