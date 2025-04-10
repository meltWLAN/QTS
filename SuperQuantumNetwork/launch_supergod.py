#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子共生系统 - 量子爆发策略回测启动器
"""

import sys
import os
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"quantum_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("QuantumBacktest")

# 主函数
def main():
    logger.info("启动量子爆发策略回测")
    
    # 导入测试模块
    import test_quantum_strategy
    
    # 运行回测
    try:
        test_quantum_strategy.main()
        logger.info("回测完成")
    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}", exc_info=True)
        return 1
    
        return 0

if __name__ == "__main__":
    sys.exit(main()) 