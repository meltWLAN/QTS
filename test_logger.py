#!/usr/bin/env python3
"""
测试日志模块功能
"""

import logging
from src.utils.logger import setup_logger, get_module_logger

def main():
    """测试日志模块的主函数"""
    # 设置日志记录器
    setup_logger(log_level=logging.DEBUG)
    
    # 获取当前模块的日志记录器
    logger = get_module_logger(__name__)
    
    # 测试各个日志级别
    logger.debug("这是来自测试脚本的调试日志")
    logger.info("这是来自测试脚本的信息日志")
    logger.warning("这是来自测试脚本的警告日志")
    logger.error("这是来自测试脚本的错误日志")
    logger.critical("这是来自测试脚本的严重错误日志")
    
    # 测试异常信息记录
    try:
        x = 1 / 0
    except Exception as e:
        logger.exception("捕获到异常: %s", str(e))
        
    logger.info("日志测试完成")

if __name__ == "__main__":
    main() 