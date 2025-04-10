#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子增强驾驶舱启动脚本
"""

import os
import sys
import logging
import argparse
import traceback
from datetime import datetime

# 设置详细的日志配置
def setup_logging():
    """配置详细的日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"quantum_cockpit_{timestamp}.log"
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging()
    logger = logging.getLogger("QuantumCockpit")
    logger.info("超神量子增强驾驶舱启动脚本 v2.1.0")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="超神量子增强驾驶舱启动选项")
    parser.add_argument("--quantum-level", type=int, default=10, 
                        help="量子增强级别 (1-10, 默认: 10)")
    parser.add_argument("--debug", action="store_true", 
                        help="启用调试模式")
    args = parser.parse_args()
    
    # 验证量子级别
    quantum_level = max(1, min(10, args.quantum_level))
    if quantum_level != args.quantum_level:
        logger.warning(f"量子级别已调整为有效范围 [1-10]: {quantum_level}")
    
    logger.info(f"量子增强级别: {quantum_level}")
    logger.info(f"调试模式: {'已启用' if args.debug else '已禁用'}")
    logger.info(f"日志文件: {log_file}")
    
    # 设置全局异常处理
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """全局未捕获异常处理器"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 正常处理Ctrl+C
            logger.info("用户中断操作 (Ctrl+C)")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("发生未捕获的异常")
        logger.critical(f"异常类型: {exc_type.__name__}")
        logger.critical(f"异常信息: {str(exc_value)}")
        logger.critical("堆栈跟踪:")
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in tb_lines:
            logger.critical(line.rstrip())
    
    # 设置全局异常处理器
    sys.excepthook = global_exception_handler
    
    try:
        # 显示启动信息
        logger.info("正在初始化超神量子增强驾驶舱...")
        
        # 导入模块
        try:
            from supergod_cockpit import run_cockpit
            logger.info("已成功导入驾驶舱模块")
        except ImportError as e:
            logger.critical(f"导入驾驶舱模块失败: {str(e)}")
            logger.critical("确保supergod_cockpit.py在当前目录或Python路径中")
            return 1
        
        # 启动驾驶舱
        logger.info(f"正在启动驾驶舱，量子级别: {quantum_level}...")
        return run_cockpit(quantum_level=quantum_level)
        
    except Exception as e:
        logger.critical(f"启动过程中发生错误: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"\n错误: 启动失败 - {str(e)}")
        print(f"详细信息请查看日志文件: {log_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 