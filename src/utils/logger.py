#!/usr/bin/env python3
"""
日志工具模块 - 为量子共生网络提供统一的日志记录功能
"""

import os
import sys
import time
import logging
import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(log_file=None, log_level=logging.INFO, log_to_console=True,
                 log_format=None, max_file_size=10*1024*1024, backup_count=5):
    """
    设置全局日志记录器。
    
    参数:
        log_file (str): 日志文件路径，如果为None则根据运行的脚本名称自动生成
        log_level (int): 日志级别，默认为INFO
        log_to_console (bool): 是否输出到控制台
        log_format (str): 日志格式，如果为None则使用默认格式
        max_file_size (int): 日志文件最大大小，默认为10MB
        backup_count (int): 保留的备份日志文件数量
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 如果没有提供日志文件名，根据调用脚本自动生成
    if log_file is None:
        # 获取调用脚本的文件名（不含路径和扩展名）
        calling_script = os.path.basename(sys.argv[0])
        script_name = os.path.splitext(calling_script)[0]
        
        # 如果是像"python -m module"这样的调用，则使用模块名
        if script_name == '':
            script_name = "quantum_symbiotic"
            
        # 创建日志目录（如果不存在）
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建带有时间戳的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    # 设置默认的日志格式
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(log_format)
    
    # 添加文件处理器
    if log_file:
        # 确保日志文件的目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # 创建滚动文件处理器
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 记录首条日志，包含环境信息
    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成，日志文件: %s", log_file if log_file else "仅控制台输出")
    logger.debug("Python版本: %s", sys.version)
    logger.debug("工作目录: %s", os.getcwd())
    
    return logger


def get_module_logger(name):
    """
    获取模块专用的日志记录器。
    
    参数:
        name (str): 模块名称
        
    返回:
        logging.Logger: 模块专用的日志记录器
    """
    return logging.getLogger(name)


# 为常用日志级别提供简便的日志记录函数
def log_info(message, *args, **kwargs):
    """记录INFO级别的日志"""
    logging.getLogger().info(message, *args, **kwargs)


def log_error(message, *args, **kwargs):
    """记录ERROR级别的日志"""
    logging.getLogger().error(message, *args, **kwargs)


def log_warning(message, *args, **kwargs):
    """记录WARNING级别的日志"""
    logging.getLogger().warning(message, *args, **kwargs)


def log_debug(message, *args, **kwargs):
    """记录DEBUG级别的日志"""
    logging.getLogger().debug(message, *args, **kwargs)


def log_critical(message, *args, **kwargs):
    """记录CRITICAL级别的日志"""
    logging.getLogger().critical(message, *args, **kwargs)


def setup_backtest_logger(strategy_name=None, log_level=logging.INFO):
    """
    设置回测专用的日志记录器。
    
    参数:
        strategy_name (str): 策略名称，用于生成日志文件名
        log_level (int): 日志级别，默认为INFO
        
    返回:
        logging.Logger: 配置好的回测日志记录器
    """
    # 确定策略名称
    if strategy_name is None:
        # 获取调用脚本的文件名作为策略名称
        calling_script = os.path.basename(sys.argv[0])
        strategy_name = os.path.splitext(calling_script)[0]
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件名
    log_file = f"quantum_backtest_{timestamp}.log"
    
    # 使用已有的setup_logger函数
    return setup_logger(
        log_file=log_file,
        log_level=log_level,
        log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        max_file_size=20*1024*1024,  # 回测日志可能较大，设为20MB
        backup_count=10  # 保留更多备份
    )


if __name__ == "__main__":
    # 测试日志记录功能
    test_logger = setup_logger(log_level=logging.DEBUG)
    test_logger.debug("这是一条调试日志")
    test_logger.info("这是一条信息日志")
    test_logger.warning("这是一条警告日志")
    test_logger.error("这是一条错误日志")
    test_logger.critical("这是一条严重错误日志") 