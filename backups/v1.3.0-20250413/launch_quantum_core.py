#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超神量子核心启动器
提供简单便捷的方式启动量子系统，支持多种运行模式
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import traceback

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"quantum_core_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumCore.Launcher")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='超神量子核心启动器')
    
    parser.add_argument('--mode', default='desktop', choices=['desktop', 'console', 'server', 'cockpit'],
                      help='运行模式 (桌面GUI/控制台/服务器/驾驶舱)')
    
    parser.add_argument('--debug', action='store_true',
                      help='启用调试模式')
    
    parser.add_argument('--no-real-data', action='store_true',
                      help='禁用实时数据获取，使用模拟数据')
    
    parser.add_argument('--qubits', type=int, default=24,
                      help='设置量子比特数量 (默认: 24)')
    
    parser.add_argument('--strategy', type=str, default='quantum_enhanced',
                      help='使用的策略名称 (默认: quantum_enhanced)')
    
    parser.add_argument('--config', type=str,
                      help='配置文件路径')
    
    parser.add_argument('--port', type=int, default=8000,
                      help='服务器模式端口号 (默认: 8000)')
    
    parser.add_argument('--evolution-level', type=int, default=1, choices=[0, 1, 2, 3],
                      help='自我进化级别 (0=禁用, 1=低, 2=中, 3=高)')
    
    return parser.parse_args()

def load_system_config(config_path=None):
    """加载系统配置"""
    import json
    
    # 默认配置
    default_config = {
        "system": {
            "name": "超神量子核心系统",
            "version": "1.3.0-20250413",
            "max_threads": 8,
            "log_level": "INFO"
        },
        "quantum": {
            "backend_type": "simulator",
            "max_qubits": 24,
            "optimization_level": 2,
            "use_gpu": True,
            "seed": 42
        },
        "market_data": {
            "use_real_data": True,
            "providers": ["tushare", "yahoo", "eastmoney"],
            "preferred_provider": "tushare",
            "cache_timeout": 3600
        },
        "strategy": {
            "default": "quantum_enhanced",
            "backtest_period": 365,
            "max_positions": 10,
            "risk_control": "adaptive"
        },
        "ui": {
            "theme": "dark",
            "font_size": 12,
            "update_interval": 1000
        }
    }
    
    # 如果指定了配置文件，尝试加载
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # 合并配置
            import copy
            config = copy.deepcopy(default_config)
            
            # 递归更新配置
            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
                
            config = update_dict(config, user_config)
            logger.info(f"已加载配置: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            logger.info("使用默认配置")
            return default_config
    
    logger.info("使用默认配置")
    return default_config

def start_desktop_mode(config):
    """启动桌面模式"""
    logger.info("启动桌面模式")
    try:
        from quantum_desktop.main import QuantumDesktopApp
        app = QuantumDesktopApp()
        
        # 如果有传入的配置，则更新应用配置
        if config:
            app.config.update(config)
            logger.info(f"应用已更新配置: {config}")
            
        app.run()
    except Exception as e:
        logger.error(f"启动桌面应用失败: {str(e)}")
        traceback.print_exc()

def start_console_mode(config):
    """启动控制台模式"""
    logger.info("正在启动控制台模式...")
    
    try:
        # 导入控制台模式
        from quantum_core.console_mode import start_console
        
        # 启动控制台
        start_console(config)
        return 0
    except ImportError:
        logger.error("控制台模式组件未找到")
        return 1
    except Exception as e:
        logger.error(f"启动控制台模式失败: {str(e)}")
        return 1

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 加载配置
    config = load_system_config(args.config)
    
    # 将命令行参数合并到配置中
    if args.debug:
        config['system']['log_level'] = 'DEBUG'
    
    if args.no_real_data:
        config['market_data']['use_real_data'] = False
        
    if args.qubits:
        config['quantum']['max_qubits'] = args.qubits
        
    if args.evolution_level:
        config['quantum']['evolution_level'] = args.evolution_level
    
    if args.strategy:
        config['strategy']['default'] = args.strategy
    
    logger.info(f"最终配置: {config}")
    
    # 根据模式启动系统
    if args.mode == 'desktop':
        return start_desktop_mode(config)
    elif args.mode == 'console':
        return start_console_mode(config)
    elif args.mode == 'server':
        logger.info("服务器模式暂未实现")
        return 1
    elif args.mode == 'cockpit':
        logger.info("正在启动驾驶舱模式...")
        try:
            from supergod_cockpit_launcher import launch_cockpit
            return launch_cockpit(config)
        except ImportError:
            logger.error("驾驶舱模式组件未找到")
            return 1
    else:
        logger.error(f"不支持的模式: {args.mode}")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"未处理的异常: {str(e)}", exc_info=True)
        sys.exit(1)
