#!/usr/bin/env python3
"""
超神系统启动重定向脚本

此脚本已不再是主要启动入口
请使用 launch_supergod.py 启动超神系统
"""

import os
import sys
import logging
import subprocess
from quantum_engine import QuantumEngine
from symbiosis_manager import SymbiosisManager
import time
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("QuantumTradingSystem")

def simulate_symbiosis_metrics():
    """模拟共生指标的随机变化"""
    return {
        'coherence': random.uniform(0.0, 1.0),
        'resonance': random.uniform(0.0, 1.0),
        'synergy': random.uniform(0.0, 1.0),
        'stability': random.uniform(0.0, 1.0)
    }

def main():
    """重定向到官方启动入口"""
    logger.info("=" * 80)
    logger.info("超神系统提示：请使用官方启动入口")
    logger.info("请使用 'python launch_supergod.py' 启动超神系统")
    logger.info("=" * 80)
    
    # 检查launch_supergod.py是否存在
    if not os.path.exists("launch_supergod.py"):
        logger.error("错误：找不到官方启动脚本 'launch_supergod.py'")
        logger.error("请确保您的超神系统安装完整")
        return 1
    
    # 询问用户是否自动启动
    print("\n是否要切换到官方启动脚本？(y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y' or choice == 'yes':
        logger.info("正在启动官方超神系统...")
        
        # 准备命令行参数（去掉当前脚本名称）
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        cmd = [sys.executable, "launch_supergod.py"] + args
        
        # 启动官方脚本并等待其完成
        try:
            subprocess.call(cmd)
        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            return 1
    else:
        logger.info("您选择了不自动启动")
        logger.info("请记住，此脚本已不再维护，建议使用官方入口:")
        logger.info("python launch_supergod.py [选项]")

    # 初始化量子计算引擎
    quantum_engine = QuantumEngine(num_qubits=4)
    
    # 初始化共生系统管理器
    symbiosis_manager = SymbiosisManager(quantum_engine)
    
    print("开始量子共生系统演示...")
    print("-" * 50)
    
    # 模拟系统运行
    for i in range(5):
        print(f"\n迭代 {i+1}:")
        
        # 更新共生指标
        new_metrics = simulate_symbiosis_metrics()
        symbiosis_manager.update_metrics(new_metrics)
        
        # 获取系统报告
        report = symbiosis_manager.get_symbiosis_report()
        
        # 打印报告
        print("\n共生指标:")
        for metric, value in report['symbiosis_metrics'].items():
            print(f"{metric}: {value:.3f}")
            
        print(f"\n共生状态: {report['symbiosis_state']}")
        
        print("\n量子计算指标:")
        for metric, value in report['quantum_metrics'].items():
            if isinstance(value, float):
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")
                
        print("-" * 50)
        time.sleep(2)  # 暂停2秒以观察变化

    return 0

if __name__ == "__main__":
    sys.exit(main()) 