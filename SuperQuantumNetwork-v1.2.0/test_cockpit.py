#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱测试脚本
用于测试驾驶舱模式的启动和退出
"""

import os
import sys
import logging
import traceback
import time
import signal

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cockpit_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CockpitTest")

def setup_signal_handlers():
    """设置信号处理函数以捕获Ctrl+C和其他终止信号"""
    def signal_handler(sig, frame):
        logger.info(f"收到信号 {sig}，准备退出...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def test_cockpit_mode():
    """测试驾驶舱模式"""
    logger.info("开始测试驾驶舱模式...")
    
    try:
        # 导入驾驶舱模块
        logger.info("导入驾驶舱模块...")
        from PyQt5.QtWidgets import QApplication
        from supergod_cockpit import SupergodCockpit
        
        # 初始化数据连接器
        logger.info("初始化数据连接器...")
        from tushare_data_connector import TushareDataConnector
        data_connector = TushareDataConnector(token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
        
        # 创建增强模块
        logger.info("创建增强模块...")
        enhancement_modules = {}
        
        # 尝试加载超神增强器
        try:
            from supergod_enhancer import SupergodEnhancer
            enhancement_modules['supergod_enhancer'] = SupergodEnhancer()
            logger.info("已加载超神增强器")
        except ImportError as e:
            logger.warning(f"无法加载超神增强器: {str(e)}")
        
        # 尝试加载混沌理论框架
        try:
            from chaos_theory_framework import ChaosTheoryAnalyzer
            enhancement_modules['chaos_theory'] = ChaosTheoryAnalyzer()
            logger.info("已加载混沌理论框架")
        except ImportError as e:
            logger.warning(f"无法加载混沌理论框架: {str(e)}")
        
        # 尝试加载量子维度扩展器
        try:
            from quantum_dimension_enhancer import QuantumDimensionEnhancer
            enhancement_modules['quantum_dimension'] = QuantumDimensionEnhancer()
            logger.info("已加载量子维度扩展器")
        except ImportError as e:
            logger.warning(f"无法加载量子维度扩展器: {str(e)}")
        
        # 创建应用
        logger.info("创建PyQt应用...")
        app = QApplication(sys.argv)
        
        # 创建驾驶舱
        logger.info("创建驾驶舱实例...")
        cockpit = SupergodCockpit()
        
        # 设置数据连接器
        logger.info("设置数据连接器...")
        if hasattr(cockpit, 'set_data_connector'):
            cockpit.set_data_connector(data_connector)
        else:
            logger.error("驾驶舱实例没有set_data_connector方法")
        
        # 设置增强模块
        logger.info("设置增强模块...")
        if hasattr(cockpit, 'set_enhancement_modules'):
            cockpit.set_enhancement_modules(enhancement_modules)
        else:
            logger.error("驾驶舱实例没有set_enhancement_modules方法")
        
        # 显示驾驶舱
        logger.info("显示驾驶舱...")
        cockpit.show()
        
        # 设置定时检查
        def check_status():
            """检查应用状态"""
            logger.info("应用仍在运行中...")
            # 每10秒检查一次
            QTimer.singleShot(10000, check_status)
        
        # 导入QTimer并启动检查
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(10000, check_status)
        
        # 设置应用关闭时的清理函数
        def cleanup():
            """应用关闭时的清理函数"""
            logger.info("应用即将关闭，执行清理...")
            if hasattr(cockpit, 'safe_stop'):
                cockpit.safe_stop()
            else:
                logger.warning("驾驶舱实例没有safe_stop方法")
        
        app.aboutToQuit.connect(cleanup)
        
        # 运行应用
        logger.info("启动应用主循环...")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"测试驾驶舱时出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    setup_signal_handlers()
    test_cockpit_mode() 