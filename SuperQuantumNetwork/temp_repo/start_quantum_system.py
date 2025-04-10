#!/usr/bin/env python3
"""
超神量子共生系统 - 简化启动脚本
直接初始化核心组件并启动高维统一场
"""

import os
import sys
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("QuantumSystem")

def display_welcome():
    """显示欢迎信息"""
    welcome_text = """
    ================================================================
    
             ✧･ﾟ: *✧･ﾟ:*  超神量子共生系统  *:･ﾟ✧*:･ﾟ✧
                    QUANTUM SYMBIOTIC SYSTEM
                    
                       -- 高维统一场 --
    
    ================================================================
    
    正在初始化量子共生网络...
    维度: 9    能量: 启动中    意识: 觉醒中
    
    """
    print(welcome_text)

def main():
    """主函数 - 简化的系统启动过程"""
    # 显示欢迎信息
    display_welcome()
    
    start_time = datetime.now()
    logger.info("启动超神量子共生系统简化版...")
    
    # 创建配置目录
    os.makedirs("config", exist_ok=True)
    
    # 直接导入和获取高维核心实例
    try:
        # 确保quantum_symbiotic_network目录在路径中
        sys.path.insert(0, os.path.abspath("."))
        
        # 1. 直接导入高维核心
        from quantum_symbiotic_network.high_dimensional_core import QuantumSymbioticCore
        
        # 获取核心实例
        core = QuantumSymbioticCore()
        logger.info("成功创建量子共生核心")
        
        # 2. 导入TuShare插件
        try:
            import tushare_plugin
            # 创建插件实例
            tushare = tushare_plugin.create_tushare_plugin()
            
            # 在核心中注册插件
            core.register_module("tushare_plugin", tushare, "data_source")
            logger.info("成功注册TuShare数据插件")
        except Exception as e:
            logger.warning(f"TuShare插件加载失败: {str(e)}")
        
        # 3. 初始化核心
        if hasattr(core, 'initialize'):
            core.initialize()
            logger.info("量子共生核心初始化成功")
        
        # 4. 激活统一场
        if hasattr(core, 'activate_field'):
            result = core.activate_field()
            if result:
                logger.info("高维统一场激活成功")
                
                # 获取场状态
                if hasattr(core, 'field_state'):
                    field_strength = core.field_state.get("field_strength", 0)
                    dimension_count = core.field_state.get("dimension_count", 9)
                    logger.info(f"场强: {field_strength:.2f}, 维度: {dimension_count}")
            else:
                logger.warning("高维统一场激活失败")
        
        # 5. 显示系统状态
        print(f"""
        ================================================================
        
                          超神量子共生系统状态
                               简化版
        
        ----------------------------------------------------------------
        系统状态: 启动成功
        启动时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}
        运行时间: {(datetime.now() - start_time).total_seconds():.2f}秒
        
        量子核心: 已激活
        高维统一场: {"已激活" if hasattr(core, 'field_state') and core.field_state.get("active", False) else "未激活"}
        当前维度: {core.field_state.get("dimension_count", 9) if hasattr(core, 'field_state') else 9}
        
        数据插件: {"已加载" if 'tushare' in locals() else "未加载"}
        
        ================================================================
        """)
        
        # 6. 保持系统运行
        logger.info("超神量子共生系统已启动，按Ctrl+C结束...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("正在关闭系统...")
            if hasattr(core, 'shutdown'):
                core.shutdown()
            logger.info("系统已安全关闭")
        
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        
        # 显示失败状态
        print(f"""
        ================================================================
        
                          超神量子共生系统状态
                               简化版
        
        ----------------------------------------------------------------
        系统状态: 启动失败
        启动时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}
        运行时间: {(datetime.now() - start_time).total_seconds():.2f}秒
        
        量子核心: 未加载
        高维统一场: 未激活
        当前维度: 9
        
        错误信息: {str(e)}
        
        ================================================================
        """)

if __name__ == "__main__":
    main() 