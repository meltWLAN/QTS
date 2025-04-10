#!/usr/bin/env python3
"""
超神系统启动脚本清理工具

此脚本用于清理旧的启动入口点，确保只使用官方入口：launch_supergod.py
"""

import os
import sys
import logging
import shutil
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("CleanupTool")

# 需要处理的旧启动脚本
OLD_LAUNCHERS = [
    "run.py",
    "main.py",
    "run_enhanced_system.py",
    "run_quantum_network.py",
    "gui_app.py",
    "super_desktop_app.py.bak",
    "simple_gui_app.py.bak",
    "supergod_desktop_analysis.py"
]

def main():
    """清理旧的启动脚本"""
    logger.info("=" * 80)
    logger.info("超神系统启动脚本清理工具")
    logger.info("此工具将帮助您清理旧的启动入口，确保只使用官方入口：launch_supergod.py")
    logger.info("=" * 80)
    
    # 检查launch_supergod.py是否存在
    if not os.path.exists("launch_supergod.py"):
        logger.error("错误：找不到官方启动脚本 'launch_supergod.py'")
        logger.error("请确保官方启动脚本存在后再运行此工具")
        return 1
    
    # 创建备份目录
    backup_dir = "backup_launchers"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        logger.info(f"创建备份目录: {backup_dir}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理每个旧启动脚本
    processed_count = 0
    for script in OLD_LAUNCHERS:
        if os.path.exists(script):
            try:
                # 创建备份
                backup_path = os.path.join(backup_dir, f"{script}.{timestamp}.bak")
                shutil.copy2(script, backup_path)
                logger.info(f"已备份 {script} 到 {backup_path}")
                
                # 修改原脚本为重定向脚本
                with open(script, "w", encoding="utf-8") as f:
                    f.write(f"""#!/usr/bin/env python3
\"\"\"
超神系统 - 已废弃的启动入口

此脚本已不再是官方启动入口
请使用 launch_supergod.py 启动超神系统
\"\"\"

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def main():
    \"\"\"提示用户使用官方入口\"\"\"
    print("\\n" + "="*80)
    print("超神系统提示：此脚本已废弃")
    print("请使用官方启动入口：python launch_supergod.py")
    print("="*80 + "\\n")
    return 1

if __name__ == "__main__":
    sys.exit(main())
""")
                logger.info(f"已将 {script} 修改为重定向脚本")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理 {script} 时出错: {str(e)}")
    
    # 总结
    if processed_count > 0:
        logger.info("\n清理完成！")
        logger.info(f"已处理 {processed_count} 个旧启动脚本")
        logger.info(f"所有脚本已备份到 {backup_dir} 目录")
        logger.info("\n请使用官方启动入口：python launch_supergod.py")
    else:
        logger.info("未发现需要处理的旧启动脚本")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 