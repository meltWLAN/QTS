#!/usr/bin/env python3
"""
超神量子共生系统 - 回滚工具
回滚系统到之前的状态
"""

import os
import sys
import shutil
import json
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rollback_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RollbackSystem")

def load_backup_config():
    """加载备份配置"""
    config_path = "backup_config.json"
    if not os.path.exists(config_path):
        logger.error(f"未找到备份配置文件: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"已加载备份配置，上次备份时间: {config.get('last_backup', '未知')}")
        return config
    except Exception as e:
        logger.error(f"加载备份配置文件失败: {str(e)}")
        return None

def restore_file_from_backup(file_path, backup_suffix=".bak"):
    """从备份恢复文件"""
    backup_path = f"{file_path}{backup_suffix}"
    
    if not os.path.exists(backup_path):
        logger.error(f"未找到备份文件: {backup_path}")
        return False
    
    try:
        # 如果原文件存在，先创建一个临时备份
        if os.path.exists(file_path):
            temp_backup = f"{file_path}.tmp"
            shutil.copy2(file_path, temp_backup)
            logger.info(f"已创建临时备份: {temp_backup}")
        
        # 从备份恢复
        shutil.copy2(backup_path, file_path)
        logger.info(f"已从 {backup_path} 恢复 {file_path}")
        return True
    except Exception as e:
        logger.error(f"恢复文件 {file_path} 失败: {str(e)}")
        return False

def rollback_core_files():
    """回滚核心文件"""
    core_files = [
        "sector_rotation_tracker.py",
        "quantum_desktop_app.py",
        "quantum_ui_components.py"
    ]
    
    success_count = 0
    for file in core_files:
        if restore_file_from_backup(file):
            success_count += 1
    
    if success_count == len(core_files):
        logger.info("所有核心文件回滚成功")
        return True
    elif success_count > 0:
        logger.warning(f"部分核心文件回滚成功 ({success_count}/{len(core_files)})")
        return True
    else:
        logger.error("所有核心文件回滚失败")
        return False

def apply_fix_scripts():
    """应用修复脚本"""
    # 应用板块轮动跟踪器修复
    try:
        logger.info("应用板块轮动跟踪器修复脚本...")
        
        # 导入修复脚本而不是直接运行它
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from fix_rotation_tracker import main as fix_rotation_tracker_main
        
        fix_rotation_tracker_main()
        logger.info("板块轮动跟踪器修复脚本应用成功")
        return True
    except Exception as e:
        logger.error(f"应用板块轮动跟踪器修复脚本失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超神量子共生系统回滚工具")
    parser.add_argument("--mode", "-m", choices=["backup", "fix", "all"], default="all",
                       help="回滚模式: backup=从备份恢复, fix=应用修复脚本, all=两者都执行")
    parser.add_argument("--skip-confirm", "-y", action="store_true",
                       help="跳过确认提示")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  超神量子共生系统 - 回滚工具")
    print("="*60)
    
    # 加载备份配置
    config = load_backup_config()
    if config:
        print(f"\n上次备份时间: {config.get('last_backup', '未知')}")
        if 'versions' in config and config['versions']:
            print(f"可用版本: {', '.join(config['versions'])}")
    
    # 确认
    if not args.skip_confirm:
        confirm = input("\n确定要回滚系统吗? 这将覆盖当前文件。(y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    success = True
    
    # 从备份恢复
    if args.mode in ["backup", "all"]:
        logger.info("开始从备份恢复文件...")
        if rollback_core_files():
            print("\n✅ 从备份恢复核心文件成功")
        else:
            print("\n❌ 从备份恢复核心文件失败")
            success = False
    
    # 应用修复脚本
    if args.mode in ["fix", "all"]:
        logger.info("开始应用修复脚本...")
        if apply_fix_scripts():
            print("\n✅ 应用修复脚本成功")
        else:
            print("\n❌ 应用修复脚本失败")
            success = False
    
    if success:
        print("\n" + "="*60)
        print("  ✅ 系统回滚完成!")
        print("="*60)
        print("\n建议使用以下命令启动系统:")
        print("  python supergod_cockpit.py  # 驾驶舱模式")
        print("  python run_supergod.py      # 常规模式\n")
    else:
        print("\n" + "="*60)
        print("  ⚠️ 系统回滚部分完成，可能仍然存在问题")
        print("="*60)

if __name__ == "__main__":
    main() 