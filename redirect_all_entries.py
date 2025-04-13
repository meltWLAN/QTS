#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
入口点重定向工具

此脚本用于扫描项目目录，查找可能的启动入口点，
并将它们转换为重定向到官方统一入口的脚本。
"""

import os
import sys
import re
import shutil
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("EntryRedirector")

# 官方启动入口
OFFICIAL_ENTRY = "start_quantum.py"

# 可能的入口点模式
ENTRY_PATTERNS = [
    r"main\.py$",
    r"app\.py$",
    r"run.*\.py$",
    r"start.*\.py$",
    r"launch.*\.py$",
    r".*_desktop\.py$",
    r".*_app\.py$",
    r".*_main\.py$"
]

# 排除的文件
EXCLUDED_FILES = [
    OFFICIAL_ENTRY,
    "quantum_system_entry.py",
    "launch_quantum_core.py",
    "redirect_all_entries.py"
]

# 重定向模板
REDIRECT_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子交易系统 - 重定向脚本

此文件已被修改为重定向到官方启动入口。
请使用 '{official_entry}' 启动系统。
"""

import os
import sys
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumTrading.Redirector")

def main():
    """重定向到官方启动入口"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 官方启动入口路径
    official_entry = os.path.join(current_dir, "{official_entry}")
    
    if not os.path.exists(official_entry):
        logger.error(f"错误: 找不到官方启动入口 '{official_entry}'")
        logger.error("请确保您的系统安装完整")
        return 1
    
    logger.warning("=" * 80)
    logger.warning("注意: 此脚本不再是有效的启动入口")
    logger.warning(f"请使用官方启动入口: python {official_entry}")
    logger.warning("=" * 80)
    
    # 询问用户是否要自动重定向
    print("\\n是否自动重定向到官方启动入口? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y' or choice == 'yes':
        logger.info(f"正在重定向到: {official_entry}")
        
        # 转发所有命令行参数
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        cmd = [sys.executable, official_entry] + args
        
        try:
            # 使用子进程调用官方入口
            return subprocess.call(cmd)
        except Exception as e:
            logger.error(f"重定向失败: {str(e)}")
            return 1
    else:
        logger.info("未重定向，退出")
        return 0

if __name__ == "__main__":
    sys.exit(main())
'''

def find_potential_entries(root_dir):
    """查找可能的入口点
    
    Args:
        root_dir: 要搜索的根目录
        
    Returns:
        list: 发现的可能入口点的路径列表
    """
    potential_entries = []
    patterns = [re.compile(pattern) for pattern in ENTRY_PATTERNS]
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in EXCLUDED_FILES:
                continue
                
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            
            # 检查文件名是否匹配任何入口点模式
            if any(pattern.search(file) for pattern in patterns):
                potential_entries.append(file_path)
                continue
                
            # 检查文件内容是否包含主函数和启动相关代码
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if ('if __name__ == "__main__"' in content or 
                        'if __name__ == \'__main__\'' in content) and (
                        'main()' in content or 
                        'app = ' in content or 
                        'QApplication' in content or 
                        'start_' in content or 
                        'launch_' in content):
                        potential_entries.append(file_path)
            except UnicodeDecodeError:
                # 不是文本文件，跳过
                pass
    
    return potential_entries

def backup_file(file_path, backup_dir):
    """备份文件
    
    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录
        
    Returns:
        str: 备份文件的路径
    """
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        
    # 使用原文件路径的相对结构
    rel_path = os.path.basename(file_path)
    backup_path = os.path.join(backup_dir, f"{rel_path}.bak")
    
    # 如果已存在同名备份，添加时间戳
    if os.path.exists(backup_path):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = os.path.join(backup_dir, f"{rel_path}.{timestamp}.bak")
    
    shutil.copy2(file_path, backup_path)
    return backup_path

def redirect_entry_point(file_path, official_entry):
    """将入口点转换为重定向脚本
    
    Args:
        file_path: 要转换的文件路径
        official_entry: 官方入口点
        
    Returns:
        bool: 是否成功转换
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(REDIRECT_TEMPLATE.format(official_entry=official_entry))
        return True
    except Exception as e:
        logger.error(f"转换文件 {file_path} 失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="入口点重定向工具")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描但不修改文件")
    parser.add_argument("--backup-dir", default="backup_entries", help="备份目录，默认为 'backup_entries'")
    parser.add_argument("--dir", default=".", help="要扫描的目录，默认为当前目录")
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.dir)
    backup_dir = os.path.join(root_dir, args.backup_dir)
    
    logger.info(f"开始扫描目录: {root_dir}")
    
    # 查找可能的入口点
    potential_entries = find_potential_entries(root_dir)
    
    if not potential_entries:
        logger.info("未发现潜在的入口点")
        return 0
    
    logger.info(f"发现 {len(potential_entries)} 个潜在入口点:")
    for i, entry in enumerate(potential_entries, 1):
        logger.info(f"{i}. {os.path.relpath(entry, root_dir)}")
    
    if args.dry_run:
        logger.info("干运行模式，不进行修改")
        return 0
    
    # 确认是否继续
    print("\n确认重定向这些入口点吗? 这将修改文件内容 (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice != 'y' and choice != 'yes':
        logger.info("操作已取消")
        return 0
    
    # 处理每个入口点
    success_count = 0
    for entry in potential_entries:
        rel_path = os.path.relpath(entry, root_dir)
        logger.info(f"处理: {rel_path}")
        
        # 备份文件
        backup_path = backup_file(entry, backup_dir)
        logger.info(f"已备份到: {os.path.relpath(backup_path, root_dir)}")
        
        # 重定向入口点
        if redirect_entry_point(entry, OFFICIAL_ENTRY):
            logger.info(f"已重定向: {rel_path}")
            success_count += 1
        else:
            logger.warning(f"重定向失败: {rel_path}")
    
    logger.info(f"操作完成. 成功处理 {success_count}/{len(potential_entries)} 个入口点")
    logger.info(f"原文件已备份到: {args.backup_dir}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n操作已取消")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        sys.exit(1) 