#!/usr/bin/env python3
"""
修复SuperQuantumNetwork/supergod_cockpit.py中的缩进问题
"""

import os

def fix_indentation():
    """修复缩进问题"""
    file_path = 'SuperQuantumNetwork/supergod_cockpit.py'
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 修复缩进问题
    lines[1294] = '        try:\n'
    lines[1295] = '            # 只使用真实数据\n'
    lines[1296] = '            return self.get_real_stock_recommendations()\n'
    lines[1297] = '        except Exception as e:\n'
    lines[1298] = '            self.logger.error(f"获取推荐股票失败: {str(e)}")\n'
    lines[1299] = '            return []\n'
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"成功修复 {file_path} 中的缩进问题")

if __name__ == "__main__":
    fix_indentation() 