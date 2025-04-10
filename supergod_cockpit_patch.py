#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱补丁
用于修复supergod_cockpit.py中的语法错误
"""

import os
import sys
import re
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("CockpitPatch")

def fix_cockpit_file():
    """修复驾驶舱文件中的语法错误"""
    source_file = "SuperQuantumNetwork/supergod_cockpit.py"
    target_file = "SuperQuantumNetwork/supergod_cockpit_fixed.py"
    
    logger.info(f"开始修复文件: {source_file}")
    
    if not os.path.exists(source_file):
        logger.error(f"源文件不存在: {source_file}")
        return False
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 搜索需要替换的代码段落
        pattern = re.compile(r'def load_recommended_stocks\(self\):.*?def get_recommended_stocks\(self\):', re.DOTALL)
        match = pattern.search(content)
        
        if not match:
            logger.error("找不到需要修复的代码段落")
            return False
        
        # 原始代码段
        old_code = match.group(0)
        
        # 修复后的代码
        fixed_code = """    def load_recommended_stocks(self):
        """加载推荐股票到表格"""
        try:
            self.stocks_table.setRowCount(0)  # 清空表格

            # 获取推荐股票数据
            stocks = self.get_recommended_stocks()
            
            # 填充表格
            for row, stock in enumerate(stocks):
                self.stocks_table.insertRow(row)
                
                # 代码
                code_item = QTableWidgetItem(stock.get('code', ''))
                code_item.setTextAlignment(Qt.AlignCenter)
                self.stocks_table.setItem(row, 0, code_item)

                # 名称
                name_item = QTableWidgetItem(stock.get('name', ''))
                self.stocks_table.setItem(row, 1, name_item)

                # 最新价
                price = stock.get('price', 0)
                price_item = QTableWidgetItem(f"{price:.2f}")
                price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.stocks_table.setItem(row, 2, price_item)

                # 涨跌幅
                change_pct = stock.get('change_pct', 0) * 100
                change_text = f"{change_pct:+.2f}%" if change_pct != 0 else "0.00%"
                change_item = QTableWidgetItem(change_text)
                change_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # 根据涨跌幅设置颜色
                if change_pct > 0:
                    change_item.setForeground(
                        QBrush(QColor(SupergodColors.POSITIVE)))
                elif change_pct < 0:
                    change_item.setForeground(
                        QBrush(QColor(SupergodColors.NEGATIVE)))

                self.stocks_table.setItem(row, 3, change_item)

                # 推荐度 - 用星星表示
                recommendation = stock.get('recommendation', 0)
                stars = "★" * int(min(5, recommendation / 20))
                recommend_item = QTableWidgetItem(stars)
                recommend_item.setTextAlignment(Qt.AlignCenter)

                # 根据推荐度设置颜色
                if recommendation > 80:
                    recommend_item.setForeground(
                        QBrush(QColor("#FF5555")))  # 红色
                elif recommendation > 60:
                    recommend_item.setForeground(
                        QBrush(QColor("#FFAA00")))  # 橙色
                else:
                    recommend_item.setForeground(
                        QBrush(QColor("#88AAFF")))  # 蓝色

                self.stocks_table.setItem(row, 4, recommend_item)

                # 行业
                industry_item = QTableWidgetItem(stock.get('industry', ''))
                industry_item.setTextAlignment(Qt.AlignCenter)
                self.stocks_table.setItem(row, 5, industry_item)

                # 推荐理由
                reason_item = QTableWidgetItem(stock.get('reason', ''))
                self.stocks_table.setItem(row, 6, reason_item)

        except Exception as e:
            self.logger.error(f"加载推荐股票出错: {str(e)}")
            traceback.print_exc()

    def get_recommended_stocks(self):\n"""
        
        # 替换代码段
        fixed_content = content.replace(old_code, fixed_code)
        
        # 修复可能存在的错误缩进
        pattern2 = re.compile(r'if not reasons:\n                return "基于技术分析和市场数据的综合评估"')
        fixed_content = pattern2.sub(r'if not reasons:\n                return "基于技术分析和市场数据的综合评估"', fixed_content)
        
        # 保存修复后的文件
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        logger.info(f"修复文件保存为: {target_file}")
        
        # 验证修复后的文件语法是否正确
        try:
            import py_compile
            py_compile.compile(target_file)
            logger.info("语法检查通过")
            return True
        except py_compile.PyCompileError as e:
            logger.error(f"修复后的文件仍存在语法错误: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"修复过程中出错: {str(e)}")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print(f"{'超神量子共生系统 - 驾驶舱补丁':^60}")
    print("="*60)
    
    result = fix_cockpit_file()
    
    print("\n" + "="*60)
    if result:
        print("✅ 驾驶舱文件修复成功")
    else:
        print("❌ 驾驶舱文件修复失败")
    print("="*60 + "\n")
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 