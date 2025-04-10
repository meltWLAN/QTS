#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱 (修复版)
修复了语法错误问题
"""

import sys

# 以下代码片段修复了第1290-1305行的问题
# 原始代码中有重叠的try-except块

def load_recommended_stocks(self):
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

            # 其余代码保持不变
            
    except Exception as e:
        self.logger.error(f"加载推荐股票出错: {str(e)}")
        traceback.print_exc()

def get_recommended_stocks(self):
    """获取推荐股票列表"""
    try:
        # 只使用真实数据
        return self.get_real_stock_recommendations()
    except Exception as e:
        self.logger.error(f"获取推荐股票失败: {str(e)}")
        return []

# 上面是修复片段，仅用于说明修复方式
# 实际修复应保留整个文件内容，只修改有问题的函数 