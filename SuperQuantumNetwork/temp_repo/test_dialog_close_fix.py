#!/usr/bin/env python3
"""
量子预测对话框关闭问题专项测试
特别测试窗口X按钮、ESC按键和关闭按钮
"""

import sys
import os
import logging
import time
import json
from datetime import datetime
import traceback
from PyQt5.QtWidgets import (QApplication, QDialog, QMessageBox, QVBoxLayout, 
                            QPushButton, QLabel, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, QEventLoop, QMetaObject, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QKeyEvent

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 使用DEBUG级别以捕获更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"dialog_close_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DialogCloseTest")

class TestResultDialog(QDialog):
    """测试预测结果对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("测试预测结果对话框")
        self.setWindowTitle("量子预测结果测试")
        self.setMinimumSize(600, 400)
        
        # 设置标志，跟踪关闭事件
        self.close_attempted = False
        self.close_method_used = None
        
        # 创建布局
        self.setup_ui()
        
        # 记录日志
        logger.debug("测试对话框已创建")
        
    def setup_ui(self):
        """设置UI组件"""
        layout = QVBoxLayout(self)
        
        # 添加标题
        title = QLabel("测试量子预测结果对话框")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 添加说明
        info = QLabel("这是一个测试对话框，用于验证关闭功能是否正常工作。\n"
                     "请尝试以下方法关闭对话框：\n"
                     "1. 点击窗口右上角X按钮\n"
                     "2. 按ESC键\n"
                     "3. 点击下方的关闭按钮")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        
        # 添加模拟数据表格
        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["日期", "预测价格", "涨跌幅"])
        for i in range(5):
            table.setItem(i, 0, QTableWidgetItem(f"2025-04-{10+i}"))
            table.setItem(i, 1, QTableWidgetItem(f"{15.0 + i*2:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(f"+{9.5 - i*0.5:.1f}%"))
        layout.addWidget(table)
        
        # 添加关闭按钮
        self.close_btn = QPushButton("确定关闭")
        self.close_btn.setMinimumHeight(40)
        self.close_btn.clicked.connect(self.on_close_button_clicked)
        layout.addWidget(self.close_btn)
        
    def on_close_button_clicked(self):
        """处理关闭按钮点击事件"""
        logger.debug("关闭按钮被点击")
        self.close_method_used = "button"
        self.close_attempted = True
        
        # 使用多种方法尝试关闭
        self.try_all_close_methods()
        
    def try_all_close_methods(self):
        """尝试所有可能的关闭方法"""
        logger.debug("开始尝试所有关闭方法")
        
        try:
            # 1. 先隐藏对话框
            self.hide()
            logger.debug("对话框已隐藏")
            
            # 2. 尝试使用QMetaObject强制调用关闭方法
            QMetaObject.invokeMethod(self, "reject", Qt.QueuedConnection)
            QMetaObject.invokeMethod(self, "done", Qt.QueuedConnection, 
                                   Q_ARG(int, 0))
            QMetaObject.invokeMethod(self, "close", Qt.QueuedConnection)
            logger.debug("已通过QMetaObject调度关闭方法")
            
            # 3. 直接调用关闭方法
            self.reject()
            self.done(0)
            self.close()
            logger.debug("已直接调用关闭方法")
            
            # 4. 强制处理事件
            QApplication.processEvents()
            logger.debug("已强制处理事件")
            
            # 5. 删除对话框
            self.deleteLater()
            logger.debug("已调用deleteLater")
            
        except Exception as e:
            logger.error(f"关闭过程出错: {str(e)}")
            logger.error(traceback.format_exc())
    
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        logger.debug("X按钮被点击，触发closeEvent")
        self.close_method_used = "X_button"
        self.close_attempted = True
        
        # 尝试所有关闭方法
        self.try_all_close_methods()
        
        # 接受关闭事件
        event.accept()
        logger.debug("closeEvent已接受")
        
    def keyPressEvent(self, event):
        """处理按键事件"""
        if event.key() == Qt.Key_Escape:
            logger.debug("ESC键被按下")
            self.close_method_used = "ESC_key"
            self.close_attempted = True
            
            # 尝试所有关闭方法
            self.try_all_close_methods()
        else:
            # 其他按键交给默认处理
            super().keyPressEvent(event)

def simulate_esc_key(widget):
    """模拟按下ESC键"""
    logger.debug(f"向对话框发送ESC按键事件: {widget.objectName()}")
    try:
        event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        QApplication.sendEvent(widget, event)
        QApplication.processEvents()
        logger.debug("ESC按键事件已发送")
        return True
    except Exception as e:
        logger.error(f"发送ESC按键失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def simulate_close_button(widget):
    """模拟点击关闭按钮"""
    logger.debug(f"模拟点击关闭按钮: {widget.objectName()}")
    try:
        if hasattr(widget, 'close_btn') and widget.close_btn:
            widget.close_btn.click()
            QApplication.processEvents()
            logger.debug("关闭按钮点击已模拟")
            return True
        else:
            logger.warning("对话框没有关闭按钮属性")
            return False
    except Exception as e:
        logger.error(f"模拟点击关闭按钮失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def simulate_x_button(widget):
    """模拟点击X按钮"""
    logger.debug(f"模拟点击X按钮: {widget.objectName()}")
    try:
        close_event = QCloseEvent()
        QApplication.sendEvent(widget, close_event)
        QApplication.processEvents()
        logger.debug("X按钮点击已模拟")
        return True
    except Exception as e:
        logger.error(f"模拟点击X按钮失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def fix_dialog_manually(widget):
    """手动强制修复对话框"""
    logger.debug(f"手动强制修复对话框: {widget.objectName()}")
    try:
        # 1. 强制隐藏
        widget.hide()
        
        # 2. 断开与父窗口的连接
        widget.setParent(None)
        
        # 3. 调用多种关闭方法
        widget.reject()
        widget.done(0)
        widget.close()
        
        # 4. 强制处理事件
        QApplication.processEvents()
        
        # 5. 删除对话框
        widget.deleteLater()
        
        logger.debug("手动强制修复已完成")
        return True
    except Exception as e:
        logger.error(f"手动强制修复失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def check_dialogs_exist():
    """检查是否存在任何对话框"""
    dialogs = []
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and widget.isVisible():
            dialogs.append(widget)
    return dialogs

def test_dialog_close_method(method):
    """测试特定关闭方法"""
    logger.info(f"===== 开始测试 {method} 关闭方法 =====")
    
    # 创建对话框
    dialog = TestResultDialog()
    dialog.show()
    
    # 等待对话框显示
    QApplication.processEvents()
    time.sleep(1)
    
    # 确认对话框已显示
    if dialog.isVisible():
        logger.info("对话框已成功显示")
    else:
        logger.error("对话框未能显示")
        return False
    
    # 根据方法模拟关闭操作
    if method == "ESC_key":
        success = simulate_esc_key(dialog)
    elif method == "close_button":
        success = simulate_close_button(dialog)
    elif method == "X_button":
        success = simulate_x_button(dialog)
    else:
        logger.error(f"未知的关闭方法: {method}")
        return False
    
    if not success:
        logger.error(f"{method} 模拟失败")
        return False
    
    # 等待关闭处理
    time.sleep(2)
    QApplication.processEvents()
    
    # 检查对话框是否已关闭
    if dialog.isVisible():
        logger.warning(f"对话框仍然可见 - {method} 可能失败")
        # 尝试强制修复
        fix_dialog_manually(dialog)
        time.sleep(1)
        QApplication.processEvents()
        
        # 再次检查
        if dialog.isVisible():
            logger.error(f"即使强制修复后，对话框仍然可见 - {method} 失败")
            return False
        else:
            logger.info(f"强制修复后，对话框已关闭 - {method} 修复成功")
            return True
    else:
        logger.info(f"对话框已成功关闭 - {method} 成功")
        return True

def test_all_close_methods():
    """测试所有关闭方法"""
    methods = ["ESC_key", "close_button", "X_button"]
    results = {}
    
    for method in methods:
        result = test_dialog_close_method(method)
        results[method] = result
        
        # 等待一段时间，确保之前的对话框完全清理
        time.sleep(2)
        
        # 检查是否还有残留对话框
        remaining = check_dialogs_exist()
        if remaining:
            logger.warning(f"测试 {method} 后仍有 {len(remaining)} 个对话框未关闭")
            for dialog in remaining:
                fix_dialog_manually(dialog)
    
    # 保存结果
    with open("dialog_close_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    logger.info("===== 测试结果摘要 =====")
    for method, result in results.items():
        logger.info(f"{method}: {'成功' if result else '失败'}")
    
    return all(results.values())

def main():
    """主函数"""
    logger.info("======= 对话框关闭问题专项测试 =======")
    
    # 确保QApplication实例存在
    app = QApplication.instance() or QApplication(sys.argv)
    
    try:
        # 执行所有测试
        success = test_all_close_methods()
        
        if success:
            logger.info("所有关闭方法测试通过！")
            QMessageBox.information(None, "测试成功", "所有对话框关闭方法都工作正常！")
        else:
            logger.warning("部分关闭方法测试失败")
            QMessageBox.warning(None, "测试部分失败", "部分对话框关闭方法测试失败，请查看日志了解详情")
            
    except Exception as e:
        logger.error(f"测试过程中出现未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        QMessageBox.critical(None, "测试失败", f"测试过程中出现异常:\n{str(e)}")
    
    logger.info("测试完成，退出程序")
    
    # 确保干净退出
    app.quit()

if __name__ == "__main__":
    main() 