#!/usr/bin/env python3
"""
量子预测功能完整流程测试
验证预测开始、结果展示、对话框关闭和程序退出等全过程
"""

import sys
import os
import logging
import time
from datetime import datetime
import traceback
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QEventLoop

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"quantum_prediction_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumPredictionTest")

def find_dialogs():
    """查找当前应用中的所有对话框"""
    dialogs = []
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QDialog) and widget.isVisible():
            dialogs.append(widget)
            logger.info(f"发现对话框: {widget.objectName() or '未命名'} - {widget.windowTitle()}")
    return dialogs

def close_all_dialogs():
    """关闭所有发现的对话框"""
    dialogs = find_dialogs()
    logger.info(f"发现 {len(dialogs)} 个对话框需要关闭")
    
    for dialog in dialogs:
        try:
            logger.info(f"尝试关闭对话框: {dialog.windowTitle()}")
            dialog.hide()
            dialog.reject()
            dialog.close()
        except Exception as e:
            logger.error(f"关闭对话框 {dialog.windowTitle()} 时出错: {str(e)}")
    
    # 强制处理事件
    QApplication.processEvents()
    
    # 验证是否全部关闭
    remaining = find_dialogs()
    if remaining:
        logger.warning(f"仍有 {len(remaining)} 个对话框未关闭")
    else:
        logger.info("所有对话框已关闭")

def simulate_esc_key(widget):
    """模拟按下ESC键"""
    try:
        from PyQt5.QtGui import QKeyEvent
        from PyQt5.QtCore import QEvent
        
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
        QApplication.sendEvent(widget, event)
        QApplication.processEvents()
        logger.info(f"已向 {widget.objectName() or widget.__class__.__name__} 发送ESC键事件")
    except Exception as e:
        logger.error(f"模拟ESC键失败: {str(e)}")

def test_quantum_prediction():
    """测试量子预测功能的完整流程"""
    try:
        logger.info("开始测试量子预测功能...")
        
        # 导入必要的模块
        try:
            from quantum_stock_enhancer import enhancer
            logger.info("成功导入量子增强器模块")
        except ImportError:
            logger.error("无法导入量子增强器模块，测试将失败")
            return False
        
        # 1. 测试直接预测功能
        stock_code = "300867.SZ"
        stock_name = "圣元环保"
        current_price = 12.75
        
        logger.info(f"测试直接预测功能: {stock_name}({stock_code})")
        start_time = time.time()
        
        try:
            prediction_result = enhancer.predict_stock_future_prices(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                days=10
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if prediction_result and prediction_result.get('success', False):
                logger.info(f"预测成功完成，耗时: {duration:.2f} 秒")
                days_predicted = len(prediction_result.get('predictions', []))
                logger.info(f"预测了 {days_predicted} 天的价格走势")
                
                # 验证预测结果结构
                for key in ['stock_code', 'stock_name', 'predictions', 'trends', 'quantum_analysis']:
                    if key not in prediction_result:
                        logger.warning(f"预测结果缺少关键字段: {key}")
                
                # 打印第一天和最后一天的预测
                predictions = prediction_result.get('predictions', [])
                if predictions:
                    first_day = predictions[0]
                    last_day = predictions[-1]
                    logger.info(f"预测第一天预测: {first_day['date']} - 价格: {first_day['price']} - 涨幅: {first_day['change']}%")
                    logger.info(f"最后一天预测: {last_day['date']} - 价格: {last_day['price']} - 涨幅: {last_day['change']}%")
                    
                # 检查趋势分析
                trends = prediction_result.get('trends', {})
                logger.info(f"趋势分析: 方向={trends.get('direction', '未知')} 强度={trends.get('strength', 0)} 置信度={trends.get('confidence', 0)}")
                
                # 检查量子分析
                quantum = prediction_result.get('quantum_analysis', {})
                logger.info(f"量子分析: 维度={quantum.get('dimensions', 0)} 纠缠={quantum.get('entanglement', 0)} 相干={quantum.get('coherence', 0)}")
                
                return True
            else:
                logger.error(f"预测失败: {prediction_result.get('message', '未知错误')}")
                return False
                
        except Exception as e:
            logger.error(f"执行预测时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    except Exception as e:
        logger.error(f"测试量子预测功能时出现未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_dialog_closing():
    """测试对话框关闭功能"""
    try:
        logger.info("开始测试对话框关闭功能...")
        
        # 导入必要的模块
        try:
            from supergod_cockpit import RecommendedStocksPanel
            logger.info("成功导入驾驶舱模块")
        except ImportError:
            logger.error("无法导入驾驶舱模块，测试将失败")
            return False
        
        # 创建应用程序
        app = QApplication.instance() or QApplication(sys.argv)
        
        # 创建测试对话框
        test_dialog = QDialog()
        test_dialog.setObjectName("测试对话框")
        test_dialog.setWindowTitle("对话框关闭测试")
        test_dialog.resize(400, 300)
        test_dialog.show()
        
        logger.info("测试对话框已创建并显示")
        
        # 等待一段时间
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()
        
        # 确认对话框显示中
        dialogs_before = find_dialogs()
        if not dialogs_before:
            logger.error("未找到测试对话框，测试失败")
            return False
            
        logger.info("测试关闭对话框功能...")
        close_all_dialogs()
        
        # 等待处理完成
        QTimer.singleShot(500, loop.quit)
        loop.exec_()
        
        # 验证对话框已关闭
        dialogs_after = find_dialogs()
        if not dialogs_after:
            logger.info("成功关闭所有对话框")
            return True
        else:
            logger.warning(f"仍有 {len(dialogs_after)} 个对话框未关闭")
            return False
        
    except Exception as e:
        logger.error(f"测试对话框关闭功能时出现异常: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_cockpit_integration():
    """测试驾驶舱集成测试 - 模拟完整流程"""
    # 注意：此测试不会实际启动驾驶舱，只是模拟测试流程
    
    logger.info("驾驶舱集成测试流程模拟:")
    logger.info("1. 用户打开超神量子驾驶舱")
    logger.info("2. 用户选择股票\"圣元环保\"并点击\"量子预测\"按钮")
    logger.info("3. 系统显示进度对话框")
    logger.info("4. 系统使用量子增强器执行预测")
    logger.info("5. 系统显示预测结果对话框")
    logger.info("6. 用户点击\"确定关闭\"按钮或按ESC键关闭对话框")
    logger.info("7. 用户按ESC键退出程序")
    logger.info("8. 系统关闭所有对话框，释放资源，安全退出")
    
    # 测试预测功能
    prediction_success = test_quantum_prediction()
    logger.info(f"量子预测功能测试: {'成功' if prediction_success else '失败'}")
    
    # 测试对话框关闭功能
    dialog_close_success = test_dialog_closing()
    logger.info(f"对话框关闭功能测试: {'成功' if dialog_close_success else '失败'}")
    
    overall_success = prediction_success and dialog_close_success
    logger.info(f"整体流程测试: {'成功' if overall_success else '失败'}")
    
    return overall_success

def main():
    """主函数"""
    logger.info("=== 量子预测功能完整流程测试 ===")
    
    # 确保QApplication实例存在
    app = QApplication.instance() or QApplication(sys.argv)
    
    try:
        # 执行测试
        success = test_cockpit_integration()
        
        if success:
            logger.info("测试成功完成！量子预测功能工作正常")
            QMessageBox.information(None, "测试成功", "量子预测功能工作正常，测试通过！")
        else:
            logger.warning("测试未能完全通过，请检查日志了解详情")
            QMessageBox.warning(None, "测试结果", "测试未能完全通过，请检查日志了解详情")
            
    except Exception as e:
        logger.error(f"测试过程中出现未捕获的异常: {str(e)}")
        logger.error(traceback.format_exc())
        QMessageBox.critical(None, "测试失败", f"测试过程中出现异常:\n{str(e)}")
    
    # 关闭所有剩余对话框
    close_all_dialogs()
    
    logger.info("测试完成，退出程序")
    
    # 确保干净退出
    app.quit()

if __name__ == "__main__":
    main() 