#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 量子意识控制器
管理量子意识对象，协调模型和视图
"""

import logging
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
import threading
import time
from datetime import datetime

# 导入量子意识模型
from quantum_symbiotic_network.quantum_consciousness import QuantumConsciousness

class ConsciousnessController(QObject):
    """量子意识控制器"""
    
    # 信号定义
    consciousness_updated = pyqtSignal(dict)
    insights_updated = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("ConsciousnessController")
        
        # 初始化量子意识模型
        self.consciousness = QuantumConsciousness()
        self.logger.info("量子意识控制器初始化完成")
        
        # 设置更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_consciousness_state)
        self.update_timer.start(1500)  # 每1.5秒更新一次状态
    
    @pyqtSlot()
    def update_consciousness_state(self):
        """更新量子意识状态"""
        try:
            # 获取当前意识状态
            state = self.consciousness.get_consciousness_state()
            if state:
                self.consciousness_updated.emit(state)
            
            # 获取最近的洞察
            insights = self.consciousness.get_recent_insights(5)
            if insights:
                self.insights_updated.emit(insights)
        
        except Exception as e:
            self.logger.error(f"更新量子意识状态失败: {str(e)}")
    
    def enhance_prediction(self, prediction_data):
        """使用量子意识增强预测结果"""
        try:
            if not self.consciousness or not prediction_data:
                return prediction_data
                
            # 使用量子意识增强预测
            enhanced = self.consciousness.enhance_prediction(prediction_data)
            return enhanced
            
        except Exception as e:
            self.logger.error(f"量子意识增强预测失败: {str(e)}")
            return prediction_data
    
    def shutdown(self):
        """关闭量子意识控制器"""
        try:
            self.update_timer.stop()
            if self.consciousness:
                self.consciousness.shutdown()
            self.logger.info("量子意识控制器已关闭")
        except Exception as e:
            self.logger.error(f"关闭量子意识控制器时出错: {str(e)}")
    
    def on_connect_symbiosis(self, symbiosis_core):
        """连接到共生核心时调用
        
        Args:
            symbiosis_core: 共生核心实例
        """
        self.logger.info("量子意识控制器已连接到共生核心")
        self.symbiosis_core = symbiosis_core
        
        # 发送一条连接消息
        if hasattr(symbiosis_core, "send_message"):
            try:
                # 尝试新版API
                symbiosis_core.send_message(
                    source="consciousness_controller",
                    target=None,  # 广播给所有模块
                    message_type="connection",
                    data={"status": "ready"}
                )
            except Exception as e:
                try:
                    # 尝试旧版API
                    symbiosis_core.send_message(
                        source_module="consciousness_controller",
                        target_module=None,  # 广播给所有模块
                        message_type="connection",
                        data={"status": "ready"}
                    )
                except Exception as ee:
                    self.logger.error(f"无法发送连接消息: {str(ee)}")
        
    def on_disconnect_symbiosis(self):
        """从共生核心断开时调用"""
        self.logger.info("量子意识控制器已断开与共生核心的连接")
        self.symbiosis_core = None
        
    def on_message(self, message):
        """处理来自共生核心的消息
        
        Args:
            message: 消息数据
        """
        try:
            message_type = message.get("type", "")
            source = message.get("source", "unknown")
            data = message.get("data", {})
            
            self.logger.debug(f"收到消息 [{message_type}] 来自 {source}")
            
            # 处理量子意识洞察消息
            if message_type == "consciousness_insight":
                insight_message = data.get("message", "")
                confidence = data.get("confidence", 0.0)
                
                if insight_message:
                    self.logger.info(f"量子意识洞察 (信心: {confidence:.2f}): {insight_message}")
                    
                    # 记录洞察到历史
                    self.insights.append({
                        "message": insight_message,
                        "confidence": confidence,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 更新界面
                    if self.view and hasattr(self.view, "add_insight"):
                        self.view.add_insight(insight_message, confidence)
            
            # 处理意识状态更新消息
            elif message_type == "consciousness_state":
                state = data.get("state", {})
                
                if state:
                    self.logger.debug(f"意识状态更新: {state}")
                    self.consciousness_state = state
                    
                    # 更新界面
                    if self.view and hasattr(self.view, "update_consciousness_state"):
                        self.view.update_consciousness_state(state)
                
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}")
            
    @property
    def insights(self):
        """获取收集的洞察
        
        Returns:
            list: 收集的洞察列表
        """
        if not hasattr(self, "_insights"):
            self._insights = []
        return self._insights 