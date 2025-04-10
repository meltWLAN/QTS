#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 宇宙控制器
管理宇宙共振和能量可视化
"""

import logging
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np

# 导入宇宙共振引擎
from quantum_symbiotic_network.cosmic_resonance import CosmicResonanceEngine

class CosmicController(QObject):
    """宇宙控制器"""
    
    # 信号定义
    resonance_updated = pyqtSignal(dict)
    cosmic_events_updated = pyqtSignal(list)
    energy_level_updated = pyqtSignal(float)
    market_patterns_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("CosmicController")
        
        # 初始化宇宙共振引擎
        self.resonance_engine = CosmicResonanceEngine()
        self.logger.info("宇宙控制器初始化完成")
        
        # 设置更新定时器
        self.resonance_timer = QTimer(self)
        self.resonance_timer.timeout.connect(self.update_resonance_state)
        self.resonance_timer.start(2000)  # 每2秒更新一次状态
        
        # 事件更新定时器
        self.events_timer = QTimer(self)
        self.events_timer.timeout.connect(self.update_cosmic_events)
        self.events_timer.start(5000)  # 每5秒更新一次事件
        
        # 能量更新定时器
        self.energy_timer = QTimer(self)
        self.energy_timer.timeout.connect(self.update_energy_level)
        self.energy_timer.start(3000)  # 每3秒更新一次能量
        
        # 初始化能量水平
        self.energy_level = 0.5
    
    @pyqtSlot()
    def update_resonance_state(self):
        """更新宇宙共振状态"""
        try:
            # 获取当前共振状态
            state = self.resonance_engine.get_resonance_state()
            if state:
                self.resonance_updated.emit(state)
                
                # 更新能量水平
                self.energy_level = state.get("resonance_level", 0.5)
        
        except Exception as e:
            self.logger.error(f"更新宇宙共振状态失败: {str(e)}")
    
    @pyqtSlot()
    def update_cosmic_events(self):
        """更新宇宙事件"""
        try:
            # 获取最近的宇宙事件
            events = self.resonance_engine.get_recent_events(5)
            if events:
                self.cosmic_events_updated.emit(events)
        
        except Exception as e:
            self.logger.error(f"更新宇宙事件失败: {str(e)}")
    
    @pyqtSlot()
    def update_energy_level(self):
        """更新能量水平"""
        try:
            # 能量水平波动
            state = self.resonance_engine.get_resonance_state()
            if state:
                self.energy_level = state.get("resonance_level", 0.5)
                self.energy_level_updated.emit(self.energy_level)
        
        except Exception as e:
            self.logger.error(f"更新能量水平失败: {str(e)}")
    
    def enhance_consciousness(self, consciousness_state):
        """增强量子意识"""
        try:
            if not self.resonance_engine or not consciousness_state:
                return consciousness_state
                
            # 使用宇宙共振增强量子意识
            enhanced = self.resonance_engine.enhance_consciousness(consciousness_state)
            return enhanced
            
        except Exception as e:
            self.logger.error(f"增强量子意识失败: {str(e)}")
            return consciousness_state
    
    def analyze_market_patterns(self, market_data):
        """分析市场模式"""
        try:
            if not self.resonance_engine or not market_data:
                return None
                
            # 分析市场模式与宇宙共振的关联
            analysis = self.resonance_engine.analyze_market_patterns(market_data)
            if analysis:
                self.market_patterns_updated.emit(analysis)
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析市场模式失败: {str(e)}")
            return None
    
    def apply_cosmic_filter(self, data):
        """应用宇宙共振滤波"""
        try:
            if not self.resonance_engine or not data:
                return data
                
            # 应用宇宙共振滤波
            filtered_data = self.resonance_engine.apply_cosmic_filter(data)
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"应用宇宙共振滤波失败: {str(e)}")
            return data
    
    def shutdown(self):
        """关闭宇宙控制器"""
        try:
            self.resonance_timer.stop()
            self.events_timer.stop()
            self.energy_timer.stop()
            
            if self.resonance_engine:
                self.resonance_engine.shutdown()
                
            self.logger.info("宇宙控制器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭宇宙控制器时出错: {str(e)}")
    
    def get_cosmic_events(self, stock_code: str, days: int = 5) -> List[Dict]:
        """
        获取指定股票的宇宙事件
        
        Args:
            stock_code: 股票代码
            days: 天数
            
        Returns:
            List[Dict]: 宇宙事件列表
        """
        self.logger.info(f"正在获取股票 {stock_code} 的宇宙事件")
        
        if not self.resonance_engine or not self.resonance_engine.running:
            self.logger.warning("宇宙共振引擎未运行，无法获取宇宙事件")
            return self._generate_mock_cosmic_events(days)
            
        try:
            # 使用宇宙共振引擎生成宇宙事件
            cosmic_events = self.resonance_engine.generate_cosmic_events(
                stock_code=stock_code,
                days=days
            )
            
            if not cosmic_events:
                self.logger.warning(f"未能为股票 {stock_code} 生成宇宙事件，使用模拟数据")
                return self._generate_mock_cosmic_events(days)
                
            self.logger.info(f"成功获取 {len(cosmic_events)} 个宇宙事件")
            return cosmic_events
            
        except Exception as e:
            self.logger.error(f"获取宇宙事件时出错: {str(e)}")
            return self._generate_mock_cosmic_events(days)
            
    def _generate_mock_cosmic_events(self, days: int = 5) -> List[Dict]:
        """
        生成模拟宇宙事件
        
        Args:
            days: 天数
            
        Returns:
            List[Dict]: 模拟宇宙事件列表
        """
        self.logger.info(f"生成 {days} 天的模拟宇宙事件")
        
        events = []
        current_date = datetime.now()
        
        event_types = ["能量波动", "量子扰动", "意识共鸣", "时间异常"]
        contents = [
            "市场能量场波动加剧，可能引发短期波动",
            "量子概率场出现扭曲，决策点临近",
            "集体意识同步性增强，趋势可能加速形成",
            "时间流异常，历史模式重复率增加"
        ]
        
        for i in range(days):
            date = current_date + timedelta(days=i)
            
            # 只为部分天数生成事件
            if np.random.random() < 0.7:
                event = {
                    "date": date.strftime("%Y-%m-%d"),
                    "type": np.random.choice(event_types),
                    "content": np.random.choice(contents),
                    "strength": round(np.random.uniform(0.3, 0.9), 2),
                    "energy": round(np.random.uniform(0.4, 0.8), 2),
                    "impact": np.random.choice(["轻微", "中等", "显著", "重大"], p=[0.3, 0.4, 0.2, 0.1])
                }
                events.append(event)
        
        self.logger.info(f"成功生成 {len(events)} 个模拟宇宙事件")
        return events
    
    def on_connect_symbiosis(self, symbiosis_core):
        """连接到共生核心时调用
        
        Args:
            symbiosis_core: 共生核心实例
        """
        self.logger.info("宇宙控制器已连接到共生核心")
        self.symbiosis_core = symbiosis_core
        
        # 发送一条连接消息
        if hasattr(symbiosis_core, "send_message"):
            try:
                # 尝试新版API
                symbiosis_core.send_message(
                    source="cosmic_controller",
                    target=None,  # 广播给所有模块
                    message_type="connection",
                    data={"status": "ready"}
                )
            except Exception as e:
                try:
                    # 尝试旧版API
                    symbiosis_core.send_message(
                        source_module="cosmic_controller",
                        target_module=None,  # 广播给所有模块
                        message_type="connection",
                        data={"status": "ready"}
                    )
                except Exception as ee:
                    self.logger.error(f"无法发送连接消息: {str(ee)}")
        
    def on_disconnect_symbiosis(self):
        """从共生核心断开时调用"""
        self.logger.info("宇宙控制器已断开与共生核心的连接")
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
            
            # 处理宇宙事件消息
            if message_type == "cosmic_event":
                event_type = data.get("event_type", "unknown")
                event_message = data.get("message", "")
                
                if event_type and event_message:
                    self.logger.info(f"宇宙事件 [{event_type}]: {event_message}")
                    
                    # 更新界面
                    if self.view and hasattr(self.view, "add_cosmic_event"):
                        self.view.add_cosmic_event(event_type, event_message)
            
            # 处理共振状态更新消息
            elif message_type == "resonance_update":
                resonance_state = data.get("state", {})
                
                if resonance_state:
                    self.logger.debug(f"共振状态更新: {resonance_state}")
                    
                    # 更新界面
                    if self.view and hasattr(self.view, "update_resonance_state"):
                        self.view.update_resonance_state(resonance_state)
                
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}") 