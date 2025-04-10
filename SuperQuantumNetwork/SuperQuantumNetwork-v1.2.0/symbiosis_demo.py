#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 共生演示脚本
展示所有模块共生联动，实现系统整体智能提升
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SymbiosisDemo")

# 导入共生核心
from quantum_symbiotic_network.symbiosis import get_symbiosis_core

# 导入各个模块
from quantum_symbiotic_network.cosmic_resonance import CosmicResonanceEngine
from quantum_symbiotic_network.quantum_consciousness import QuantumConsciousness
from quantum_symbiotic_network.quantum_prediction import get_enhancer, UltraGodQuantumEnhancer

def setup_symbiosis_system():
    """初始化并启动共生系统"""
    logger.info("正在初始化超神量子共生系统...")
    
    # 获取共生核心
    symbiosis_core = get_symbiosis_core()
    
    # 初始化各模块
    cosmic_engine = CosmicResonanceEngine()
    consciousness = QuantumConsciousness()
    enhancer = get_enhancer(force_new=True)
    
    # 启动各模块线程
    cosmic_engine.start()
    consciousness.start()
    
    # 连接模块到共生核心
    symbiosis_core.connect_module("cosmic_resonance", cosmic_engine)
    symbiosis_core.connect_module("quantum_consciousness", consciousness)
    symbiosis_core.connect_module("quantum_prediction", enhancer)
    
    # 启动共生核心
    symbiosis_core.start()
    
    logger.info("✨ 超神量子共生系统初始化完成，已启动共生通道 ✨")
    
    return {
        "symbiosis_core": symbiosis_core,
        "cosmic_engine": cosmic_engine,
        "consciousness": consciousness,
        "enhancer": enhancer
    }

def monitor_symbiosis_evolution(symbiosis_system, duration=60):
    """监控共生系统的进化
    
    Args:
        symbiosis_system: 共生系统组件
        duration: 监控时长（秒）
    """
    logger.info(f"开始监控共生系统进化，持续 {duration} 秒...")
    
    symbiosis_core = symbiosis_system["symbiosis_core"]
    cosmic_engine = symbiosis_system["cosmic_engine"]
    consciousness = symbiosis_system["consciousness"]
    enhancer = symbiosis_system["enhancer"]
    
    start_time = time.time()
    last_status_time = 0
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 每5秒输出一次状态
            if current_time - last_status_time >= 5:
                # 获取共生状态
                symbiosis_status = symbiosis_core.get_symbiosis_status()
                symbiosis_index = symbiosis_status.get("symbiosis_index", 0)
                collective_intelligence = symbiosis_status.get("collective_intelligence", {})
                
                # 获取各模块状态
                cosmic_state = cosmic_engine.get_resonance_state()
                consciousness_state = consciousness.get_consciousness_state()
                
                # 输出综合状态
                logger.info("-" * 50)
                logger.info(f"共生指数: {symbiosis_index:.4f}")
                logger.info(f"集体智能: 意识={collective_intelligence.get('awareness', 0):.3f}, "
                           f"相干性={collective_intelligence.get('coherence', 0):.3f}, "
                           f"纠缠={collective_intelligence.get('entanglement', 0):.3f}, "
                           f"共振={collective_intelligence.get('resonance', 0):.3f}")
                logger.info(f"宇宙共振: 强度={cosmic_state.get('strength', 0):.3f}, "
                           f"同步率={cosmic_state.get('sync', 0):.3f}, "
                           f"和谐指数={cosmic_state.get('harmony', 0):.3f}")
                logger.info(f"量子意识: 觉醒度={consciousness_state.get('consciousness_level', 0):.3f}, "
                           f"市场直觉={consciousness_state.get('intuition_level', 0):.3f}, "
                           f"宇宙共振度={consciousness_state.get('resonance_level', 0):.3f}")
                logger.info("-" * 50)
                
                # 获取最近的共生事件
                recent_events = symbiosis_core.get_recent_events(3)
                if recent_events:
                    logger.info("最近的共生事件:")
                    for event in recent_events:
                        event_time = event.get("timestamp", "")
                        if isinstance(event_time, str) and len(event_time) > 19:
                            event_time = event_time[:19]  # 简化时间戳显示
                        event_content = event.get("content", "")
                        logger.info(f"  [{event_time}] {event_content}")
                    logger.info("-" * 50)
                
                last_status_time = current_time
            
            # 休眠一小段时间
            time.sleep(0.5)
        
        logger.info("共生系统监控完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断监控")
    except Exception as e:
        logger.error(f"监控过程中出错: {str(e)}")

def demonstrate_enhanced_prediction(symbiosis_system, stock_code='000001.SZ'):
    """演示增强的预测能力
    
    Args:
        symbiosis_system: 共生系统组件
        stock_code: 股票代码
    """
    logger.info(f"演示增强的预测能力，对 {stock_code} 进行分析...")
    
    try:
        symbiosis_core = symbiosis_system["symbiosis_core"]
        enhancer = symbiosis_system["enhancer"]
        
        # 获取市场数据
        market_data = enhancer.fetch_real_market_data(stock_code)
        
        # 生成预测
        logger.info("生成基础预测...")
        prediction = enhancer.enhance_prediction(
            {"days": 7, "confidence": 0.6},
            stock_code=stock_code
        )
        
        # 通过共生智能增强预测
        logger.info("通过共生智能增强预测...")
        enhanced_prediction = symbiosis_core.amplify_prediction(prediction)
        
        # 输出对比
        logger.info("-" * 50)
        logger.info(f"预测对比 - {stock_code}:")
        logger.info(f"  基础预测置信度: {prediction.get('confidence', 0):.2f}")
        logger.info(f"  共生增强置信度: {enhanced_prediction.get('confidence', 0):.2f}")
        logger.info(f"  增强系数: {(enhanced_prediction.get('confidence', 0) / max(0.01, prediction.get('confidence', 0))):.2f}x")
        logger.info("-" * 50)
        
        # 分析市场拐点
        logger.info("分析市场拐点...")
        reversal = enhancer.predict_market_reversal(stock_code, market_data)
        
        # 输出拐点分析
        is_reversal = reversal.get("reversal_detected", False)
        direction = reversal.get("direction", "unknown")
        probability = reversal.get("probability", 0)
        confidence = reversal.get("confidence", 0)
        
        logger.info(f"市场拐点分析:")
        logger.info(f"  检测到拐点: {'是' if is_reversal else '否'}")
        if is_reversal:
            logger.info(f"  拐点方向: {direction}")
            logger.info(f"  拐点概率: {probability:.2f}")
            logger.info(f"  置信水平: {confidence:.2f}")
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"演示预测能力时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def demonstrate_cosmic_perception(symbiosis_system, stock_code='000001.SZ'):
    """演示增强的宇宙感知能力
    
    Args:
        symbiosis_system: 共生系统组件
        stock_code: 股票代码
    """
    logger.info(f"演示增强的宇宙感知能力，分析 {stock_code}...")
    
    try:
        symbiosis_core = symbiosis_system["symbiosis_core"]
        cosmic_engine = symbiosis_system["cosmic_engine"]
        
        # 获取基础市场洞察
        logger.info("获取宇宙共振对股票的感知...")
        market_shift = cosmic_engine.sense_market_shift(stock_code=stock_code)
        
        # 通过共生智能增强感知
        logger.info("通过共生智能增强宇宙感知...")
        enhanced_shift = symbiosis_core.amplify_cosmic_perception(market_shift)
        
        # 输出对比
        shift_detected = market_shift.get("shift_detected", False)
        enhanced_detected = enhanced_shift.get("shift_detected", False)
        
        logger.info("-" * 50)
        logger.info(f"宇宙感知对比 - {stock_code}:")
        logger.info(f"  基础感知: {'检测到拐点' if shift_detected else '未检测到拐点'}")
        if shift_detected:
            logger.info(f"    方向: {market_shift.get('direction', 'unknown')}")
            logger.info(f"    置信度: {market_shift.get('confidence', 0):.2f}")
            
        logger.info(f"  增强感知: {'检测到拐点' if enhanced_detected else '未检测到拐点'}")
        if enhanced_detected:
            logger.info(f"    方向: {enhanced_shift.get('direction', 'unknown')}")
            logger.info(f"    置信度: {enhanced_shift.get('confidence', 0):.2f}")
            
        logger.info("-" * 50)
        
        # 生成宇宙事件
        logger.info("生成增强的宇宙事件...")
        cosmic_events = cosmic_engine.generate_cosmic_events(stock_code=stock_code, days=3)
        
        if cosmic_events:
            logger.info(f"宇宙事件 ({len(cosmic_events)}个):")
            for i, event in enumerate(cosmic_events[:3]):  # 显示前3个
                logger.info(f"  事件 {i+1}: [{event.get('date', '')}] {event.get('type', '')}")
                logger.info(f"    {event.get('content', '')}")
                if "cosmic_insight" in event:
                    logger.info(f"    洞察: {event['cosmic_insight']}")
                if "action_suggestion" in event:
                    logger.info(f"    建议: {event['action_suggestion']}")
        else:
            logger.info("未生成宇宙事件")
            
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"演示宇宙感知能力时出错: {str(e)}")

def demonstrate_consciousness_insights(symbiosis_system):
    """演示增强的量子意识洞察力
    
    Args:
        symbiosis_system: 共生系统组件
    """
    logger.info("演示增强的量子意识洞察力...")
    
    try:
        symbiosis_core = symbiosis_system["symbiosis_core"]
        consciousness = symbiosis_system["consciousness"]
        
        # 获取量子意识洞察
        insights = consciousness.get_recent_insights(5)
        
        if insights:
            logger.info("量子意识洞察:")
            for i, insight in enumerate(insights):
                logger.info(f"  洞察 {i+1}: {insight}")
        else:
            logger.info("未生成量子意识洞察")
            
        # 获取意识引导
        guidance = consciousness.generate_market_guidance()
        
        if guidance:
            logger.info("市场引导建议:")
            for item in guidance:
                logger.info(f"  * {item}")
                
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"演示量子意识洞察力时出错: {str(e)}")

def shutdown_symbiosis_system(symbiosis_system):
    """关闭共生系统
    
    Args:
        symbiosis_system: 共生系统组件
    """
    logger.info("正在关闭超神量子共生系统...")
    
    try:
        # 关闭共生核心
        if "symbiosis_core" in symbiosis_system:
            symbiosis_system["symbiosis_core"].stop()
            
        # 关闭各模块
        if "cosmic_engine" in symbiosis_system:
            symbiosis_system["cosmic_engine"].stop()
            
        if "consciousness" in symbiosis_system:
            symbiosis_system["consciousness"].stop()
            
        logger.info("超神量子共生系统已关闭")
        
    except Exception as e:
        logger.error(f"关闭共生系统时出错: {str(e)}")

def run_demo():
    """运行完整演示"""
    try:
        logger.info("=" * 50)
        logger.info("  超神量子共生网络交易系统 - 共生演示")
        logger.info("=" * 50)
        
        # 设置并启动共生系统
        symbiosis_system = setup_symbiosis_system()
        
        # 监控共生系统进化（30秒）
        monitor_symbiosis_evolution(symbiosis_system, duration=30)
        
        # 演示增强的预测能力
        demonstrate_enhanced_prediction(symbiosis_system)
        
        # 演示增强的宇宙感知能力
        demonstrate_cosmic_perception(symbiosis_system)
        
        # 演示增强的量子意识洞察力
        demonstrate_consciousness_insights(symbiosis_system)
        
        # 关闭共生系统
        shutdown_symbiosis_system(symbiosis_system)
        
        logger.info("演示完成！")
        
    except KeyboardInterrupt:
        logger.info("用户中断演示")
    except Exception as e:
        logger.error(f"演示过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_demo() 