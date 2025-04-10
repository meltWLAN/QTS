#!/usr/bin/env python3
"""
超神量子核心组件验证脚本
验证量子核心组件是否正常工作
"""

import os
import sys
import logging
import importlib
import time
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumCoreValidator")

def check_directory_structure():
    """检查目录结构"""
    logger.info("检查量子核心目录结构...")
    
    # 检查主目录
    quantum_core_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantum_core")
    if not os.path.exists(quantum_core_dir):
        logger.error(f"量子核心主目录不存在: {quantum_core_dir}")
        return False
    
    # 检查子目录
    required_dirs = [
        os.path.join(quantum_core_dir, "visualization"),
        os.path.join(quantum_core_dir, "voice_control")
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.error(f"缺少目录: {directory}")
            return False
    
    logger.info("目录结构检查通过")
    return True

def validate_core_imports():
    """验证核心组件导入"""
    logger.info("验证核心组件导入...")
    
    core_modules = [
        "quantum_core.event_system",
        "quantum_core.data_pipeline",
        "quantum_core.component_system",
        "quantum_core.quantum_backend",
        "quantum_core.market_to_quantum",
        "quantum_core.quantum_interpreter",
        "quantum_core.multidimensional_analysis",
        "quantum_core.dimension_integrator",
        "quantum_core.genetic_strategy_optimizer",
        "quantum_core.strategy_evaluator",
        "quantum_core.adaptive_parameters",
        "quantum_core.event_driven_coordinator",
        "quantum_core.fault_tolerance",
        "quantum_core.system_monitor"
    ]
    
    success_count = 0
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"导入成功: {module_name}")
            success_count += 1
        except ImportError as e:
            logger.error(f"导入失败: {module_name} - {str(e)}")
    
    if success_count == len(core_modules):
        logger.info("所有核心组件导入成功")
        return True
    else:
        logger.warning(f"部分组件导入失败: {success_count}/{len(core_modules)}成功")
        return False

def validate_visualization_imports():
    """验证可视化组件导入"""
    logger.info("验证可视化组件导入...")
    
    viz_modules = [
        "quantum_core.visualization.dashboard_3d",
        "quantum_core.visualization.pattern_visualizer"
    ]
    
    success_count = 0
    for module_name in viz_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"导入成功: {module_name}")
            success_count += 1
        except ImportError as e:
            logger.error(f"导入失败: {module_name} - {str(e)}")
    
    if success_count == len(viz_modules):
        logger.info("所有可视化组件导入成功")
        return True
    else:
        logger.warning(f"部分组件导入失败: {success_count}/{len(viz_modules)}成功")
        return False

def validate_voice_imports():
    """验证语音控制组件导入"""
    logger.info("验证语音控制组件导入...")
    
    voice_modules = [
        "quantum_core.voice_control.voice_commander"
    ]
    
    success_count = 0
    for module_name in voice_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"导入成功: {module_name}")
            success_count += 1
        except ImportError as e:
            logger.error(f"导入失败: {module_name} - {str(e)}")
    
    if success_count == len(voice_modules):
        logger.info("所有语音控制组件导入成功")
        return True
    else:
        logger.warning(f"部分组件导入失败: {success_count}/{len(voice_modules)}成功")
        return False

def test_event_system():
    """测试事件系统"""
    logger.info("测试事件系统...")
    try:
        from quantum_core.event_system import QuantumEventSystem
        
        # 创建事件系统
        event_system = QuantumEventSystem()
        
        # 设置测试事件处理程序
        events_received = []
        
        async def test_handler(event_data):
            events_received.append(event_data)
            logger.info(f"收到事件: {event_data}")
        
        # 注册处理程序
        event_system.subscribe("test_event", test_handler)
        
        # 启动事件系统
        event_system.start()
        
        # 发送测试事件
        event_system.emit_event("test_event", {"test_data": "Hello Quantum World!"})
        
        # 等待事件处理
        time.sleep(1)
        
        # 停止事件系统
        event_system.stop()
        
        # 验证结果
        if len(events_received) > 0:
            logger.info("事件系统测试通过")
            return True
        else:
            logger.error("事件系统测试失败: 未收到事件")
            return False
    except Exception as e:
        logger.error(f"事件系统测试出错: {str(e)}")
        return False

def test_runtime_environment():
    """测试运行时环境"""
    logger.info("测试运行时环境...")
    try:
        from quantum_core.event_driven_coordinator import RuntimeEnvironment
        
        # 创建运行时环境
        runtime = RuntimeEnvironment()
        
        # 创建测试组件
        class TestComponent:
            def __init__(self):
                self.started = False
                
            def start(self):
                self.started = True
                logger.info("测试组件已启动")
                
            def stop(self):
                self.started = False
                logger.info("测试组件已停止")
        
        # 注册组件
        test_component = TestComponent()
        runtime.register_component("test_component", test_component)
        
        # 启动运行时环境
        runtime.start()
        
        # 验证组件已启动
        if test_component.started:
            logger.info("运行时环境启动组件成功")
        else:
            logger.error("运行时环境启动组件失败")
            return False
        
        # 获取组件状态
        state = runtime.get_component_state("test_component")
        logger.info(f"组件状态: {state}")
        
        # 停止运行时环境
        runtime.stop()
        
        # 验证组件已停止
        if not test_component.started:
            logger.info("运行时环境停止组件成功")
        else:
            logger.error("运行时环境停止组件失败")
            return False
        
        logger.info("运行时环境测试通过")
        return True
    except Exception as e:
        logger.error(f"运行时环境测试出错: {str(e)}")
        return False

def test_fault_tolerance():
    """测试容错机制"""
    logger.info("测试容错机制...")
    try:
        from quantum_core.fault_tolerance import CircuitBreaker, GracefulDegradation
        
        # 测试断路器
        circuit_breaker = CircuitBreaker("test_circuit")
        
        # 模拟成功函数
        def success_func():
            return "成功"
        
        # 模拟失败函数
        def fail_func():
            raise Exception("模拟失败")
        
        # 测试成功调用
        try:
            result = circuit_breaker.execute(success_func)
            logger.info(f"断路器成功调用: {result}")
        except Exception as e:
            logger.error(f"断路器成功调用失败: {str(e)}")
            return False
        
        # 测试失败调用
        failure_count = 0
        for i in range(10):
            try:
                circuit_breaker.execute(fail_func)
            except Exception:
                failure_count += 1
        
        logger.info(f"断路器失败调用记录: {failure_count}/10")
        
        # 测试优雅降级
        degradation = GracefulDegradation()
        
        # 注册降级方案
        degradation.register_fallback(
            "test_service",
            fail_func,
            lambda: "降级成功"
        )
        
        # 测试降级
        try:
            result = degradation.execute("test_service")
            if result == "降级成功":
                logger.info("优雅降级测试通过")
            else:
                logger.error(f"优雅降级结果不符合预期: {result}")
                return False
        except Exception as e:
            logger.error(f"优雅降级测试失败: {str(e)}")
            return False
        
        logger.info("容错机制测试通过")
        return True
    except Exception as e:
        logger.error(f"容错机制测试出错: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("超神量子核心组件验证 - 开始")
    logger.info("=" * 50)
    
    # 执行验证步骤
    validations = [
        ("目录结构", check_directory_structure),
        ("核心组件导入", validate_core_imports),
        ("可视化组件导入", validate_visualization_imports),
        ("语音控制组件导入", validate_voice_imports),
        ("事件系统", test_event_system),
        ("运行时环境", test_runtime_environment),
        ("容错机制", test_fault_tolerance)
    ]
    
    results = {}
    all_passed = True
    
    for name, validation_func in validations:
        logger.info(f"\n{'-' * 30}\n验证 {name}\n{'-' * 30}")
        try:
            result = validation_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"验证 {name} 时出错: {str(e)}")
            results[name] = False
            all_passed = False
    
    # 显示结果摘要
    logger.info("\n" + "=" * 50)
    logger.info("验证结果摘要")
    logger.info("=" * 50)
    
    for name, result in results.items():
        status = "通过" if result else "失败"
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("\n所有验证通过，量子核心组件运行正常！")
        return 0
    else:
        logger.warning("\n部分验证失败，请检查日志了解详情。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 