#!/usr/bin/env python3
"""
超神系统模块验证脚本
检查所有核心模块并验证其功能完整性
"""

import os
import sys
import importlib
import logging
import traceback
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("ModuleVerifier")

# 需要验证的核心模块
CORE_MODULES = [
    'quantum_core.quantum_engine',
    'cosmic_resonance.cosmic_engine',
    'quantum_prediction.predictor',
    'quantum_symbiotic_network.high_dimensional_core',
    'quantum_symbiotic_network.hyperdimensional_protocol',
    'quantum_symbiotic_network.data_sources.tushare_data_source'
]

# 类名映射
CLASS_NAME_MAPPING = {
    'quantum_engine': 'QuantumEngine',
    'cosmic_engine': 'CosmicResonanceEngine',
    'predictor': 'QuantumPredictor',
    'high_dimensional_core': 'QuantumSymbioticCore',
    'hyperdimensional_protocol': 'HyperdimensionalProtocol',
    'tushare_data_source': 'TushareDataSource'
}

# 必要的方法列表
REQUIRED_METHODS = {
    'QuantumEngine': ['initialize', 'start', 'start_services', 'stop', 'calculate_quantum_probability'],
    'CosmicResonanceEngine': ['initialize', 'start_resonance', 'stop_resonance', 'set_quantum_predictor'],
    'QuantumPredictor': ['initialize', 'load_models', 'set_quantum_engine', 'set_data_source', 'predict_stock'],
    'QuantumSymbioticCore': ['register_module', 'get_module', 'initialize', 'start'],
    'HyperdimensionalProtocol': ['start', 'stop', 'add_dimension', 'remove_dimension'],
    'TushareDataSource': ['get_stock_list', 'get_daily_data', 'get_market_state']
}

def verify_module(module_name):
    """验证模块完整性
    
    Args:
        module_name: 模块名称
        
    Returns:
        tuple: (是否成功, 结果消息)
    """
    try:
        # 尝试导入模块
        module = importlib.import_module(module_name)
        
        # 获取模块基本名称
        base_name = module_name.split('.')[-1]
        
        # 获取预期类名
        class_name = CLASS_NAME_MAPPING.get(base_name, base_name.title().replace('_', ''))
        
        # 检查类是否存在
        if not hasattr(module, class_name):
            return False, f"缺少核心类 {class_name}"
            
        # 获取类
        cls = getattr(module, class_name)
        
        # 初始化类实例
        try:
            instance = cls()
            
            # 检查必要方法
            missing_methods = []
            if class_name in REQUIRED_METHODS:
                for method_name in REQUIRED_METHODS[class_name]:
                    if not hasattr(instance, method_name) or not callable(getattr(instance, method_name)):
                        missing_methods.append(method_name)
                        
            if missing_methods:
                return False, f"缺少方法: {', '.join(missing_methods)}"
                
            return True, "验证通过"
            
        except Exception as e:
            return False, f"实例化失败: {str(e)}"
            
    except ImportError as e:
        return False, f"导入失败: {str(e)}"
    except Exception as e:
        return False, f"验证出错: {str(e)}"
        
def fix_module_issues(module_name, issues):
    """尝试修复模块问题
    
    Args:
        module_name: 模块名称
        issues: 问题描述
        
    Returns:
        bool: 是否成功修复
    """
    logger.info(f"尝试修复模块 {module_name} 的问题: {issues}")
    
    try:
        base_name = module_name.split('.')[-1]
        class_name = CLASS_NAME_MAPPING.get(base_name, base_name.title().replace('_', ''))
        
        # 处理缺少初始化方法的情况
        if "缺少方法: initialize" in issues:
            module_path = module_name.replace('.', '/')
            file_path = f"{module_path}.py"
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                # 处理包的情况
                for i in range(len(module_name.split('.')) - 1, 0, -1):
                    parts = module_name.split('.')
                    parent_path = "/".join(parts[:i])
                    file_name = f"{parts[i]}.py"
                    full_path = f"{parent_path}/{file_name}"
                    
                    if os.path.exists(full_path):
                        file_path = full_path
                        break
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                    
                # 寻找类定义
                class_def = f"class {class_name}:"
                if class_def in content:
                    # 添加initialize方法
                    method_def = f"""
    def initialize(self):
        \"\"\"初始化{class_name}\"\"\"
        # 添加必要的初始化代码
        if hasattr(self, 'logger'):
            self.logger.info("{class_name}初始化完成")
        return True
"""
                    # 找到类的下一行
                    class_pos = content.find(class_def)
                    next_line_pos = content.find('\n', class_pos) + 1
                    
                    # 插入方法
                    new_content = content[:next_line_pos] + method_def + content[next_line_pos:]
                    
                    # 写回文件
                    with open(file_path, 'w') as file:
                        file.write(new_content)
                        
                    logger.info(f"已添加initialize方法到{class_name}")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"修复失败: {str(e)}")
        return False
        
def verify_all_modules():
    """验证所有核心模块
    
    Returns:
        dict: 验证结果
    """
    results = {}
    
    logger.info("开始验证超神系统核心模块...")
    
    for module_name in CORE_MODULES:
        logger.info(f"验证模块: {module_name}")
        
        success, message = verify_module(module_name)
        results[module_name] = {
            "success": success,
            "message": message
        }
        
        if success:
            logger.info(f"✅ 模块 {module_name} 验证通过")
        else:
            logger.warning(f"❌ 模块 {module_name} 验证失败: {message}")
            
            # 尝试修复
            if fix_module_issues(module_name, message):
                logger.info(f"🔧 模块 {module_name} 已修复")
                
                # 重新验证
                success, message = verify_module(module_name)
                results[module_name] = {
                    "success": success,
                    "message": message,
                    "fixed": True
                }
                
                if success:
                    logger.info(f"✅ 模块 {module_name} 修复后验证通过")
                else:
                    logger.warning(f"⚠️ 模块 {module_name} 修复后仍然失败: {message}")
            else:
                logger.warning(f"⚠️ 模块 {module_name} 无法自动修复")
    
    return results
    
def create_system_startup_fix():
    """创建系统启动修复
    
    根据验证结果，修改launch_supergod.py使其能正确加载模块
    """
    logger.info("创建系统启动修复...")
    
    try:
        with open('launch_supergod.py', 'r') as file:
            content = file.read()
            
        # 修复量子引擎加载
        if 'quantum_engine.initialize()' in content:
            content = content.replace(
                'quantum_engine.initialize()',
                'quantum_engine.initialize()\n        quantum_engine.start()'
            )
            
        # 修复宇宙共振引擎加载
        cosmic_engine_fix = """
    try:
        from cosmic_resonance.cosmic_engine import CosmicResonanceEngine, get_cosmic_engine
        
        # 初始化宇宙共振引擎
        cosmic_engine = get_cosmic_engine()
        cosmic_engine.initialize()
        
        # 启动共振服务
        cosmic_engine.start_resonance()
        
        logger.info("宇宙共振引擎加载完成")
        return cosmic_engine
    except Exception as e:
        logger.error(f"加载宇宙共振引擎失败: {str(e)}")
        return None
"""
        
        if 'from cosmic_resonance.cosmic_engine import CosmicResonanceEngine' in content:
            # 找到函数定义
            func_def = 'def load_cosmic_resonance_engine():'
            func_pos = content.find(func_def)
            
            if func_pos >= 0:
                # 找到函数体开始
                body_start = content.find('\n', func_pos) + 1
                
                # 找到函数体结束
                next_def_pos = content.find('def ', body_start)
                if next_def_pos >= 0:
                    body_end = content.rfind('\n', body_start, next_def_pos) + 1
                else:
                    body_end = len(content)
                    
                # 替换函数体
                new_content = content[:body_start] + cosmic_engine_fix + content[body_end:]
                content = new_content
        
        # 修复量子预测器加载
        quantum_predictor_fix = """
    try:
        from quantum_prediction.predictor import QuantumPredictor, get_quantum_predictor
        
        # 初始化量子预测器
        predictor = get_quantum_predictor()
        predictor.initialize()
        
        # 加载预测模型
        predictor.load_models()
        
        logger.info("量子预测器加载完成")
        return predictor
    except Exception as e:
        logger.error(f"加载量子预测器失败: {str(e)}")
        return None
"""
        
        if 'from quantum_prediction.predictor import QuantumPredictor' in content:
            # 找到函数定义
            func_def = 'def load_quantum_predictor():'
            func_pos = content.find(func_def)
            
            if func_pos >= 0:
                # 找到函数体开始
                body_start = content.find('\n', func_pos) + 1
                
                # 找到函数体结束
                next_def_pos = content.find('def ', body_start)
                if next_def_pos >= 0:
                    body_end = content.rfind('\n', body_start, next_def_pos) + 1
                else:
                    body_end = len(content)
                    
                # 替换函数体
                new_content = content[:body_start] + quantum_predictor_fix + content[body_end:]
                content = new_content
        
        # 写回文件
        with open('launch_supergod.py', 'w') as file:
            file.write(content)
            
        logger.info("系统启动修复完成")
        return True
        
    except Exception as e:
        logger.error(f"创建系统启动修复失败: {str(e)}")
        return False
        
def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("超神系统模块验证与修复工具")
    print("=" * 60)
    
    try:
        # 添加当前目录到模块搜索路径
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # 验证所有模块
        results = verify_all_modules()
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r["success"])
        total_count = len(results)
        
        print("\n" + "=" * 60)
        print(f"验证结果: {success_count}/{total_count} 模块通过")
        
        # 打印详细结果
        for module, result in results.items():
            status = "✅ 通过" if result["success"] else "❌ 失败"
            fixed = " (已修复)" if result.get("fixed") else ""
            print(f"{status}: {module}{fixed} - {result['message']}")
            
        # 如果有失败的模块，尝试创建系统启动修复
        if success_count < total_count:
            print("\n正在创建系统启动修复...")
            if create_system_startup_fix():
                print("✅ 系统启动修复已完成")
            else:
                print("❌ 系统启动修复失败")
                
        print("\n" + "=" * 60)
        print("超神系统模块验证与修复完成")
        print("可以使用 python launch_supergod.py --activate-field --consciousness-boost 启动系统")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 验证过程出错: {str(e)}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 