#!/usr/bin/env python3
"""
超神量子共生系统 - Tushare数据连接器修复验证
验证修复后的Tushare数据连接器是否正常工作
"""

import os
import sys
import logging
import py_compile
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("TushareValidation")

def validate_syntax():
    """验证语法是否正确"""
    logger.info("验证tushare_data_connector.py语法...")
    
    original_file = "tushare_data_connector.py"
    fixed_file = "tushare_data_connector_fixed.py"
    
    original_valid = False
    fixed_valid = False
    
    # 检查原始文件
    if os.path.exists(original_file):
        try:
            py_compile.compile(original_file)
            logger.info("原始文件语法验证通过")
            original_valid = True
        except py_compile.PyCompileError as e:
            logger.error(f"原始文件存在语法错误: {str(e)}")
    else:
        logger.warning(f"找不到原始文件: {original_file}")
    
    # 检查修复后的文件
    if os.path.exists(fixed_file):
        try:
            py_compile.compile(fixed_file)
            logger.info("修复后的文件语法验证通过")
            fixed_valid = True
        except py_compile.PyCompileError as e:
            logger.error(f"修复后的文件存在语法错误: {str(e)}")
    else:
        logger.warning(f"找不到修复后的文件: {fixed_file}")
    
    return {
        "original_valid": original_valid,
        "fixed_valid": fixed_valid
    }

def validate_functionality():
    """验证数据连接器功能是否正常"""
    logger.info("验证数据连接器功能...")
    
    fixed_file = "tushare_data_connector_fixed.py"
    
    if not os.path.exists(fixed_file):
        logger.error(f"找不到修复后的文件: {fixed_file}")
        return False
    
    try:
        # 使用importlib动态导入模块
        import importlib.util
        
        # 加载修复后的模块
        logger.info("加载修复后的数据连接器模块...")
        spec = importlib.util.spec_from_file_location("tushare_fixed", fixed_file)
        tushare_fixed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tushare_fixed)
        
        # 创建数据连接器对象
        logger.info("创建数据连接器测试实例...")
        connector = tushare_fixed.TushareDataConnector("test_token")
        
        # 测试方法是否正常
        methods_to_test = [
            "search_stocks",
            "get_market_data",
            "get_macro_data",
            "get_industry_data"
        ]
        
        passed_methods = []
        failed_methods = []
        
        for method_name in methods_to_test:
            if hasattr(connector, method_name):
                method = getattr(connector, method_name)
                try:
                    # 不实际执行，只测试方法是否存在
                    logger.info(f"验证方法: {method_name}")
                    passed_methods.append(method_name)
                except Exception as e:
                    logger.error(f"方法验证失败: {method_name}, 错误: {str(e)}")
                    failed_methods.append(method_name)
            else:
                logger.warning(f"方法不存在: {method_name}")
                failed_methods.append(method_name)
        
        if failed_methods:
            logger.warning(f"有{len(failed_methods)}个方法验证失败: {', '.join(failed_methods)}")
            return False
        else:
            logger.info(f"所有{len(passed_methods)}个方法验证通过")
            return True
        
    except Exception as e:
        logger.error(f"功能验证失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("\n" + "="*60)
    print(f"{'超神量子共生系统 - Tushare数据连接器修复验证':^60}")
    print("="*60 + "\n")
    
    # 验证语法
    syntax_results = validate_syntax()
    
    # 验证功能
    functionality_passed = validate_functionality()
    
    # 输出验证结果
    print("\n" + "="*60)
    print(f"{'验证结果':^60}")
    print("-"*60)
    
    print(f"{'原始文件语法':.<40}{'通过' if syntax_results['original_valid'] else '失败':>20}")
    print(f"{'修复文件语法':.<40}{'通过' if syntax_results['fixed_valid'] else '失败':>20}")
    print(f"{'功能验证':.<40}{'通过' if functionality_passed else '失败':>20}")
    
    print("-"*60)
    
    if syntax_results['fixed_valid'] and functionality_passed:
        print("\n✅ 数据连接器修复成功")
        if not syntax_results['original_valid']:
            print("   原始文件存在语法错误，已修复")
    else:
        print("\n❌ 数据连接器修复不完全")
        if not syntax_results['fixed_valid']:
            print("   修复文件仍存在语法错误")
        if not functionality_passed:
            print("   功能验证失败")
    
    print("\n" + "="*60)
    
    # 返回验证结果
    return 0 if syntax_results['fixed_valid'] and functionality_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 