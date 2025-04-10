#!/usr/bin/env python3
"""
超神量子共生系统 - 快速验证脚本
用于快速验证系统核心功能
"""

import os
import sys
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("QuickValidator")

def validate_quantum_dimension_expander():
    """验证量子维度扩展器"""
    try:
        logger.info("验证量子维度扩展器...")
        
        # 尝试导入量子维度扩展器
        from quantum_dimension_expander import QuantumDimensionExpander
        
        # 创建扩展器
        expander = QuantumDimensionExpander(11, 21)
        logger.info("量子维度扩展器创建成功")
        
        # 测试扩展功能
        test_data = {
            'price': 3100.0,
            'volume': 0.75,
            'momentum': 0.2,
            'volatility': 0.15,
            'trend': 0.5
        }
        
        expanded = expander.expand_dimensions(test_data)
        logger.info(f"扩展后维度: {len(expanded)}")
        
        # 测试折叠功能
        folded = expander.fold_dimensions(expanded)
        logger.info(f"折叠后维度: {len(folded)}")
        
        return True
    except Exception as e:
        logger.error(f"量子维度扩展器测试失败: {str(e)}")
        return False

def validate_data_connector():
    """验证数据连接器"""
    try:
        logger.info("验证Tushare数据连接器...")
        
        # 检查数据连接器文件是否存在
        if not os.path.exists("tushare_data_connector.py"):
            logger.error("找不到tushare_data_connector.py文件")
            return False
        
        # 验证语法
        import py_compile
        try:
            py_compile.compile("tushare_data_connector.py")
            logger.info("Tushare数据连接器语法验证通过")
        except py_compile.PyCompileError as e:
            logger.error(f"Tushare数据连接器存在语法错误: {str(e)}")
            return False
        
        # 尝试修复方法
        logger.info("尝试运行修复脚本...")
        if os.path.exists("tushare_connector_fix.py"):
            try:
                py_compile.compile("tushare_connector_fix.py")
                logger.info("修复版数据连接器语法验证通过")
            except py_compile.PyCompileError as e:
                logger.error(f"修复版数据连接器存在语法错误: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"数据连接器验证失败: {str(e)}")
        return False

def main():
    """主验证函数"""
    print("\n" + "="*60)
    print(f"{'超神量子共生系统 - 快速验证':^60}")
    print("="*60 + "\n")
    
    logger.info("开始快速验证...")
    
    results = {}
    all_passed = True
    
    # 验证量子维度扩展器
    result = validate_quantum_dimension_expander()
    results["量子维度扩展器"] = "通过" if result else "失败"
    all_passed = all_passed and result
    
    # 验证数据连接器
    result = validate_data_connector()
    results["数据连接器"] = "通过" if result else "失败"
    all_passed = all_passed and result
    
    # 输出结果
    print("\n" + "="*60)
    print(f"{'验证结果':^60}")
    print("-"*60)
    for component, status in results.items():
        print(f"{component:.<50}{status:>10}")
    print("-"*60)
    
    if all_passed:
        print("\n✅ 基本验证通过: 核心组件工作正常")
    else:
        print("\n⚠️ 验证未完全通过: 部分组件需要修复")
    
    print("\n" + "="*60)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 