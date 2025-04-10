#!/usr/bin/env python3
"""
超神量子共生系统 - 综合修复验证
验证所有修复是否成功应用并正常工作
"""

import os
import sys
import logging
import py_compile
import importlib.util
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("FixValidator")

def validate_tushare_fix():
    """验证Tushare数据连接器修复"""
    logger.info("验证Tushare数据连接器修复...")
    
    # 文件路径
    original_file = "tushare_data_connector.py"
    fixed_file = "tushare_data_connector_fixed.py"
    
    # 验证结果
    results = {
        "original_exists": os.path.exists(original_file),
        "fixed_exists": os.path.exists(fixed_file),
        "original_valid": False,
        "fixed_valid": False
    }
    
    # 检查原始文件语法
    if results["original_exists"]:
        try:
            py_compile.compile(original_file)
            results["original_valid"] = True
            logger.info("原始Tushare连接器语法有效")
        except py_compile.PyCompileError as e:
            logger.error(f"原始Tushare连接器存在语法错误: {str(e)}")
    
    # 检查修复后的文件语法
    if results["fixed_exists"]:
        try:
            py_compile.compile(fixed_file)
            results["fixed_valid"] = True
            logger.info("修复后的Tushare连接器语法有效")
            
            # 测试功能
            try:
                # 加载修复后的模块
                spec = importlib.util.spec_from_file_location("tushare_fixed", fixed_file)
                tushare_fixed = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tushare_fixed)
                
                # 验证类是否存在
                if hasattr(tushare_fixed, "TushareDataConnector"):
                    logger.info("TushareDataConnector类加载成功")
                    results["class_valid"] = True
                else:
                    logger.error("找不到TushareDataConnector类")
                    results["class_valid"] = False
            except Exception as e:
                logger.error(f"加载修复后的模块失败: {str(e)}")
                results["class_valid"] = False
        except py_compile.PyCompileError as e:
            logger.error(f"修复后的Tushare连接器仍存在语法错误: {str(e)}")
    
    # 标记是否修复成功
    results["fix_successful"] = results.get("fixed_valid", False) and results.get("class_valid", False)
    
    return results

def validate_cockpit_fix():
    """验证驾驶舱修复"""
    logger.info("验证驾驶舱修复...")
    
    # 文件路径
    original_file = "SuperQuantumNetwork/supergod_cockpit.py"
    fixed_file = "SuperQuantumNetwork/supergod_cockpit_fixed.py"
    patch_file = "supergod_cockpit_patch.py"
    
    # 验证结果
    results = {
        "original_exists": os.path.exists(original_file),
        "fixed_exists": os.path.exists(fixed_file),
        "patch_exists": os.path.exists(patch_file),
        "original_valid": False,
        "fixed_valid": False,
        "patch_valid": False
    }
    
    # 检查原始文件语法
    if results["original_exists"]:
        try:
            py_compile.compile(original_file)
            results["original_valid"] = True
            logger.info("原始驾驶舱文件语法有效")
        except py_compile.PyCompileError as e:
            logger.error(f"原始驾驶舱文件存在语法错误: {str(e)}")
    
    # 检查修复后的文件语法
    if results["fixed_exists"]:
        try:
            py_compile.compile(fixed_file)
            results["fixed_valid"] = True
            logger.info("修复后的驾驶舱文件语法有效")
        except py_compile.PyCompileError as e:
            logger.error(f"修复后的驾驶舱文件仍存在语法错误: {str(e)}")
    
    # 检查补丁文件语法
    if results["patch_exists"]:
        try:
            py_compile.compile(patch_file)
            results["patch_valid"] = True
            logger.info("补丁文件语法有效")
        except py_compile.PyCompileError as e:
            logger.error(f"补丁文件存在语法错误: {str(e)}")
    
    # 如果修复文件不存在但补丁存在，尝试应用补丁
    if not results["fixed_exists"] and results["patch_exists"] and results["patch_valid"]:
        try:
            logger.info("尝试应用补丁...")
            
            # 导入补丁模块
            spec = importlib.util.spec_from_file_location("cockpit_patch", patch_file)
            patch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_module)
            
            # 运行补丁
            if hasattr(patch_module, "fix_cockpit_file"):
                patch_result = patch_module.fix_cockpit_file()
                results["patch_applied"] = patch_result
                
                if patch_result:
                    logger.info("补丁应用成功")
                    # 再次检查修复后的文件
                    if os.path.exists(fixed_file):
                        try:
                            py_compile.compile(fixed_file)
                            results["fixed_valid"] = True
                            logger.info("应用补丁后，驾驶舱文件语法有效")
                        except py_compile.PyCompileError as e:
                            logger.error(f"应用补丁后，驾驶舱文件仍存在语法错误: {str(e)}")
                else:
                    logger.error("补丁应用失败")
            else:
                logger.error("补丁模块中找不到fix_cockpit_file函数")
                results["patch_applied"] = False
        except Exception as e:
            logger.error(f"应用补丁时出错: {str(e)}")
            results["patch_applied"] = False
    
    # 标记是否修复成功
    results["fix_successful"] = results["fixed_valid"] or results.get("patch_applied", False)
    
    return results

def validate_quantum_dimension_expander():
    """验证量子维度扩展器"""
    logger.info("验证量子维度扩展器...")
    
    file_path = "SuperQuantumNetwork/quantum_dimension_expander.py"
    
    # 验证结果
    results = {
        "exists": os.path.exists(file_path),
        "valid": False,
        "functional": False
    }
    
    if results["exists"]:
        try:
            py_compile.compile(file_path)
            results["valid"] = True
            logger.info("量子维度扩展器语法有效")
            
            # 测试功能
            try:
                import subprocess
                process = subprocess.run(
                    ["python", file_path, "--validate"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if process.returncode == 0:
                    results["functional"] = True
                    logger.info("量子维度扩展器功能正常")
                else:
                    logger.error(f"量子维度扩展器功能测试失败: {process.stderr}")
            except Exception as e:
                logger.error(f"量子维度扩展器功能测试出错: {str(e)}")
        except py_compile.PyCompileError as e:
            logger.error(f"量子维度扩展器存在语法错误: {str(e)}")
    else:
        logger.error(f"找不到量子维度扩展器文件: {file_path}")
    
    return results

def check_system_integrity():
    """检查系统完整性"""
    logger.info("检查系统完整性...")
    
    # 关键文件列表
    key_files = [
        "SuperQuantumNetwork/quantum_dimension_expander.py",
        "tushare_data_connector.py",
        "tushare_data_connector_fixed.py",
        "SuperQuantumNetwork/supergod_cockpit.py",
        "validate_all_fixes.py"
    ]
    
    # 验证结果
    results = {
        "files_checked": 0,
        "files_missing": 0,
        "missing_files": []
    }
    
    for file_path in key_files:
        results["files_checked"] += 1
        if not os.path.exists(file_path):
            results["files_missing"] += 1
            results["missing_files"].append(file_path)
            logger.warning(f"找不到文件: {file_path}")
    
    if results["files_missing"] == 0:
        logger.info("所有关键文件都存在")
    else:
        logger.warning(f"缺少 {results['files_missing']} 个关键文件")
    
    return results

def main():
    """主函数"""
    print("\n" + "="*70)
    print(f"{'超神量子共生系统 - 综合修复验证':^70}")
    print("="*70 + "\n")
    
    logger.info("开始综合验证...")
    
    # 检查系统完整性
    integrity_results = check_system_integrity()
    
    # 验证各个组件修复
    tushare_results = validate_tushare_fix()
    cockpit_results = validate_cockpit_fix()
    quantum_results = validate_quantum_dimension_expander()
    
    # 生成综合报告
    tushare_fixed = tushare_results.get("fix_successful", False)
    cockpit_fixed = cockpit_results.get("fix_successful", False)
    quantum_ok = quantum_results.get("functional", False)
    
    all_fixed = tushare_fixed and cockpit_fixed and quantum_ok
    
    print("\n" + "="*70)
    print(f"{'验证结果':^70}")
    print("-"*70)
    
    print(f"{'系统完整性检查':.<50}{'通过' if integrity_results['files_missing'] == 0 else '失败':>20}")
    print(f"{'Tushare数据连接器修复':.<50}{'通过' if tushare_fixed else '失败':>20}")
    print(f"{'驾驶舱修复':.<50}{'通过' if cockpit_fixed else '失败':>20}")
    print(f"{'量子维度扩展器':.<50}{'通过' if quantum_ok else '失败':>20}")
    
    print("-"*70)
    
    if all_fixed:
        print("\n✅ 所有组件修复成功")
        print("\n系统可以正常运行")
    else:
        print("\n⚠️ 部分组件修复不完整")
        if not tushare_fixed:
            print("   - Tushare数据连接器仍需修复")
        if not cockpit_fixed:
            print("   - 驾驶舱仍需修复")
        if not quantum_ok:
            print("   - 量子维度扩展器需要检查")
    
    print("\n" + "="*70)
    
    return 0 if all_fixed else 1

if __name__ == "__main__":
    sys.exit(main()) 