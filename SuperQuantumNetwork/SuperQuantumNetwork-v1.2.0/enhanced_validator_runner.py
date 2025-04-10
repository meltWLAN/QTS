#!/usr/bin/env python3
"""
超神量子共生系统 - 全面验证测试启动器
运行所有验证工具并生成综合报告
"""

import os
import sys
import logging
import subprocess
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validator_runner.log", 'a')
    ]
)

logger = logging.getLogger("ValidatorRunner")

def run_all_validators():
    """运行所有验证器并整合结果"""
    print("\n" + "=" * 80)
    print(f"{'超神量子共生系统 - 全面验证测试':^80}")
    print("=" * 80 + "\n")
    
    logger.info("开始全面系统验证")
    
    all_passed = True
    validation_results = {}
    recommendations = []
    report_files = []
    
    # 运行高级验证工具
    try:
        logger.info("运行高级验证工具...")
        print("\n运行高级验证工具...")
        
        result = subprocess.run(['python', 'advanced_validator.py'], 
                            capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            logger.info("高级验证测试通过")
            validation_results["advanced_validator"] = "通过"
        else:
            logger.warning("高级验证测试失败")
            validation_results["advanced_validator"] = f"失败 (返回码: {result.returncode})"
            all_passed = False
        
        # 解析输出以获取最近报告文件
        for line in result.stdout.split('\n'):
            if line.startswith("完整报告已保存至:"):
                report_file = line.split(":")[1].strip()
                if os.path.exists(report_file):
                    report_files.append(("高级验证报告", report_file))
                    
                    # 从报告中提取建议
                    with open(report_file, 'r') as f:
                        content = f.read()
                        if "改进建议:" in content:
                            rec_section = content.split("改进建议:")[1]
                            if "=" * 10 in rec_section:
                                rec_section = rec_section.split("=" * 10)[0]
                            
                            for line in rec_section.strip().split('\n'):
                                if line.startswith("-"):
                                    recommendations.append("【高级验证】" + line)
    except Exception as e:
        logger.error(f"运行高级验证工具时出错: {str(e)}")
        validation_results["advanced_validator"] = f"错误: {str(e)}"
        all_passed = False
    
    # 运行增强验证工具
    try:
        logger.info("运行增强验证工具...")
        print("\n运行增强验证工具...")
        
        result = subprocess.run(['python', 'enhanced_validation.py'], 
                            capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            logger.info("增强验证测试通过")
            validation_results["enhanced_validator"] = "通过"
        else:
            logger.warning("增强验证测试失败")
            validation_results["enhanced_validator"] = f"失败 (返回码: {result.returncode})"
            all_passed = False
        
        # 解析输出以获取最近报告文件
        for line in result.stdout.split('\n'):
            if line.startswith("完整报告已保存至:"):
                report_file = line.split(":")[1].strip()
                if os.path.exists(report_file):
                    report_files.append(("增强验证报告", report_file))
                    
                    # 从报告中提取建议
                    with open(report_file, 'r') as f:
                        content = f.read()
                        if "改进建议:" in content:
                            rec_section = content.split("改进建议:")[1]
                            if "=" * 10 in rec_section:
                                rec_section = rec_section.split("=" * 10)[0]
                            
                            for line in rec_section.strip().split('\n'):
                                if line.startswith("-"):
                                    recommendations.append("【增强验证】" + line)
    except Exception as e:
        logger.error(f"运行增强验证工具时出错: {str(e)}")
        validation_results["enhanced_validator"] = f"错误: {str(e)}"
        all_passed = False
    
    # 运行一致性验证测试
    try:
        logger.info("进行跨组件一致性测试...")
        print("\n进行跨组件一致性测试...")
        
        # 测试统一入口和驾驶舱组件的兼容性
        try:
            # 检查关键文件是否存在
            required_files = [
                "run_supergod_unified.py",
                "supergod_cockpit.py",
                "tushare_data_connector.py",
                "quantum_dimension_expander.py"
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if not missing_files:
                logger.info("核心组件文件完整性测试通过")
                validation_results["core_files_integrity"] = "通过"
            else:
                logger.warning(f"核心组件文件不完整，缺少: {', '.join(missing_files)}")
                validation_results["core_files_integrity"] = f"不完整，缺少: {len(missing_files)}个文件"
                all_passed = False
                recommendations.append(f"【一致性测试】确保所有核心文件存在: {', '.join(missing_files)}")
                
            # 检查文件权限
            executable_files = ["run_supergod_unified.py"]
            for file in executable_files:
                if os.path.exists(file):
                    mode = oct(os.stat(file).st_mode)[-3:]
                    if mode[0] != '7' and mode[0] != '5':  # 检查执行权限
                        logger.warning(f"文件缺少执行权限: {file} ({mode})")
                        validation_results[f"file_permission_{file}"] = f"缺少执行权限 ({mode})"
                        recommendations.append(f"【一致性测试】为{file}添加执行权限: chmod +x {file}")
                        all_passed = False
        except Exception as e:
            logger.error(f"核心组件测试出错: {str(e)}")
            validation_results["core_components_test"] = f"错误: {str(e)}"
            all_passed = False
    except Exception as e:
        logger.error(f"跨组件一致性测试出错: {str(e)}")
        validation_results["consistency_test"] = f"错误: {str(e)}"
        all_passed = False
    
    # 生成综合报告
    generate_final_report(all_passed, validation_results, recommendations, report_files)
    
    return all_passed

def generate_final_report(all_passed, validation_results, recommendations, report_files):
    """生成综合验证报告"""
    logger.info("生成综合验证报告...")
    print("\n生成综合验证报告...")
    
    # 打印报告
    print("\n" + "=" * 80)
    print(f"{'超神量子共生系统 - 综合验证测试报告':^80}")
    print("=" * 80)
    
    print(f"\n验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证结果: {'全部通过' if all_passed else '部分失败'}")
    
    # 打印测试结果摘要
    print("\n验证工具运行结果:")
    print("-" * 80)
    print(f"{'验证工具':<40}{'结果':<40}")
    print("-" * 80)
    
    for key, value in validation_results.items():
        print(f"{key:<40}{str(value):<40}")
    
    # 打印建议
    if recommendations:
        print("\n综合建议:")
        unique_recommendations = set(recommendations)
        for recommendation in unique_recommendations:
            print(f"{recommendation}")
    
    # 打印报告文件
    if report_files:
        print("\n详细报告文件:")
        for name, file in report_files:
            print(f"- {name}: {file}")
    
    print("\n" + "=" * 80)
    
    # 保存报告到文件
    report_file = f"supergod_comprehensive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{'超神量子共生系统 - 综合验证测试报告':^80}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"验证结果: {'全部通过' if all_passed else '部分失败'}\n\n")
        
        f.write("验证工具运行结果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'验证工具':<40}{'结果':<40}\n")
        f.write("-" * 80 + "\n")
        
        for key, value in validation_results.items():
            f.write(f"{key:<40}{str(value):<40}\n")
        
        if recommendations:
            f.write("\n综合建议:\n")
            unique_recommendations = set(recommendations)
            for recommendation in unique_recommendations:
                f.write(f"{recommendation}\n")
        
        if report_files:
            f.write("\n详细报告文件:\n")
            for name, file in report_files:
                f.write(f"- {name}: {file}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"综合验证报告已保存至: {report_file}")
    print(f"\n综合报告已保存至: {report_file}")
    
    return report_file

def main():
    """主函数"""
    print("开始执行超神量子共生系统全面验证...")
    result = run_all_validators()
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 