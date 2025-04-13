#!/usr/bin/env python3
"""
超神量子共生系统 - GitHub备份工具
用于将系统备份到GitHub仓库并创建回滚点
"""

import os
import sys
import subprocess
import logging
import datetime
import argparse
import json
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("BackupSystem")

# 系统核心组件列表
CORE_COMPONENTS = [
    "cosmic_consciousness.py",
    "sentiment_integration.py",
    "news_crawler.py",
    "system_integration.py",
    "validate_system.py"
]

# 备份配置文件名
BACKUP_CONFIG = "backup_config.json"

def run_command(command, cwd=None):
    """执行Shell命令并返回结果"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"错误: {e.stderr.strip()}"

def is_git_repo(directory):
    """检查目录是否为Git仓库"""
    git_dir = os.path.join(directory, ".git")
    return os.path.isdir(git_dir)

def init_git_repo(directory):
    """初始化Git仓库"""
    if is_git_repo(directory):
        logger.info("Git仓库已存在")
        return True
    
    logger.info("初始化Git仓库...")
    success, output = run_command("git init", cwd=directory)
    if success:
        logger.info("Git仓库初始化成功")
        return True
    else:
        logger.error(f"Git仓库初始化失败: {output}")
        return False

def create_gitignore(directory):
    """创建.gitignore文件"""
    gitignore_path = os.path.join(directory, ".gitignore")
    if os.path.exists(gitignore_path):
        logger.info(".gitignore文件已存在")
        return
    
    logger.info("创建.gitignore文件...")
    ignore_patterns = [
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        ".env",
        ".venv",
        "env/",
        "venv/",
        "ENV/",
        ".idea/",
        ".vscode/",
        "*.log",
        "*.swp",
        ".DS_Store"
    ]
    
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ignore_patterns))
    
    logger.info(".gitignore文件创建成功")

def get_system_version():
    """获取系统版本号"""
    # 首先尝试从VERSION文件获取
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    if os.path.exists(version_file):
        try:
            with open(version_file, "r") as f:
                version = f.read().strip()
                return version
        except Exception as e:
            logger.error(f"读取VERSION文件失败: {str(e)}")
    
    # 如果VERSION文件不存在或读取失败，生成时间戳版本号
    timestamp = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
    return f"v{timestamp}"

def commit_changes(directory, version):
    """提交变更"""
    logger.info(f"提交系统变更，版本: {version}...")
    
    # 添加所有文件
    success, output = run_command("git add .", cwd=directory)
    if not success:
        logger.error(f"添加文件失败: {output}")
        return False
    
    # 提交变更
    commit_message = f"超神量子共生系统 - 版本 {version}"
    success, output = run_command(f'git commit -m "{commit_message}"', cwd=directory)
    if success:
        logger.info(f"变更提交成功: {output}")
        return True
    else:
        if "nothing to commit" in output:
            logger.info("没有需要提交的变更")
            return True
        logger.error(f"提交变更失败: {output}")
        return False

def create_tag(directory, version, message):
    """创建标签作为回滚点"""
    logger.info(f"创建标签 {version} 作为回滚点...")
    
    success, output = run_command(f'git tag -a "{version}" -m "{message}"', cwd=directory)
    if success:
        logger.info(f"标签创建成功: {version}")
        return True
    else:
        logger.error(f"标签创建失败: {output}")
        return False

def create_rollback_branch(directory, version):
    """创建回滚分支"""
    branch_name = f"rollback-{version}"
    logger.info(f"创建回滚分支: {branch_name}...")
    
    success, output = run_command(f"git branch {branch_name}", cwd=directory)
    if success:
        logger.info(f"回滚分支创建成功: {branch_name}")
        return True
    else:
        logger.error(f"回滚分支创建失败: {output}")
        return False

def setup_github_remote(directory, repo_url):
    """设置GitHub远程仓库"""
    logger.info("检查远程仓库设置...")
    
    # 检查是否已设置远程仓库
    success, output = run_command("git remote -v", cwd=directory)
    if success and "origin" in output:
        logger.info("远程仓库已设置")
        return True
    
    if not repo_url:
        logger.error("未提供GitHub仓库URL，无法设置远程仓库")
        return False
    
    logger.info(f"设置GitHub远程仓库: {repo_url}")
    success, output = run_command(f"git remote add origin {repo_url}", cwd=directory)
    if success:
        logger.info("GitHub远程仓库设置成功")
        return True
    else:
        logger.error(f"设置GitHub远程仓库失败: {output}")
        return False

def push_to_github(directory, version):
    """推送到GitHub"""
    logger.info("推送到GitHub...")
    
    # 获取当前分支名
    success, output = run_command("git rev-parse --abbrev-ref HEAD", cwd=directory)
    if not success:
        logger.error(f"获取当前分支名失败: {output}")
        return False
    
    current_branch = output.strip()
    logger.info(f"当前分支: {current_branch}")
    
    # 推送当前分支
    success, output = run_command(f"git push -u origin {current_branch}", cwd=directory)
    if not success:
        logger.error(f"推送主分支失败: {output}")
        return False
    
    # 推送标签
    success, output = run_command("git push --tags", cwd=directory)
    if success:
        logger.info("推送标签成功")
        return True
    else:
        logger.error(f"推送标签失败: {output}")
        return False

def check_system_components(directory):
    """检查系统核心组件是否存在"""
    missing_components = []
    for component in CORE_COMPONENTS:
        path = os.path.join(directory, component)
        if not os.path.exists(path):
            missing_components.append(component)
    
    if missing_components:
        logger.warning(f"以下核心组件缺失: {', '.join(missing_components)}")
        return False
    
    logger.info("所有核心组件正常")
    return True

def load_backup_config(directory):
    """加载备份配置"""
    config_path = os.path.join(directory, BACKUP_CONFIG)
    if not os.path.exists(config_path):
        return {
            "last_backup": None,
            "versions": [],
            "github_repo": None
        }
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载备份配置失败: {str(e)}")
        return {
            "last_backup": None,
            "versions": [],
            "github_repo": None
        }

def save_backup_config(directory, config):
    """保存备份配置"""
    config_path = os.path.join(directory, BACKUP_CONFIG)
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"保存备份配置失败: {str(e)}")
        return False

def create_readme_if_not_exists(directory):
    """如果不存在README.md则创建"""
    readme_path = os.path.join(directory, "README.md")
    if os.path.exists(readme_path):
        return
    
    logger.info("创建README.md文件...")
    content = """# 超神量子共生系统

超神量子共生系统是一个基于高维量子算法的金融市场分析与预测系统，整合了宇宙意识、情感分析和新闻情绪感知功能。

## 核心组件

- **宇宙意识模块**: 感知市场能量场和宇宙共振状态
- **情绪集成模块**: 连接宇宙意识与量子预测引擎，融合情绪因子
- **新闻爬虫模块**: 获取财经新闻和社交媒体数据，为情绪分析提供数据源
- **系统集成模块**: 整合所有组件，提供统一接口和功能调用

## 系统功能

- 高维量子市场分析
- 股票价格趋势预测
- 市场情绪感知和分析
- 能量场探测与共振频率识别

*注意：本系统仅供研究和学习使用，不构成投资建议。*
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    logger.info("README.md文件创建成功")

def backup_system(directory, github_repo=None, skip_push=False):
    """备份系统到GitHub"""
    start_time = datetime.datetime.now()
    logger.info(f"开始备份超神量子共生系统: {directory}")
    
    # 检查系统组件
    if not check_system_components(directory):
        logger.warning("系统组件检查不完整，但将继续备份")
    
    # 创建README文件
    create_readme_if_not_exists(directory)
    
    # 加载配置
    config = load_backup_config(directory)
    
    # 更新GitHub仓库URL
    if github_repo:
        config["github_repo"] = github_repo
    
    # 初始化Git仓库
    if not init_git_repo(directory):
        return False
    
    # 创建.gitignore文件
    create_gitignore(directory)
    
    # 获取版本号
    version = get_system_version()
    logger.info(f"生成版本号: {version}")
    
    # 提交变更
    if not commit_changes(directory, version):
        logger.error("提交变更失败，备份终止")
        return False
    
    # 创建标签作为回滚点
    tag_message = f"超神量子共生系统 - 版本 {version} - 回滚点"
    if not create_tag(directory, version, tag_message):
        logger.error("创建标签失败")
        return False
    
    # 创建回滚分支
    if not create_rollback_branch(directory, version):
        logger.warning("创建回滚分支失败，但将继续备份")
    
    # 设置GitHub远程仓库（如果提供了URL）
    if config["github_repo"] and not skip_push:
        if not setup_github_remote(directory, config["github_repo"]):
            logger.error("设置GitHub远程仓库失败，无法推送")
            skip_push = True
    else:
        skip_push = True
    
    # 推送到GitHub（如果不跳过）
    if not skip_push:
        if not push_to_github(directory, version):
            logger.error("推送到GitHub失败")
            return False
    else:
        logger.info("跳过推送到GitHub")
    
    # 更新备份配置
    config["last_backup"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if version not in config["versions"]:
        config["versions"].append(version)
    
    # 保存备份配置
    save_backup_config(directory, config)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"系统备份完成，用时 {duration:.2f} 秒")
    
    return version

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超神量子共生系统备份工具")
    parser.add_argument("--directory", "-d", default=".", help="系统目录路径")
    parser.add_argument("--github", "-g", help="GitHub仓库URL")
    parser.add_argument("--skip-push", "-s", action="store_true", help="跳过推送到GitHub")
    args = parser.parse_args()
    
    # 获取绝对路径
    directory = os.path.abspath(args.directory)
    
    # 执行备份
    version = backup_system(
        directory=directory,
        github_repo=args.github,
        skip_push=args.skip_push
    )
    
    if version:
        print("\n" + "="*60)
        print(f"✅ 系统备份成功!")
        print(f"   版本: {version}")
        if not args.skip_push and args.github:
            print(f"   已推送到GitHub: {args.github}")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("❌ 系统备份失败，请检查日志")
        print("="*60 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main() 