#!/bin/bash
# 超神量子共生系统回滚脚本
# 创建时间: 2025-04-07

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示banner
echo -e "${BLUE}"
echo "============================================================"
echo "       超神量子共生系统 - 版本回滚工具"
echo "============================================================"
echo -e "${NC}"

# 函数: 显示帮助信息
show_help() {
    echo -e "${YELLOW}用法:${NC} $0 [选项]"
    echo 
    echo "选项:"
    echo "  -p, --patched    恢复到修复版本 (v1.1-patched-20250407)"
    echo "  -s, --stable     恢复到稳定版本 (v20250407.2300)"
    echo "  -o, --original   恢复到原始版本 (recovery-point-v1.0)"
    echo "  -l, --list       列出所有可恢复的版本"
    echo "  -h, --help       显示此帮助信息"
    echo
    echo "例子:"
    echo "  $0 --patched     # 恢复到修复版本"
    echo "  $0 --list        # 列出所有版本"
    echo
}

# 函数: 列出所有可恢复的版本
list_versions() {
    echo -e "${GREEN}可恢复的版本:${NC}"
    echo "1. 修复版本 (v1.1-patched-20250407) - 包含信号槽修复和数据兼容性优化"
    echo "2. 稳定版本 (v20250407.2300) - 修复前的稳定版本"
    echo "3. 原始版本 (recovery-point-v1.0) - 初始版本"
    echo
}

# 函数: 创建备份分支
create_backup() {
    local backup_branch="backup-$(date +%Y%m%d%H%M%S)"
    echo -e "${YELLOW}创建当前状态的备份分支: $backup_branch${NC}"
    git checkout -b $backup_branch
    git checkout -
    echo -e "${GREEN}已创建备份分支 $backup_branch${NC}"
}

# 函数: 恢复到指定版本
restore_version() {
    local version=$1
    local version_name=$2
    
    # 首先检查分支是否存在
    if ! git rev-parse --verify $version >/dev/null 2>&1; then
        echo -e "${RED}错误: $version_name ($version) 不存在!${NC}"
        exit 1
    fi
    
    # 创建备份
    create_backup
    
    echo -e "${YELLOW}正在恢复到 $version_name ($version)...${NC}"
    git checkout $version
    
    echo -e "${GREEN}成功恢复到 $version_name!${NC}"
    echo "当前状态:"
    git status
}

# 如果没有参数，显示帮助
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# 解析参数
case "$1" in
    -p|--patched)
        restore_version "v1.1-patched-20250407" "修复版本"
        ;;
    -s|--stable)
        restore_version "v20250407.2300" "稳定版本"
        ;;
    -o|--original)
        restore_version "recovery-point-v1.0" "原始版本"
        ;;
    -l|--list)
        list_versions
        ;;
    -h|--help)
        show_help
        ;;
    *)
        echo -e "${RED}错误: 未知选项 $1${NC}"
        show_help
        exit 1
        ;;
esac

exit 0 