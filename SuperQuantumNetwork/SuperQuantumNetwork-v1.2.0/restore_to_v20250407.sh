#!/bin/bash
# 超神量子共生系统 - 版本回滚脚本
# 将系统回滚到v20250407.1830版本

echo "开始回滚到超神量子共生系统 v20250407.1830 版本..."
echo "----------------------------------------------------"

# 确保我们有最新的远程变更
echo "1. 更新远程仓库信息..."
git fetch --all
echo "   完成"

# 创建临时分支以防万一
current_branch=$(git rev-parse --abbrev-ref HEAD)
backup_branch="backup_before_rollback_$(date +%Y%m%d_%H%M%S)"
echo "2. 创建备份分支: $backup_branch"
git branch $backup_branch
echo "   完成 - 原始状态已备份到分支 '$backup_branch'"

# 检查恢复点分支是否存在
echo "3. 检查恢复点分支..."
if git show-ref --verify --quiet refs/remotes/origin/recovery-point-v1.0; then
    echo "   恢复点分支存在，切换到该分支"
    git checkout recovery-point-v1.0
    git pull origin recovery-point-v1.0
else
    echo "   恢复点分支不存在，直接使用标签"
    git checkout v20250407.1830
fi

echo "4. 回滚完成！当前系统已恢复到 v20250407.1830 版本"
echo ""
echo "如果需要恢复到回滚前的状态，请运行:"
echo "git checkout $current_branch"
echo ""
echo "或者使用备份分支:"
echo "git checkout $backup_branch"
echo "----------------------------------------------------" 