#!/bin/bash
# 超神量子共生系统 - 依赖安装脚本

echo "开始安装超神量子共生系统依赖..."
echo "---------------------------------------"

# 检查Python环境
python_version=$(python3 --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "错误: 未检测到Python3，请先安装Python3.7或更高版本"
    exit 1
fi

echo "检测到Python版本: $python_version"

# 创建数据目录
mkdir -p data
echo "已创建数据目录"

# 安装依赖
echo "开始安装基础依赖..."
pip3 install pandas numpy matplotlib seaborn PyQt5 mplfinance

echo "开始安装Tushare数据接口..."
pip3 install tushare

echo "开始安装matplotlib 3D支持..."
pip3 install matplotlib[qt5] matplotlib[3d]

echo "依赖安装完成!"
echo "---------------------------------------"
echo "现在可以启动超神桌面系统了:"
echo "python supergod_cockpit.py"
echo "---------------------------------------" 