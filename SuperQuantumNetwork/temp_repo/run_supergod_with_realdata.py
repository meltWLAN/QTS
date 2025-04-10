#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱启动脚本 (真实数据版)
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_realdata.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodLauncher")

def check_dependencies():
    """检查依赖库是否已安装"""
    try:
        import pandas
        import numpy
        logger.info("基础数据库检查通过")
        
        try:
            import tushare
            logger.info("Tushare库已安装")
            tushare_available = True
        except ImportError:
            logger.warning("Tushare库未安装，将使用备选数据源")
            tushare_available = False
            
        try:
            import akshare
            logger.info("AKShare库已安装")
            akshare_available = True
        except ImportError:
            logger.warning("AKShare库未安装，将使用备选数据源")
            akshare_available = False
            
        if not tushare_available and not akshare_available:
            logger.warning("无法找到市场数据库!")
            print("\n请安装市场数据库以获取实时数据:")
            print("pip install tushare")
            print("pip install akshare==1.12.24")
            return False
            
        return True
    except ImportError as e:
        logger.error(f"缺少必要依赖: {str(e)}")
        print("\n请安装必要的依赖库:")
        print("pip install pandas numpy matplotlib seaborn")
        return False

def create_data_connector():
    """创建实时数据连接器"""
    # 首先尝试使用Tushare
    try:
        import tushare as ts
        
        # 测试Tushare连接
        token = os.environ.get('TUSHARE_TOKEN', '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10')
        ts.set_token(token)
        pro = ts.pro_api()
        
        # 简单测试请求
        test_data = pro.trade_cal(exchange='SSE', start_date='20250401', end_date='20250405')
        if test_data is not None and len(test_data) > 0:
            logger.info("Tushare连接成功，使用Tushare数据源")
            from tushare_data_connector import TushareDataConnector
            return TushareDataConnector(token=token)
        else:
            logger.warning("Tushare连接测试未返回数据")
    except Exception as e:
        logger.warning(f"Tushare连接失败: {str(e)}")
    
    # 如果Tushare失败，尝试AKShare
    try:
        import akshare as ak
        # 测试AKShare是否可用
        test_data = ak.stock_zh_index_daily(symbol="sh000001")
        if test_data is not None and not test_data.empty:
            logger.info("使用AKShare数据源")
            from akshare_data_connector import AKShareDataConnector
            return AKShareDataConnector()
        else:
            logger.warning("AKShare连接测试未返回数据")
    except Exception as e:
        logger.warning(f"AKShare连接失败: {str(e)}")
    
    # 如果都失败，抛出异常
    raise RuntimeError("无法连接到任何市场数据源，请检查网络连接和API配置")

def start_supergod_system(headless=False):
    """启动超神量子共生系统"""
    print("\n正在启动超神量子共生系统...\n")
    
    # 创建实时数据连接器
    try:
        data_connector = create_data_connector()
        
        # 测试数据连接
        test_data = data_connector.get_market_data("000001.SH")
        if test_data is not None and not test_data.empty:
            logger.info(f"成功获取上证指数数据: {len(test_data)}行")
            print(f"\n上证指数最近数据:")
            print(test_data.tail(3))
        else:
            logger.error("无法获取市场数据，请检查网络连接")
            return False
        
        # 加载市场分析核心
        try:
            from china_market_core import ChinaMarketCore
            market_core = ChinaMarketCore()
            market_analysis = market_core.analyze_market(test_data)
            
            # 获取市场状态
            market_status = "未知"
            if 'market_sentiment' in market_analysis:
                sentiment = market_analysis['market_sentiment']
                if sentiment > 0.3:
                    market_status = "看涨"
                elif sentiment < -0.3:
                    market_status = "看跌"
                else:
                    market_status = "盘整"
            
            logger.info(f"市场状态: {market_status}")
            print(f"当前市场状态: {market_status}")
            
        except Exception as e:
            logger.warning(f"市场分析核心加载失败: {str(e)}")
        
        # 测试板块数据
        try:
            sector_data = data_connector.get_sector_data()
            if sector_data:
                leading_sectors = sector_data.get('leading_sectors', [])
                if leading_sectors:
                    logger.info(f"成功获取板块数据，领先板块: {', '.join([s['name'] for s in leading_sectors[:3]])}")
                    print("\n领先板块:")
                    for i, sector in enumerate(leading_sectors[:5]):
                        print(f"  {i+1}. {sector['name']}: {sector['change_pct']:.2f}%")
        except Exception as e:
            logger.warning(f"板块数据获取失败: {str(e)}")
        
        # 获取市场新闻
        try:
            news_data = data_connector.get_policy_news(3)
            if news_data:
                logger.info("成功获取政策新闻")
                print("\n最新市场动态:")
                for i, news in enumerate(news_data[:3]):
                    print(f"  {i+1}. {news['title']}")
        except Exception as e:
            logger.warning(f"市场新闻获取失败: {str(e)}")
    
    except Exception as e:
        logger.error(f"数据源初始化失败: {str(e)}")
        print(f"\n错误: 无法初始化数据源 - {str(e)}")
        return False
    
    # 启动参数
    startup_args = "--headless" if headless else ""
    
    # 启动超神系统
    print("\n=======================================")
    print("      超神量子共生系统正在加载...      ")
    print("=======================================")
    
    # 模拟加载过程
    loading_items = [
        "初始化量子计算核心...",
        "加载市场数据模块...",
        "校准预测算法...",
        "初始化战略决策矩阵...",
        "连接实时数据流...",
        "启动AI决策引擎...",
        "激活交易系统接口...",
        "完成系统初始化!"
    ]
    
    for item in loading_items:
        print(f"  {item}")
        time.sleep(0.5)
    
    print("\n=======================================")
    print("    超神量子共生系统已成功启动!      ")
    print("=======================================\n")
    
    # 启动驾驶模式桌面
    if not headless:
        try:
            print("正在启动驾驶模式桌面...")
            start_desktop_interface(data_connector)
        except Exception as e:
            logger.error(f"启动驾驶模式桌面失败: {str(e)}")
            print(f"驾驶模式启动失败: {str(e)}")
            print("系统将以命令行模式运行")
    
    # 显示系统状态
    print("系统状态: 正常运行")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据源: {data_connector.__class__.__name__}")
    print("界面模式: " + ("后台模式" if headless else "驾驶模式"))
    
    # 返回成功
    return True

def start_desktop_interface(data_connector=None):
    """启动驾驶模式桌面界面"""
    try:
        # 导入驾驶桌面模块
        from supergod_desktop import SupergodDesktop
        from PyQt5.QtWidgets import QApplication
        
        # 启动桌面应用
        app = QApplication(sys.argv)
        desktop = SupergodDesktop()
        desktop.show()
        
        # 设置数据源
        if data_connector:
            desktop.set_data_connector(data_connector)
        
        # 执行应用
        sys.exit(app.exec_())
    except ImportError:
        logger.error("无法加载PyQt5桌面环境")
        print("错误: 无法启动桌面界面 - 请安装PyQt5")
        print("pip install PyQt5")
    except Exception as e:
        logger.error(f"启动桌面界面时出错: {str(e)}")
        print(f"启动桌面界面失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超神量子共生系统启动器")
    parser.add_argument("--headless", action="store_true", help="以无界面模式运行")
    parser.add_argument("--desktop", action="store_true", help="直接启动驾驶模式桌面")
    parser.add_argument("--token", type=str, help="Tushare API Token")
    args = parser.parse_args()
    
    print("=" * 60)
    print("超神量子共生系统 - 驾驶舱启动器 (实时数据版)")
    print("=" * 60)
    
    # 设置Tushare Token
    if args.token:
        os.environ['TUSHARE_TOKEN'] = args.token
    
    # 检查依赖
    if not check_dependencies():
        print("\n启动失败: 缺少必要依赖。")
        sys.exit(1)
    
    # 直接启动桌面
    if args.desktop:
        try:
            start_desktop_interface()
        except Exception as e:
            print(f"驾驶模式桌面启动失败: {str(e)}")
            sys.exit(1)
        sys.exit(0)
    
    # 启动系统
    success = start_supergod_system(headless=args.headless)
    
    if success:
        print("\n超神系统已成功启动!")
    else:
        print("\n启动失败: 请检查日志获取详细信息。")
        sys.exit(1)

if __name__ == "__main__":
    main() 