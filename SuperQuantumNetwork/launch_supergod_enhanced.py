#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子共生系统 - 增强版启动器
集成了全面的数据接口和量子爆发策略
"""

import sys
import os
import logging
from datetime import datetime
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"supergod_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SuperGodEnhanced")

# 导入数据集成服务
try:
    from data_integration import create_integration_service
except ImportError:
    logger.error("无法导入数据集成服务，请确保data_integration.py文件存在并且正确")
    sys.exit(1)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="超神量子共生系统 - 增强版启动器")
    parser.add_argument('--token', type=str, default=None, help="Tushare API令牌")
    parser.add_argument('--config', type=str, default=None, help="配置文件路径")
    parser.add_argument('--mode', type=str, default='backtest', help="运行模式: backtest/live/analyze")
    parser.add_argument('--strategy', type=str, default='enhanced', help="策略类型: enhanced/quantum/combined")
    parser.add_argument('--start_date', type=str, default='20200101', help="回测开始日期")
    parser.add_argument('--end_date', type=str, default='20231231', help="回测结束日期")
    parser.add_argument('--capital', type=float, default=2000000.0, help="初始资金")
    parser.add_argument('--report', type=str, default=None, help="回测报告输出路径")
    parser.add_argument('--index', type=str, default='000001.SH', help="基准指数")
    return parser.parse_args()

# 主函数
def main():
    """主函数"""
    logger.info("启动超神量子共生系统 - 增强版")
    
    # 解析命令行参数
    args = parse_args()
    
    # 显示配置信息
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"策略类型: {args.strategy}")
    if args.mode == 'backtest':
        logger.info(f"回测区间: {args.start_date} - {args.end_date}")
        logger.info(f"初始资金: {args.capital:,.2f}")
        logger.info(f"基准指数: {args.index}")
    
    # 创建数据集成服务
    logger.info("初始化数据集成服务...")
    data_service = create_integration_service(args.token, args.config)
    
    # 根据模式执行不同操作
    if args.mode == 'backtest':
        run_backtest(args, data_service)
    elif args.mode == 'live':
        run_live_trading(args, data_service)
    elif args.mode == 'analyze':
        run_analysis(args, data_service)
    else:
        logger.error(f"不支持的运行模式: {args.mode}")
        return 1
    
    logger.info("超神量子共生系统 - 增强版 运行完成")
    return 0

def run_backtest(args, data_service):
    """运行回测
    
    Args:
        args: 命令行参数
        data_service: 数据集成服务
    """
    logger.info("开始回测过程...")
    
    # 根据策略类型选择不同的回测脚本
    if args.strategy == 'enhanced':
        # 导入增强版量子爆发策略回测模块
        import test_enhanced_strategy
        
        # 设置回测参数
        test_enhanced_strategy.backtest_args = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'initial_capital': args.capital,
            'benchmark': args.index,
            'data_service': data_service
        }
        
        # 运行回测
        try:
            logger.info("运行增强版量子爆发策略回测...")
            test_enhanced_strategy.main()
            logger.info("增强版量子爆发策略回测完成")
        except Exception as e:
            logger.error(f"回测过程中发生错误: {str(e)}", exc_info=True)
            return False
    
    elif args.strategy == 'quantum':
        # 导入基础量子爆发策略回测模块
        import test_quantum_strategy
        
        # 设置回测参数
        test_quantum_strategy.backtest_args = {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'initial_capital': args.capital,
            'benchmark': args.index,
            'data_service': data_service
        }
        
        # 运行回测
        try:
            logger.info("运行基础量子爆发策略回测...")
            test_quantum_strategy.main()
            logger.info("基础量子爆发策略回测完成")
        except Exception as e:
            logger.error(f"回测过程中发生错误: {str(e)}", exc_info=True)
            return False
    
    elif args.strategy == 'combined':
        logger.info("运行组合策略回测...")
        # TODO: 实现组合策略回测
        logger.warning("组合策略回测尚未实现")
        return False
    
    else:
        logger.error(f"不支持的策略类型: {args.strategy}")
        return False
    
    # 生成回测报告
    if args.report:
        generate_report(args, data_service)
    
    return True

def run_live_trading(args, data_service):
    """运行实盘交易
    
    Args:
        args: 命令行参数
        data_service: 数据集成服务
    """
    logger.info("实盘交易模式尚未实现")
    return False

def run_analysis(args, data_service):
    """运行市场分析
    
    Args:
        args: 命令行参数
        data_service: 数据集成服务
    """
    logger.info("开始市场分析...")
    
    # 获取市场指数数据
    index_data = data_service.get_market_index_data(
        index_code=args.index,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if index_data is not None:
        logger.info(f"获取到 {len(index_data)} 条指数数据")
        
        # 保存指数数据
        data_service.save_data(index_data, f"{args.index}_analysis.csv")
        
        # TODO: 实现更多分析功能
        
        logger.info("市场分析完成")
        return True
    else:
        logger.error("未能获取市场指数数据，分析失败")
        return False

def generate_report(args, data_service):
    """生成回测报告
    
    Args:
        args: 命令行参数
        data_service: 数据集成服务
    """
    logger.info(f"生成回测报告: {args.report}")
    
    # TODO: 实现回测报告生成
    
    logger.info("回测报告生成完成")
    return True

if __name__ == "__main__":
    try:
        # 添加当前目录到Python路径，确保可以导入其他模块
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        # 执行主程序
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"执行过程中发生未处理的异常: {str(e)}", exc_info=True)
        sys.exit(1) 