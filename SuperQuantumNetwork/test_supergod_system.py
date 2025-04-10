#!/usr/bin/env python3
"""
超神量子共生系统 - 集成测试
验证中国市场分析核心、政策分析器和板块轮动跟踪器的协同工作
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import traceback
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("supergod_test.log")
    ]
)

logger = logging.getLogger("SupergodTest")

# 导入核心模块
try:
    from china_market_core import ChinaMarketCore, MarketCycle, MarketFactor
    from policy_analyzer import PolicyAnalyzer
    from sector_rotation_tracker import SectorRotationTracker
    
    logger.info("成功导入所有核心模块")
except ImportError as e:
    logger.error(f"导入核心模块失败: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def generate_sample_market_data(days=100):
    """生成样本市场数据"""
    logger.info("生成样本市场数据...")
    
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 设置随机种子以保证结果可重现
    np.random.seed(42)
    
    # 生成基础价格序列 (模拟大盘指数)
    base_changes = np.random.normal(0.0005, 0.015, len(dates))
    
    # 添加一些趋势和季节性
    trend = np.linspace(0, 0.1, len(dates))
    seasonality = 0.02 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    
    # 组合成最终的价格变化
    changes = base_changes + np.diff(np.append([0], trend)) + np.diff(np.append([0], seasonality))
    
    # 计算价格序列
    prices = 3000 * np.cumprod(1 + changes)
    
    # 生成成交量数据
    volumes = np.random.normal(1e9, 2e8, len(dates))
    volumes = np.abs(volumes)  # 确保成交量为正
    
    # 创建市场数据DataFrame
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': volumes
    })
    
    logger.info(f"样本市场数据生成完成，包含 {len(market_data)} 个交易日")
    return market_data

def generate_sample_policy_news(count=10):
    """生成样本政策新闻"""
    logger.info(f"生成 {count} 条样本政策新闻...")
    
    # 政策新闻样本
    news_templates = [
        {
            'title': '央行宣布下调存款准备金率{rate}个百分点',
            'content': '中国人民银行今日宣布，下调金融机构存款准备金率{rate}个百分点，释放长期资金约{amount}亿元。此举旨在优化金融机构资金结构，增强金融服务实体经济的能力。',
            'params': {'rate': ['0.25', '0.5', '0.75', '1.0'], 'amount': ['5000', '8000', '10000', '12000']}
        },
        {
            'title': '发改委批复多项重大{sector}项目',
            'content': '国家发展改革委近日批复了多项重大{sector}建设项目，总投资超过{amount}亿元，涉及{details}等多个领域，旨在扩大有效投资，促进经济稳定增长。',
            'params': {
                'sector': ['基础设施', '能源', '交通', '水利'],
                'amount': ['3000', '5000', '8000'],
                'details': ['高速铁路、城市轨道交通', '清洁能源、电网改造', '高速公路、航空枢纽', '水库、灌溉系统']
            }
        },
        {
            'title': '财政部发布关于{action}的通知',
            'content': '财政部近日发布关于{action}的通知，{details}。相关政策将于{date}起实施，预计将对{impact}产生积极影响。',
            'params': {
                'action': ['实施积极财政政策', '进一步减税降费', '加大财政支持力度', '专项债券发行'],
                'details': [
                    '明确了今年积极财政政策的具体措施和实施路径',
                    '提出了一系列针对小微企业和制造业的减税降费措施',
                    '要求各地加大财政支出力度，重点支持重大项目建设',
                    '安排了新一批专项债券额度，重点支持基础设施建设'
                ],
                'date': ['本月底', '下月初', '本季度末', '年底前'],
                'impact': ['实体经济', '制造业升级', '基建投资', '内需扩张']
            }
        },
        {
            'title': '中央经济工作会议强调{focus}',
            'content': '近日召开的中央经济工作会议强调，要{focus}，{details}。会议指出，要{actions}，为经济高质量发展提供有力支撑。',
            'params': {
                'focus': ['稳中求进', '促进经济平稳健康发展', '加快构建新发展格局', '深化供给侧结构性改革'],
                'details': [
                    '统筹推进稳增长、促改革、调结构、惠民生、防风险',
                    '加快建设现代化经济体系，推动质量变革、效率变革、动力变革',
                    '加快科技自立自强，提升产业链供应链韧性和安全水平',
                    '扎实做好碳达峰碳中和工作，推动绿色低碳发展'
                ],
                'actions': [
                    '保持宏观政策连续性稳定性，增强调控的前瞻性精准性',
                    '提升产业基础能力和产业链现代化水平',
                    '强化企业创新主体地位，促进科技与产业深度融合',
                    '扩大高水平对外开放，推动贸易创新发展'
                ]
            }
        },
        {
            'title': '{department}发布{industry}行业监管新规',
            'content': '{department}近日发布了{industry}行业监管新规，{details}。新规将于{date}起实施，业内人士认为，此举将{impact}。',
            'params': {
                'department': ['证监会', '银保监会', '市场监管总局', '国家网信办'],
                'industry': ['互联网金融', '房地产', '医药', '教育培训', '平台经济'],
                'details': [
                    '明确了企业合规经营的具体要求和红线',
                    '加强了对行业关键环节的风险管控',
                    '规范了市场竞争秩序，防止资本无序扩张',
                    '完善了用户数据和隐私保护措施'
                ],
                'date': ['下个月', '本季度末', '年底前', '明年初'],
                'impact': [
                    '促进行业健康有序发展',
                    '有效防范化解金融风险',
                    '引导企业回归主业，专注技术创新',
                    '保障消费者合法权益'
                ]
            }
        }
    ]
    
    # 生成新闻
    news_items = []
    end_date = datetime.now()
    
    for i in range(count):
        # 随机选择模板
        template = news_templates[i % len(news_templates)]
        
        # 填充模板参数
        title = template['title']
        content = template['content']
        
        for param, values in template['params'].items():
            value = np.random.choice(values)
            title = title.replace('{' + param + '}', value)
            content = content.replace('{' + param + '}', value)
        
        # 分配日期 (过去的日期，越近的越多)
        days_ago = int(np.random.exponential(7))
        days_ago = min(days_ago, 30)  # 最多30天前
        news_date = (end_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # 创建新闻项
        news_item = {
            'title': title,
            'content': content,
            'time': news_date,
            'source': f'样本来源 {i+1}'
        }
        
        news_items.append(news_item)
    
    logger.info(f"样本政策新闻生成完成，共 {len(news_items)} 条")
    return news_items

def generate_sample_sector_data(sectors, days=100):
    """生成样本板块数据"""
    # 确保sectors是列表
    if isinstance(sectors, str):
        sectors = [sectors]
        
    logger.info(f"为 {len(sectors)} 个板块生成样本数据...")
    
    # 生成日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # 设置随机种子以保证结果可重现
    np.random.seed(42)
    
    # 生成基础市场变化 (所有板块都会受到的共同影响)
    base_market_changes = np.random.normal(0.0005, 0.01, len(dates))
    
    # 为每个板块生成数据
    sector_data = {}
    
    for idx, sector in enumerate(sectors):
        # 使用不同的随机种子，使每个板块有不同的特性
        np.random.seed(42 + idx)
        
        # 板块特定的变化
        sector_specific_changes = np.random.normal(0.0, 0.015, len(dates))
        
        # 添加一些板块特定的趋势
        if sector in ['医药生物', '电子', '计算机']:
            # 成长型板块，有上升趋势
            trend = np.linspace(0, 0.15, len(dates))
        elif sector in ['银行', '房地产', '公用事业']:
            # 价值型板块，趋势较平稳
            trend = np.linspace(0, 0.05, len(dates))
        elif sector in ['有色金属', '钢铁', '化工']:
            # 周期型板块，有波动的趋势
            trend = 0.1 * np.sin(np.linspace(0, 3*np.pi, len(dates)))
        else:
            # 其他板块，随机趋势
            trend = 0.08 * np.sin(np.linspace(0, 2*np.pi + idx, len(dates)))
        
        # 加入一些特殊事件（某些时段表现特别好或特别差）
        special_events = np.zeros(len(dates))
        
        # 对于一些板块，加入政策影响
        if sector in ['医药生物', '食品饮料'] and idx % 3 == 0:
            # 某个时间段表现特别好
            good_period_start = len(dates) // 3
            good_period_end = good_period_start + len(dates) // 6
            special_events[good_period_start:good_period_end] = 0.008
            
        if sector in ['银行', '房地产'] and idx % 2 == 0:
            # 某个时间段表现特别差
            bad_period_start = len(dates) // 2
            bad_period_end = bad_period_start + len(dates) // 5
            special_events[bad_period_start:bad_period_end] = -0.006
        
        # 组合所有影响因素
        combined_changes = base_market_changes + sector_specific_changes + np.diff(np.append([0], trend)) + special_events
        
        # 计算价格序列
        base_price = 100
        prices = base_price * np.cumprod(1 + combined_changes)
        
        # 生成成交量数据
        volumes = np.random.normal(1000000, 200000, len(dates))
        volumes = np.abs(volumes)  # 确保成交量为正
        
        # 创建DataFrame
        sector_df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.015, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.015, len(dates))),
            'close': prices,
            'volume': volumes
        })
        
        sector_data[sector] = sector_df
    
    logger.info(f"样本板块数据生成完成，每个板块包含 {len(dates)} 个交易日")
    return sector_data

def test_modules_individually():
    """测试各个模块的独立功能"""
    success = True
    
    # 测试中国市场分析核心
    try:
        logger.info("测试中国市场分析核心...")
        market_core = ChinaMarketCore()
        market_data = generate_sample_market_data()
        
        analysis_result = market_core.analyze_market(market_data)
        
        if not analysis_result or 'error' in analysis_result:
            logger.error(f"市场分析失败: {analysis_result.get('error', '未知错误')}")
            success = False
        else:
            logger.info(f"市场分析成功，当前市场周期: {analysis_result['market_cycle']}")
            logger.info(f"检测到 {len(analysis_result['detected_anomalies'])} 个市场异常")
            logger.info(f"生成了 {len(analysis_result['recommendations'])} 条建议")
    except Exception as e:
        logger.error(f"测试中国市场分析核心时出错: {str(e)}")
        logger.error(traceback.format_exc())
        success = False
    
    # 测试政策分析器
    try:
        logger.info("测试政策分析器...")
        policy_analyzer = PolicyAnalyzer()
        policy_news = generate_sample_policy_news()
        
        # 添加政策新闻
        policy_analyzer.add_policy_news(policy_news)
        
        # 注入政策事件
        policy_analyzer.inject_policy_event('interest_rate_change', -0.5)
        
        # 分析政策环境
        policy_result = policy_analyzer.analyze()
        
        if not policy_result:
            logger.error("政策分析失败")
            success = False
        else:
            logger.info(f"政策分析成功，当前政策方向: {policy_result['policy_direction']:.2f}")
            logger.info(f"政策不确定性: {policy_result['policy_uncertainty']:.2f}")
            logger.info(f"政策影响行业数量: {len(policy_result['policy_impact_sectors'])}")
    except Exception as e:
        logger.error(f"测试政策分析器时出错: {str(e)}")
        logger.error(traceback.format_exc())
        success = False
    
    # 测试板块轮动跟踪器
    try:
        logger.info("测试板块轮动跟踪器...")
        tracker = SectorRotationTracker()
        
        # 生成板块数据
        sectors = ['银行', '房地产', '医药生物', '食品饮料', '电子', '计算机',
                  '有色金属', '钢铁', '军工', '汽车', '家电', '电力设备']
        sector_data = generate_sample_sector_data(sectors)
        
        # 更新板块数据
        tracker.update_sector_data(sector_data)
        
        # 分析板块轮动
        rotation_result = tracker.analyze()
        
        if not rotation_result:
            logger.error("板块轮动分析失败")
            success = False
        else:
            logger.info(f"板块轮动分析成功，轮动强度: {rotation_result['rotation_strength']:.2f}")
            logger.info(f"轮动方向: {rotation_result['rotation_direction']}")
            logger.info(f"领先板块数量: {len(rotation_result['leading_sectors'])}")
            logger.info(f"滞后板块数量: {len(rotation_result['lagging_sectors'])}")
    except Exception as e:
        logger.error(f"测试板块轮动跟踪器时出错: {str(e)}")
        logger.error(traceback.format_exc())
        success = False
    
    return success

def test_modules_integration():
    """测试模块间的集成"""
    try:
        logger.info("测试模块集成...")
        
        # 初始化各模块
        market_core = ChinaMarketCore()
        policy_analyzer = PolicyAnalyzer()
        sector_tracker = SectorRotationTracker()
        
        # 生成样本数据
        market_data = generate_sample_market_data()
        policy_news = generate_sample_policy_news()
        sectors = ['银行', '房地产', '医药生物', '食品饮料', '电子', '计算机',
                  '有色金属', '钢铁', '军工', '汽车', '家电', '电力设备']
        sector_data = generate_sample_sector_data(sectors)
        
        # 注册子模块到市场核心
        market_core.register_policy_analyzer(policy_analyzer)
        market_core.register_sector_rotation_tracker(sector_tracker)
        
        # 更新数据
        policy_analyzer.add_policy_news(policy_news)
        sector_tracker.update_sector_data(sector_data)
        
        # 进行综合分析
        logger.info("执行综合市场分析...")
        additional_data = {
            'policy_news': policy_news,
            'policy_events': [
                {'type': 'interest_rate_change', 'value': -0.25},
                {'type': 'fiscal_stimulus', 'value': 0.5}
            ]
        }
        
        analysis_result = market_core.analyze_market(market_data, additional_data)
        
        if not analysis_result or 'error' in analysis_result:
            logger.error(f"综合分析失败: {analysis_result.get('error', '未知错误')}")
            return False
        
        # 输出分析结果
        logger.info("综合分析结果:")
        logger.info(f"- 市场周期: {analysis_result['market_cycle']}")
        logger.info(f"- 周期置信度: {analysis_result['cycle_confidence']:.2f}")
        logger.info(f"- 市场情绪: {analysis_result['market_sentiment']:.2f}")
        logger.info(f"- 政策方向: {analysis_result['policy_direction']:.2f}")
        logger.info(f"- 板块轮动: {analysis_result['sector_rotation']}")
        
        logger.info(f"- 主导因子: {', '.join([f['factor'] for f in analysis_result['dominant_factors']])}")
        
        logger.info("- 检测到的异常:")
        for anomaly in analysis_result.get('detected_anomalies', []):
            logger.info(f"  * {anomaly['type']}: 严重度 {anomaly['severity']:.2f}")
        
        logger.info("- 建议:")
        for rec in analysis_result.get('recommendations', []):
            logger.info(f"  * [{rec['type']}] {rec['content']}")
        
        logger.info("模块集成测试完成")
        return True
        
    except Exception as e:
        logger.error(f"模块集成测试失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_advanced_warning_system():
    """测试超神系统的高级市场预警功能"""
    from china_market_core import ChinaMarketCore
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("\n===== 测试高级市场预警系统 =====")
    
    # 创建中国市场分析核心
    market_core = ChinaMarketCore()
    
    # 激活高级预警系统
    market_core.activate_advanced_warning_system(True)
    
    # 生成模拟市场数据
    days = 250
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # 模拟三种市场状态: 趋势上涨、盘整、急跌
    price = 3000
    prices = []
    volumes = []
    
    # 第一阶段: 稳步上涨 (0-100)
    for i in range(100):
        # 生成小幅波动的上涨趋势
        change = np.random.normal(0.5, 1.0)
        price *= (1 + change / 100)
        prices.append(price)
        # 生成相应的成交量
        volume = np.random.normal(1000000, 200000)
        volumes.append(volume)
    
    # 第二阶段: 盘整震荡 (100-180)
    for i in range(80):
        # 生成小幅波动的盘整
        change = np.random.normal(0, 1.5)
        price *= (1 + change / 100)
        prices.append(price)
        # 生成相应的成交量
        volume = np.random.normal(800000, 150000)
        volumes.append(volume)
    
    # 第三阶段: 市场拐点和急跌 (180-250)
    for i in range(20):
        # 生成拐点区域的波动
        change = np.random.normal(-0.2, 1.8)
        price *= (1 + change / 100)
        prices.append(price)
        # 成交量逐渐放大
        volume = np.random.normal(1200000, 300000)
        volumes.append(volume)
    
    # 急跌阶段
    for i in range(50):
        # 生成剧烈下跌
        change = np.random.normal(-1.5, 2.2)
        price *= (1 + change / 100)
        prices.append(price)
        # 成交量急剧放大
        volume = np.random.normal(1800000, 500000) 
        volumes.append(volume)
    
    # 创建市场数据DataFrame
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * np.random.uniform(0.99, 1.01, len(prices)),
        'high': prices * np.random.uniform(1.01, 1.03, len(prices)),
        'low': prices * np.random.uniform(0.97, 0.99, len(prices)),
        'close': prices,
        'volume': volumes
    })
    
    # 设置日期为索引
    market_data.set_index('date', inplace=True)
    
    # 进行市场分析
    print("开始对模拟市场数据进行高级市场分析...")
    result = market_core.analyze_market(market_data)
    
    # 检查预警系统结果
    warnings = result.get('warning_signals', [])
    anomalies = result.get('detected_anomalies', [])
    
    print(f"\n检测到的预警信号: {len(warnings)}")
    for warning in warnings:
        print(f"- {warning['type']}: {warning['description']}, 置信度: {warning.get('confidence', 'N/A')}")
    
    print(f"\n检测到的市场异常: {len(anomalies)}")
    for anomaly in anomalies:
        print(f"- {anomaly['type']}: {anomaly.get('description', '')}, 严重程度: {anomaly.get('severity', 'N/A')}")
    
    # 打印市场结构分析结果
    structure = result.get('market_structure', {})
    if structure:
        print("\n市场结构分析:")
        print(f"- 赫斯特指数: {structure.get('hurst_exponent', 0):.3f} (>0.5趋势性, <0.5反转性)")
        print(f"- 分形维度: {structure.get('fractal_dimension', 0):.3f}")
        print(f"- 市场熵值: {structure.get('entropy', 0):.3f}")
        print(f"- 复杂度: {structure.get('complexity', 0):.3f}")
        print(f"- 市场状态: {structure.get('regime', 'unknown')}")
        print(f"- 稳定性: {structure.get('stability', 0):.3f}")
    
    # 绘制市场数据和预警点
    plt.figure(figsize=(14, 10))
    
    # 股价走势图
    plt.subplot(2, 1, 1)
    plt.plot(market_data.index, market_data['close'], label='收盘价')
    plt.title('市场走势和高级预警系统检测结果')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.grid(True)
    
    # 标记拐点
    turning_points = market_core.advanced_warning_system.get('turning_points', [])
    for tp in turning_points:
        idx = tp.get('index')
        if idx is not None and idx < len(market_data):
            plt.plot(market_data.index[idx], market_data['close'].iloc[idx], 'ro', markersize=8, 
                     label=f"{tp.get('direction', '')}拐点" if 'direction' in tp else '拐点')
    
    plt.legend()
    
    # 成交量图
    plt.subplot(2, 1, 2)
    plt.bar(market_data.index, market_data['volume'], alpha=0.7)
    plt.title('成交量')
    plt.xlabel('日期')
    plt.ylabel('成交量')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('market_warning_test.png')
    print("\n结果图表已保存为 market_warning_test.png")
    
    return market_core, result, market_data

def main():
    logger.info("=" * 50)
    logger.info("超神量子共生系统 - 集成测试开始")
    logger.info("=" * 50)
    
    # 测试各模块独立功能
    modules_ok = test_modules_individually()
    
    if not modules_ok:
        logger.error("独立模块测试失败，不继续进行集成测试")
        return False
    
    # 测试模块集成
    integration_ok = test_modules_integration()
    
    if not integration_ok:
        logger.error("模块集成测试失败")
        return False
    
    # 进行高级预警系统测试
    try:
        market_core, result, market_data = test_advanced_warning_system()
        print("\n高级市场预警系统测试完成!")
    except Exception as e:
        print(f"\n高级市场预警系统测试失败: {str(e)}")
        traceback.print_exc()
    
    logger.info("=" * 50)
    logger.info("超神量子共生系统 - 集成测试成功完成")
    logger.info("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 