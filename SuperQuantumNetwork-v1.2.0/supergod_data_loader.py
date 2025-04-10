#!/usr/bin/env python3
"""
超神量子共生系统 - 数据加载器
加载各类数据，包括市场数据、政策数据和板块数据
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta

# 导入Tushare数据连接器
try:
    from tushare_data_connector import TushareDataConnector
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

# 设置日志
logger = logging.getLogger("SupergodDataLoader")

class SupergodDataLoader:
    """数据加载器 - 负责加载各类数据"""
    
    def __init__(self, data_dir="data", use_tushare=True, tushare_token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"):
        """
        初始化数据加载器
        
        参数:
            data_dir: 数据目录
            use_tushare: 是否使用Tushare Pro API获取实时数据
            tushare_token: Tushare Pro API的访问令牌
        """
        self.data_dir = data_dir
        self.use_tushare = use_tushare and TUSHARE_AVAILABLE
        self.tushare_token = tushare_token
        self.tushare_connector = None
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"创建数据目录: {data_dir}")
            
        # 初始化Tushare连接器
        if self.use_tushare:
            try:
                self.tushare_connector = TushareDataConnector(token=tushare_token)
                logger.info("初始化Tushare数据连接器成功")
            except Exception as e:
                logger.error(f"初始化Tushare数据连接器失败: {str(e)}")
                self.use_tushare = False
    
    def load_market_data(self, file_path=None, days=100, use_demo=True, code="000001.SH", use_real_data=True):
        """
        加载市场数据
        
        参数:
            file_path: 数据文件路径，如果为None则尝试使用Tushare或演示数据
            days: 演示数据的天数
            use_demo: 是否使用演示数据（当无法获取实时数据时）
            code: 股票或指数代码，当使用Tushare时有效
            use_real_data: 是否优先使用真实数据
            
        返回:
            DataFrame: 市场数据
        """
        logger.info("加载市场数据...")
        
        # 尝试使用Tushare获取实时数据
        if use_real_data and self.use_tushare and self.tushare_connector:
            try:
                logger.info(f"正在从Tushare获取真实市场数据: {code}")
                # 获取最近30天的数据
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
                
                df = self.tushare_connector.get_market_data(
                    code=code, 
                    start_date=start_date, 
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    logger.info(f"成功从Tushare获取 {len(df)} 条市场数据")
                    
                    # 保存到本地文件以备将来使用
                    self.save_market_data(df, f"market_data_{code.replace('.', '_')}.csv")
                    
                    return df
                else:
                    logger.warning("从Tushare获取数据失败，尝试其他方式")
            except Exception as e:
                logger.error(f"从Tushare获取市场数据出错: {str(e)}")
        
        # 如果无法从Tushare获取，则尝试从文件加载
        if file_path and os.path.exists(file_path):
            try:
                # 尝试加载CSV文件
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    logger.info(f"从CSV文件加载市场数据: {file_path}")
                # 尝试加载Excel文件
                elif file_path.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file_path)
                    logger.info(f"从Excel文件加载市场数据: {file_path}")
                else:
                    logger.warning(f"未支持的文件格式: {file_path}")
                    return self._generate_demo_market_data(days) if use_demo else None
                
                # 处理日期列
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
            except Exception as e:
                logger.error(f"加载市场数据文件失败: {str(e)}")
                return self._generate_demo_market_data(days) if use_demo else None
        
        # 检查数据目录下是否有市场数据文件
        code_suffix = f"_{code.replace('.', '_')}" if code else ""
        market_data_path = os.path.join(self.data_dir, f'market_data{code_suffix}.csv')
        if os.path.exists(market_data_path):
            try:
                df = pd.read_csv(market_data_path)
                logger.info(f"从默认位置加载市场数据: {market_data_path}")
                
                # 处理日期列
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                return df
            except Exception as e:
                logger.error(f"加载默认市场数据文件失败: {str(e)}")
        
        # 如果无法从文件加载，则生成演示数据
        if use_demo:
            logger.info("使用演示市场数据")
            return self._generate_demo_market_data(days)
        
        logger.warning("无法加载市场数据")
        return None
    
    def _generate_demo_market_data(self, days=100):
        """
        生成演示市场数据
        
        参数:
            days: 天数
            
        返回:
            DataFrame: 演示市场数据
        """
        import numpy as np
        
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 创建价格序列 - 添加趋势、循环和噪声
        base = 3000
        trend = np.linspace(0, 0.15, len(dates))
        cycle1 = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        cycle2 = 0.03 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.01, len(dates))
        
        # 计算每日涨跌幅
        changes = np.diff(np.concatenate([[0], trend])) + cycle1 + cycle2 + noise
        
        # 计算价格
        prices = base * np.cumprod(1 + changes)
        
        # 生成成交量
        volume_base = 1e9
        volume_trend = np.linspace(0, 0.3, len(dates))
        volume_cycle = 0.2 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
        volume_noise = np.random.normal(0, 0.15, len(dates))
        volumes = volume_base * (1 + volume_trend + volume_cycle + volume_noise)
        volumes = np.abs(volumes)  # 确保成交量为正
        
        # 创建数据框
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - 0.005 * np.random.random(len(dates))),
            'high': prices * (1 + 0.01 * np.random.random(len(dates))),
            'low': prices * (1 - 0.01 * np.random.random(len(dates))),
            'close': prices,
            'volume': volumes,
            'turnover_rate': 0.5 * (1 + 0.5 * np.random.random(len(dates)))
        })
        
        data.set_index('date', inplace=True)
        logger.info(f"生成演示市场数据: {len(data)}行")
        return data
    
    def load_policy_data(self, use_real_data=True):
        """
        加载政策数据
        
        参数:
            use_real_data: 是否优先使用真实数据
            
        返回:
            dict: 政策数据
        """
        logger.info("加载政策数据...")
        
        # 尝试使用Tushare获取实时政策新闻数据
        if use_real_data and self.use_tushare and self.tushare_connector:
            try:
                logger.info("正在从Tushare获取真实政策数据")
                policy_data = self.tushare_connector.get_policy_news()
                
                if policy_data and policy_data['policy_news']:
                    logger.info(f"成功从Tushare获取 {len(policy_data['policy_news'])} 条政策新闻")
                    
                    # 保存到本地文件以备将来使用
                    with open(os.path.join(self.data_dir, 'policy_data_real.json'), 'w', encoding='utf-8') as f:
                        json.dump(policy_data, f, ensure_ascii=False, indent=2)
                    
                    return policy_data
                else:
                    logger.warning("从Tushare获取政策数据失败，尝试从文件加载")
            except Exception as e:
                logger.error(f"从Tushare获取政策数据出错: {str(e)}")
        
        # 尝试从文件加载
        policy_data_path = os.path.join(self.data_dir, 'policy_data.json')
        if os.path.exists(policy_data_path):
            try:
                with open(policy_data_path, 'r', encoding='utf-8') as f:
                    policy_data = json.load(f)
                logger.info(f"从文件加载政策数据: {policy_data_path}")
                return policy_data
            except Exception as e:
                logger.error(f"加载政策数据失败: {str(e)}")
        else:
            logger.warning(f"政策数据文件不存在: {policy_data_path}")
        
        # 无法加载，返回空数据
        return {
            "policy_news": [],
            "policy_events": []
        }
    
    def load_sector_data(self, use_real_data=True):
        """
        加载板块数据
        
        参数:
            use_real_data: 是否优先使用真实数据
            
        返回:
            dict: 板块数据
        """
        logger.info("加载板块数据...")
        
        # 尝试使用Tushare获取实时板块数据
        if use_real_data and self.use_tushare and self.tushare_connector:
            try:
                logger.info("正在从Tushare获取真实板块数据")
                sector_data = self.tushare_connector.get_sector_data()
                
                if sector_data:
                    # 检查sector_data的结构，兼容不同格式
                    if 'sectors' not in sector_data and 'leading_sectors' in sector_data and 'lagging_sectors' in sector_data:
                        # 创建sectors键，包含领先和滞后板块的合并数据
                        sector_data['sectors'] = sector_data['leading_sectors'] + sector_data['lagging_sectors']
                        if 'all_sectors' in sector_data and sector_data['all_sectors']:
                            # 如果有all_sectors，使用它替代sectors
                            sector_data['sectors'] = sector_data['all_sectors']
                    
                    logger.info(f"成功从Tushare获取 {len(sector_data.get('sectors', []))} 个板块数据")
                    
                    # 保存到本地文件以备将来使用
                    with open(os.path.join(self.data_dir, 'sector_data_real.json'), 'w', encoding='utf-8') as f:
                        json.dump(sector_data, f, ensure_ascii=False, indent=2)
                    
                    return sector_data
                else:
                    logger.warning("从Tushare获取板块数据失败，尝试从文件加载")
            except Exception as e:
                logger.error(f"从Tushare获取板块数据出错: {str(e)}")
        
        # 尝试从文件加载
        sector_data_path = os.path.join(self.data_dir, 'sector_data.json')
        if os.path.exists(sector_data_path):
            try:
                with open(sector_data_path, 'r', encoding='utf-8') as f:
                    sector_data = json.load(f)
                logger.info(f"从文件加载板块数据: {sector_data_path}")
                return sector_data
            except Exception as e:
                logger.error(f"加载板块数据失败: {str(e)}")
        else:
            logger.warning(f"板块数据文件不存在: {sector_data_path}")
        
        # 无法加载，返回空数据
        return {
            "sectors": [],
            "sector_rotation": {},
            "industry_groups": {}
        }
    
    def save_market_data(self, data, file_name="market_data.csv"):
        """
        保存市场数据
        
        参数:
            data: DataFrame格式的市场数据
            file_name: 文件名
        
        返回:
            bool: 是否成功
        """
        if data is None or not isinstance(data, pd.DataFrame):
            logger.error("无效的市场数据")
            return False
        
        # 保存路径
        file_path = os.path.join(self.data_dir, file_name)
        
        try:
            # 重置索引，确保日期列存在
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
            
            # 保存为CSV
            data.to_csv(file_path, index=False)
            logger.info(f"市场数据保存成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存市场数据失败: {str(e)}")
            return False
    
    def download_market_data(self, code="000001.SH", start_date=None, end_date=None, provider="demo"):
        """
        下载市场数据
        
        参数:
            code: 股票或指数代码
            start_date: 开始日期
            end_date: 结束日期
            provider: 数据提供商，目前支持"demo"
            
        返回:
            DataFrame: 市场数据
        """
        logger.info(f"下载市场数据: {code}")
        
        # 如果没有指定日期，使用默认值
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # 将日期转换为字符串
        start_str = start_date.strftime("%Y%m%d") if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime("%Y%m%d") if isinstance(end_date, datetime) else end_date
        
        # 仅支持demo模式
        if provider.lower() == "demo":
            days = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
            return self._generate_demo_market_data(days)
        
        # 将来可以支持实际数据提供商，比如tushare
        logger.warning(f"不支持的数据提供商: {provider}")
        return None

def get_data_loader(tushare_token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"):
    """
    获取数据加载器实例
    
    参数:
        tushare_token: TuShare Pro API的访问令牌
    
    返回:
        SupergodDataLoader: 数据加载器实例
    """
    return SupergodDataLoader(tushare_token=tushare_token)

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据加载器
    loader = SupergodDataLoader()
    
    # 测试加载市场数据
    market_data = loader.load_market_data()
    print(f"市场数据: {market_data.shape if market_data is not None else 'None'}")
    
    # 测试加载政策数据
    policy_data = loader.load_policy_data()
    print(f"政策数据: {len(policy_data.get('policy_news', [])) if policy_data else 'None'} 条新闻")
    
    # 测试加载板块数据
    sector_data = loader.load_sector_data()
    print(f"板块数据: {len(sector_data.get('sectors', [])) if sector_data else 'None'} 个板块") 