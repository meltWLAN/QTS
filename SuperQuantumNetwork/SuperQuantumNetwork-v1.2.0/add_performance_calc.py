#!/usr/bin/env python3
"""
添加_calculate_sector_performance和_load_config方法到sector_rotation_tracker.py文件
"""

import re

def main():
    # 读取文件
    with open('sector_rotation_tracker.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加_calculate_sector_performance方法
    calculate_perf_method = """
    def _calculate_sector_performance(self):
        \"\"\"计算板块表现\"\"\"
        if not self.sector_data:
            return
            
        self.sector_performance = {}
        
        for sector_name, sector_df in self.sector_data.items():
            try:
                # 确保数据足够
                if len(sector_df) < self.config['long_term_window']:
                    self.logger.warning(f"板块 {sector_name} 数据不足，跳过表现计算")
                    continue
                
                # 计算收益率
                sector_df['return'] = sector_df['close'].pct_change()
                
                # 计算不同时间窗口的表现
                short_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['short_term_window']] - 1
                medium_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['medium_term_window']] - 1
                long_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['long_term_window']] - 1
                
                # 计算动量
                momentum = self._calculate_momentum(sector_df)
                
                # 计算波动率
                volatility = sector_df['return'].std() * (252 ** 0.5)  # 年化波动率
                
                # 计算成交量变化
                volume_change = sector_df['volume'].iloc[-5:].mean() / sector_df['volume'].iloc[-20:-5].mean() - 1
                
                # 存储表现指标
                self.sector_performance[sector_name] = {
                    'short_term': short_term,
                    'medium_term': medium_term,
                    'long_term': long_term,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume_change': volume_change
                }
                
            except Exception as e:
                self.logger.error(f"计算板块 {sector_name} 表现时出错: {str(e)}")
    
    def _calculate_momentum(self, sector_df):
        \"\"\"计算动量\"\"\"
        # 使用移动平均比较计算动量
        momentum_window = self.config['momentum_window']
        
        if len(sector_df) < momentum_window * 2:
            return 0.0
            
        short_ma = sector_df['close'].rolling(momentum_window // 2).mean()
        long_ma = sector_df['close'].rolling(momentum_window).mean()
        
        momentum = short_ma.iloc[-1] / long_ma.iloc[-1] - 1
        return momentum
    
    def _load_config(self, config_path):
        \"\"\"从文件加载配置\"\"\"
        import json
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 更新配置
            self.config.update(config)
            self.logger.info(f"从 {config_path} 加载了配置")
        except Exception as e:
            self.logger.error(f"加载配置时出错: {str(e)}")
    """
    
    # 寻找添加方法的位置 - 在_detect_rotation方法之后
    pattern = r'def _detect_rotation\(self\):.*?self\.logger\.info\(f"板块轮动检测完成.*?\)(\s*)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # 找到位置，添加新方法
        insert_pos = match.end(1)
        new_content = content[:insert_pos] + calculate_perf_method + content[insert_pos:]
        
        # 写入文件
        with open('sector_rotation_tracker.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("已添加_calculate_sector_performance, _calculate_momentum和_load_config方法")
    else:
        print("未找到合适位置添加方法")

if __name__ == "__main__":
    main() 