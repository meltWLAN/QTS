#!/usr/bin/env python3
"""
添加_calculate_sector_rankings方法到sector_rotation_tracker.py文件
"""

import re

def main():
    # 读取文件
    with open('sector_rotation_tracker.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加_calculate_sector_rankings方法
    rankings_method = """
    def _calculate_sector_rankings(self):
        \"\"\"计算板块排名\"\"\"
        if not self.sector_performance:
            return
            
        # 基于不同指标的排名
        rankings = {}
        
        try:
            # 短期表现排名
            short_term_data = {k: v['short_term'] for k, v in self.sector_performance.items() if isinstance(v, dict) and 'short_term' in v}
            short_term_ranking = sorted(short_term_data.items(), key=lambda x: x[1], reverse=True)
            rankings['short_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(short_term_ranking)}
            
            # 中期表现排名
            medium_term_data = {k: v['medium_term'] for k, v in self.sector_performance.items() if isinstance(v, dict) and 'medium_term' in v}
            medium_term_ranking = sorted(medium_term_data.items(), key=lambda x: x[1], reverse=True)
            rankings['medium_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(medium_term_ranking)}
            
            # 长期表现排名
            long_term_data = {k: v['long_term'] for k, v in self.sector_performance.items() if isinstance(v, dict) and 'long_term' in v}
            long_term_ranking = sorted(long_term_data.items(), key=lambda x: x[1], reverse=True)
            rankings['long_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(long_term_ranking)}
            
            # 动量排名
            momentum_data = {k: v['momentum'] for k, v in self.sector_performance.items() if isinstance(v, dict) and 'momentum' in v}
            momentum_ranking = sorted(momentum_data.items(), key=lambda x: x[1], reverse=True)
            rankings['momentum'] = {sector: rank + 1 for rank, (sector, _) in enumerate(momentum_ranking)}
            
            # 综合排名 (使用加权平均)
            composite_scores = {}
            for sector in short_term_data.keys():
                if (sector in rankings['short_term'] and sector in rankings['medium_term'] and 
                    sector in rankings['momentum']):
                    # 计算综合得分 (排名越小越好)
                    composite_scores[sector] = (
                        rankings['short_term'][sector] * 0.3 +
                        rankings['medium_term'][sector] * 0.3 +
                        rankings['momentum'][sector] * 0.4
                    )
                    
            composite_ranking = sorted(composite_scores.items(), key=lambda x: x[1])
            rankings['composite'] = {sector: rank + 1 for rank, (sector, _) in enumerate(composite_ranking)}
            
            # 更新轮动状态
            self.rotation_state['sector_rankings'] = rankings
            
        except Exception as e:
            self.logger.error(f"计算板块排名时出错: {str(e)}")
    """
    
    # 寻找添加方法的位置 - 在_load_config方法之后
    pattern = r'def _load_config\(self, config_path\):.*?self\.logger\.error\(f"加载配置时出错: {str\(e\)}"\)(\s*)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # 找到位置，添加新方法
        insert_pos = match.end(1)
        new_content = content[:insert_pos] + rankings_method + content[insert_pos:]
        
        # 写入文件
        with open('sector_rotation_tracker.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("已添加_calculate_sector_rankings方法")
    else:
        print("未找到合适位置添加方法")

if __name__ == "__main__":
    main() 