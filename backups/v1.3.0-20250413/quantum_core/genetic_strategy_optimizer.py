"""
genetic_strategy_optimizer - 量子核心组件
基于遗传算法的策略优化器
"""

import logging
import random
import numpy as np
import time
from typing import Dict, List, Any, Callable, Tuple

logger = logging.getLogger(__name__)

class GeneticStrategyOptimizer:
    """遗传算法策略优化器 - 优化交易策略参数"""
    
    def __init__(self, population_size: int = 50, generations: int = 20):
        self.is_running = False
        self.population_size = population_size
        self.generations = generations
        self.strategies = {}
        self.current_optimization = None
        self.fitness_function = None
        self.best_strategies = {}
        logger.info("遗传算法策略优化器初始化完成")
        
    def start(self):
        """启动优化器"""
        if self.is_running:
            logger.warning("策略优化器已在运行")
            return
            
        logger.info("启动策略优化器...")
        self.is_running = True
        logger.info("策略优化器启动完成")
        
    def stop(self):
        """停止优化器"""
        if not self.is_running:
            logger.warning("策略优化器已停止")
            return
            
        logger.info("停止策略优化器...")
        self.is_running = False
        
        # 清理当前优化任务
        self.current_optimization = None
        
        logger.info("策略优化器已停止")
        
    def register_strategy(self, strategy_id: str, parameters: Dict[str, Dict], 
                         evaluator: Callable = None):
        """注册策略及其参数范围"""
        if strategy_id in self.strategies:
            logger.warning(f"策略 '{strategy_id}' 已存在，将被覆盖")
            
        self.strategies[strategy_id] = {
            'parameters': parameters,
            'evaluator': evaluator
        }
        
        logger.info(f"注册策略: {strategy_id}，参数数量: {len(parameters)}")
        return True
        
    def unregister_strategy(self, strategy_id: str):
        """注销策略"""
        if strategy_id not in self.strategies:
            logger.warning(f"策略 '{strategy_id}' 不存在")
            return False
            
        del self.strategies[strategy_id]
        
        # 如果有该策略的最佳策略记录，也一并删除
        if strategy_id in self.best_strategies:
            del self.best_strategies[strategy_id]
            
        logger.info(f"注销策略: {strategy_id}")
        return True
        
    def optimize(self, strategy_id: str, fitness_function: Callable, 
               population_size: int = None, generations: int = None, 
               market_data: Any = None) -> Dict:
        """优化策略参数"""
        if not self.is_running:
            logger.warning("策略优化器未运行，无法执行优化")
            return {'status': 'error', 'message': '策略优化器未运行'}
            
        if strategy_id not in self.strategies:
            logger.warning(f"策略 '{strategy_id}' 不存在")
            return {'status': 'error', 'message': f"策略 '{strategy_id}' 不存在"}
            
        logger.info(f"开始优化策略: {strategy_id}")
        
        # 使用设置的值或默认值
        pop_size = population_size or self.population_size
        gens = generations or self.generations
        
        strategy_config = self.strategies[strategy_id]
        parameters = strategy_config['parameters']
        
        # 设置当前优化任务
        self.current_optimization = {
            'strategy_id': strategy_id,
            'start_time': time.time(),
            'status': 'running',
            'progress': 0.0,
            'generations': gens,
            'population_size': pop_size,
        }
        
        # 设置适应度函数
        self.fitness_function = fitness_function
        
        try:
            # 初始化种群
            population = self._initialize_population(parameters, pop_size)
            
            best_individual = None
            best_fitness = float('-inf')
            
            # 开始进化
            for generation in range(gens):
                # 计算适应度
                fitness_scores = []
                for individual in population:
                    fitness = fitness_function(individual, market_data)
                    fitness_scores.append(fitness)
                    
                    # 更新最佳个体
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        
                # 选择
                selected = self._selection(population, fitness_scores)
                
                # 交叉
                offspring = self._crossover(selected)
                
                # 变异
                population = self._mutation(offspring, parameters)
                
                # 更新进度
                progress = (generation + 1) / gens
                self.current_optimization['progress'] = progress
                
                logger.debug(f"第 {generation+1}/{gens} 代，最佳适应度: {best_fitness}")
                
            # 保存最佳策略
            self.best_strategies[strategy_id] = {
                'parameters': best_individual,
                'fitness': best_fitness,
                'optimization_time': time.time() - self.current_optimization['start_time']
            }
            
            # 更新优化状态
            self.current_optimization['status'] = 'completed'
            self.current_optimization['end_time'] = time.time()
            self.current_optimization['best_fitness'] = best_fitness
            
            logger.info(f"策略 '{strategy_id}' 优化完成，最佳适应度: {best_fitness}")
            
            return {
                'status': 'success',
                'strategy_id': strategy_id,
                'best_parameters': best_individual,
                'best_fitness': best_fitness,
                'generations': gens,
                'population_size': pop_size,
                'optimization_time': self.current_optimization['end_time'] - self.current_optimization['start_time']
            }
            
        except Exception as e:
            logger.error(f"优化策略 '{strategy_id}' 时出错: {str(e)}")
            self.current_optimization['status'] = 'error'
            self.current_optimization['error'] = str(e)
            
            return {
                'status': 'error',
                'message': str(e)
            }
        finally:
            # 清理当前优化任务
            self.current_optimization = None
            
    def _initialize_population(self, parameters: Dict[str, Dict], population_size: int) -> List[Dict]:
        """初始化种群"""
        population = []
        
        for _ in range(population_size):
            individual = {}
            
            for param_name, param_config in parameters.items():
                param_type = param_config.get('type', 'float')
                
                if param_type == 'float':
                    min_val = param_config.get('min', 0.0)
                    max_val = param_config.get('max', 1.0)
                    individual[param_name] = random.uniform(min_val, max_val)
                    
                elif param_type == 'int':
                    min_val = param_config.get('min', 0)
                    max_val = param_config.get('max', 10)
                    individual[param_name] = random.randint(min_val, max_val)
                    
                elif param_type == 'choice':
                    choices = param_config.get('choices', [])
                    if choices:
                        individual[param_name] = random.choice(choices)
                    else:
                        individual[param_name] = None
                        
            population.append(individual)
            
        return population
        
    def _selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """选择操作 - 使用轮盘赌选择"""
        selected = []
        
        # 处理所有适应度为负的情况
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            # 偏移所有适应度使最小值为0.1
            fitness_scores = [score - min_fitness + 0.1 for score in fitness_scores]
            
        # 计算适应度总和
        total_fitness = sum(fitness_scores)
        
        if total_fitness <= 0:
            # 如果总适应度为0，使用均匀概率
            selection_probs = [1.0 / len(population)] * len(population)
        else:
            # 计算选择概率
            selection_probs = [fitness / total_fitness for fitness in fitness_scores]
            
        # 选择个体
        for _ in range(len(population)):
            idx = self._roulette_wheel_selection(selection_probs)
            selected.append(population[idx].copy())
            
        return selected
        
    def _roulette_wheel_selection(self, probabilities: List[float]) -> int:
        """轮盘赌选择"""
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i
                
        # 如果因为浮点误差没有选择，返回最后一个
        return len(probabilities) - 1
        
    def _crossover(self, selected: List[Dict]) -> List[Dict]:
        """交叉操作"""
        offspring = []
        
        # 随机配对父代
        random.shuffle(selected)
        
        # 对每对父代进行交叉
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                # 单点交叉
                child1, child2 = self._single_point_crossover(parent1, parent2)
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # 如果剩余一个没配对的个体，直接加入下一代
                offspring.append(selected[i].copy())
                
        return offspring
        
    def _single_point_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """单点交叉"""
        child1 = {}
        child2 = {}
        
        # 获取所有参数名
        param_names = list(parent1.keys())
        
        if not param_names:
            return parent1.copy(), parent2.copy()
            
        # 选择交叉点
        crossover_point = random.randint(1, len(param_names) - 1)
        
        # 执行交叉
        for i, param in enumerate(param_names):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
                
        return child1, child2
        
    def _mutation(self, population: List[Dict], parameters: Dict[str, Dict]) -> List[Dict]:
        """变异操作"""
        # 变异率
        mutation_rate = 0.1
        
        for individual in population:
            for param_name, param_value in individual.items():
                # 使用变异率决定是否变异
                if random.random() < mutation_rate:
                    param_config = parameters[param_name]
                    param_type = param_config.get('type', 'float')
                    
                    if param_type == 'float':
                        min_val = param_config.get('min', 0.0)
                        max_val = param_config.get('max', 1.0)
                        individual[param_name] = random.uniform(min_val, max_val)
                        
                    elif param_type == 'int':
                        min_val = param_config.get('min', 0)
                        max_val = param_config.get('max', 10)
                        individual[param_name] = random.randint(min_val, max_val)
                        
                    elif param_type == 'choice':
                        choices = param_config.get('choices', [])
                        if choices:
                            individual[param_name] = random.choice(choices)
                            
        return population
        
    def get_best_strategy(self, strategy_id: str) -> Dict:
        """获取策略的最佳参数"""
        if strategy_id not in self.best_strategies:
            return {
                'status': 'error',
                'message': f"策略 '{strategy_id}' 未优化或不存在"
            }
            
        return {
            'status': 'success',
            'strategy_id': strategy_id,
            'parameters': self.best_strategies[strategy_id]['parameters'],
            'fitness': self.best_strategies[strategy_id]['fitness'],
            'optimization_time': self.best_strategies[strategy_id]['optimization_time']
        }
        
    def get_optimization_status(self) -> Dict:
        """获取当前优化状态"""
        if not self.current_optimization:
            return {
                'status': 'idle',
                'message': '没有正在进行的优化任务'
            }
            
        return self.current_optimization
        
    def get_all_strategies(self) -> List[str]:
        """获取所有注册的策略ID"""
        return list(self.strategies.keys())

