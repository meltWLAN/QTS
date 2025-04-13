"""
event_system - 量子核心组件
超神事件系统模块 - 提供异步事件处理功能
"""

import logging
import asyncio
import threading
import queue
import time
import uuid
from typing import Dict, Any, Callable, List, Set

logger = logging.getLogger(__name__)

class QuantumEventSystem:
    """量子事件系统 - 处理系统内部事件流"""
    
    def __init__(self):
        self.subscribers = {}  # 事件类型 -> 回调函数集合
        self.event_queue = asyncio.Queue()
        self.is_running = False
        self.event_loop = None
        self.event_loop_thread = None
        logger.info("量子事件系统初始化完成")
        
    def start(self):
        """启动事件处理循环"""
        if self.is_running:
            logger.warning("事件系统已在运行中")
            return
            
        logger.info("启动事件处理循环")
        self.is_running = True
        
        # 在新线程中启动事件循环
        self.event_loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True
        )
        self.event_loop_thread.start()
        
    def stop(self):
        """停止事件处理循环"""
        if not self.is_running:
            return
            
        logger.info("停止事件处理循环")
        self.is_running = False
        
        if self.event_loop and self.event_loop.is_running():
            # 尝试优雅地停止事件循环
            asyncio.run_coroutine_threadsafe(
                self._shutdown(), self.event_loop
            )
            
        if self.event_loop_thread and self.event_loop_thread.is_alive():
            self.event_loop_thread.join(timeout=2)
            
    def subscribe(self, event_type: str, callback: Callable):
        """订阅特定类型的事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()
            
        self.subscribers[event_type].add(callback)
        logger.debug(f"已订阅事件：{event_type}")
        return True
        
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅特定类型的事件"""
        if event_type not in self.subscribers:
            return False
            
        if callback not in self.subscribers[event_type]:
            return False
            
        self.subscribers[event_type].remove(callback)
        logger.debug(f"已取消订阅事件：{event_type}")
        return True
        
    def emit_event(self, event_type: str, event_data: Any):
        """发送事件"""
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'data': event_data,
            'timestamp': time.time()
        }
        
        if self.event_loop and self.is_running:
            # 将事件添加到队列中
            asyncio.run_coroutine_threadsafe(
                self.event_queue.put(event), self.event_loop
            )
            logger.debug(f"已发送事件：{event_type}")
            return event['id']
        else:
            logger.warning(f"事件系统未运行，无法发送事件：{event_type}")
            return None
            
    def _run_event_loop(self):
        """运行事件循环"""
        # 创建新的事件循环
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        # 启动事件处理协程
        self.event_loop.create_task(self._process_events())
        
        try:
            self.event_loop.run_forever()
        except Exception as e:
            logger.error(f"事件循环异常：{str(e)}")
        finally:
            self.event_loop.close()
            logger.info("事件循环已关闭")
            
    async def _process_events(self):
        """处理事件队列中的事件"""
        logger.info("开始处理事件")
        
        while self.is_running:
            try:
                # 获取事件，最多等待0.1秒
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # 处理事件
                await self._handle_event(event)
                
                # 标记任务完成
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"处理事件时出错：{str(e)}")
                await asyncio.sleep(0.1)  # 避免CPU过载
                
        logger.info("事件处理循环结束")
        
    async def _handle_event(self, event):
        """处理单个事件"""
        event_type = event['type']
        
        if event_type not in self.subscribers or not self.subscribers[event_type]:
            logger.debug(f"没有订阅者处理事件类型：{event_type}")
            return
            
        # 创建任务列表
        tasks = []
        for callback in list(self.subscribers[event_type]):
            if asyncio.iscoroutinefunction(callback):
                # 异步回调
                task = asyncio.create_task(callback(event['data']))
            else:
                # 同步回调，在线程池中运行
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, callback, event['data'])
                
            tasks.append(task)
            
        # 等待所有任务完成，忽略异常
        if tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.ALL_COMPLETED
            )
            
            # 检查是否有任务抛出异常
            for task in done:
                try:
                    task.result()
                except Exception as e:
                    logger.error(f"事件处理回调出错：{str(e)}")
                    
    async def _shutdown(self):
        """关闭事件循环"""
        logger.info("正在关闭事件循环...")
        
        # 取消所有挂起的任务
        tasks = [task for task in asyncio.all_tasks(self.event_loop) 
                if task is not asyncio.current_task(self.event_loop)]
                
        for task in tasks:
            task.cancel()
            
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # 停止事件循环
        self.event_loop.stop()

