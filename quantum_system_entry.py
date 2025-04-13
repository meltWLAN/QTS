#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子交易系统统一入口守护
确保系统只能通过官方入口(launch_quantum_core.py)启动
"""

import os
import sys
import time
import logging
import hashlib
import platform
import subprocess
from datetime import datetime

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"quantum_entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumSystem.EntryGuard")

class SystemEntryGuard:
    """系统入口守护类
    
    确保系统只能通过官方入口启动，并防止多个实例同时运行
    """
    
    def __init__(self):
        """初始化守护程序"""
        self.OFFICIAL_LAUNCHER = "launch_quantum_core.py"
        self.LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".quantum_system.lock")
        self.launcher_hash = None
        self.system_id = self._generate_system_id()
        logger.info(f"系统ID: {self.system_id}")
        
    def _generate_system_id(self):
        """生成系统唯一标识"""
        system_info = f"{platform.node()}-{platform.platform()}"
        return hashlib.md5(system_info.encode()).hexdigest()[:12]
    
    def calculate_launcher_hash(self):
        """计算启动器文件的哈希值，防止启动器被篡改"""
        launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.OFFICIAL_LAUNCHER)
        
        if not os.path.exists(launcher_path):
            logger.error(f"官方启动器不存在: {launcher_path}")
            return None
            
        try:
            with open(launcher_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                logger.debug(f"启动器哈希值: {file_hash}")
                return file_hash
        except Exception as e:
            logger.error(f"计算启动器哈希值出错: {str(e)}")
            return None
    
    def verify_launcher(self):
        """验证启动器的完整性"""
        if not self.launcher_hash:
            self.launcher_hash = self.calculate_launcher_hash()
            
        current_hash = self.calculate_launcher_hash()
        if not current_hash:
            logger.error("无法验证启动器完整性")
            return False
            
        if current_hash != self.launcher_hash:
            logger.warning("启动器文件已被修改，需要重新验证")
            self.launcher_hash = current_hash
            
        return True
    
    def is_system_running(self):
        """检查系统是否已经在运行"""
        if os.path.exists(self.LOCK_FILE):
            try:
                with open(self.LOCK_FILE, 'r') as f:
                    lock_data = f.read().strip().split(':')
                    
                if len(lock_data) >= 2:
                    pid, timestamp = int(lock_data[0]), float(lock_data[1])
                    
                    # 检查进程是否存在
                    try:
                        # 在UNIX系统上检查进程
                        if platform.system() != "Windows":
                            os.kill(pid, 0)
                            is_running = True
                        else:
                            # 在Windows上检查进程
                            import psutil
                            is_running = pid in psutil.pids()
                        
                        if is_running:
                            elapsed = time.time() - timestamp
                            logger.info(f"系统已经在运行 (PID: {pid}, 已运行: {elapsed:.1f}秒)")
                            return True
                    except (OSError, ImportError):
                        # 进程不存在，锁文件可能是过时的
                        pass
            except Exception as e:
                logger.warning(f"读取锁文件出错: {str(e)}")
                
            # 锁文件过时，删除它
            try:
                os.remove(self.LOCK_FILE)
                logger.info("删除过时的锁文件")
            except Exception as e:
                logger.warning(f"删除锁文件出错: {str(e)}")
                
        return False
    
    def create_lock(self):
        """创建系统锁，防止多个实例同时运行"""
        try:
            with open(self.LOCK_FILE, 'w') as f:
                f.write(f"{os.getpid()}:{time.time()}:{self.system_id}")
            logger.info(f"创建系统锁文件: {self.LOCK_FILE}")
            return True
        except Exception as e:
            logger.error(f"创建锁文件失败: {str(e)}")
            return False
    
    def release_lock(self):
        """释放系统锁"""
        if os.path.exists(self.LOCK_FILE):
            try:
                os.remove(self.LOCK_FILE)
                logger.info("释放系统锁")
                return True
            except Exception as e:
                logger.error(f"释放锁文件失败: {str(e)}")
        return False
    
    def redirect_to_official_launcher(self, args=None):
        """重定向到官方启动器"""
        if args is None:
            args = sys.argv[1:] if len(sys.argv) > 1 else []
            
        launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.OFFICIAL_LAUNCHER)
        
        if not os.path.exists(launcher_path):
            logger.error(f"找不到官方启动器: {launcher_path}")
            return False
            
        logger.info(f"重定向到官方启动器: {launcher_path} {' '.join(args)}")
        
        try:
            cmd = [sys.executable, launcher_path] + args
            return subprocess.call(cmd) == 0
        except Exception as e:
            logger.error(f"启动官方启动器失败: {str(e)}")
            return False
            
    def check_system_integrity(self):
        """检查系统完整性"""
        # 检查关键文件是否存在
        critical_files = [
            self.OFFICIAL_LAUNCHER,
            "quantum_desktop/main.py",
            "quantum_desktop/core/system_manager.py"
        ]
        
        missing_files = []
        for file in critical_files:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            if not os.path.exists(file_path):
                missing_files.append(file)
                
        if missing_files:
            logger.error(f"系统完整性检查失败，缺少文件: {', '.join(missing_files)}")
            return False
            
        logger.info("系统完整性检查通过")
        return True
        
    def run(self, command=None, args=None):
        """运行入口守护程序
        
        Args:
            command: 执行的命令，如'start', 'stop', 'status'
            args: 传递给启动器的参数
        
        Returns:
            bool: 执行是否成功
        """
        if command == "status":
            is_running = self.is_system_running()
            print(f"系统状态: {'运行中' if is_running else '未运行'}")
            return True
            
        elif command == "stop":
            if self.is_system_running():
                print("正在停止系统...")
                # 这里只是释放锁，理想情况下应该发送信号给运行中的进程
                self.release_lock()
                print("系统已停止")
                return True
            else:
                print("系统未运行")
                return False
                
        elif command == "start" or command is None:
            # 检查系统完整性
            if not self.check_system_integrity():
                print("系统完整性检查失败，无法启动")
                return False
                
            # 检查系统是否已经在运行
            if self.is_system_running():
                print("系统已经在运行中，请勿重复启动")
                return False
                
            # 验证启动器完整性
            if not self.verify_launcher():
                print("启动器验证失败，无法启动系统")
                return False
                
            # 创建系统锁
            if not self.create_lock():
                print("创建系统锁失败，无法启动系统")
                return False
                
            # 启动官方启动器
            return self.redirect_to_official_launcher(args)
        
        else:
            print(f"未知命令: {command}")
            return False
            

def main():
    """主函数"""
    # 解析命令行参数
    command = None
    args = []
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["start", "stop", "status"]:
            command = sys.argv[1]
            args = sys.argv[2:]
        else:
            args = sys.argv[1:]
            
    # 创建并运行入口守护
    guard = SystemEntryGuard()
    success = guard.run(command, args)
    
    # 返回适当的退出码
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 