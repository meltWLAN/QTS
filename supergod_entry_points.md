# 超神量子共生系统 - 入口文件清单

本文档整理了超神量子共生系统的各种启动入口点，便于用户快速定位所需功能。

## 主要入口文件

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| supergod_cockpit_launcher.py | 项目根目录 | 超神量子共生系统的**唯一**驾驶舱模式入口 |
| launch_supergod_desktop.py | 项目根目录 | 专用于启动桌面版的简化入口 |
| run_quantum_system.py | 项目根目录 | 量子系统主入口，支持回测模式和图形界面模式 |
| run_quantum_strategy.py | 项目根目录 | 量子策略回测启动脚本，用于运行量子爆发策略回测 |
| main_desktop.py | 项目根目录 | 量子共生网络桌面版入口 |

## 驾驶舱模式相关入口

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| supergod_cockpit_launcher.py | 项目根目录 | 唯一官方驾驶舱模式启动器，固定使用备份目录中的稳定版本 |
| SuperQuantumNetwork/backup_current/run_supergod.py | SuperQuantumNetwork/backup_current | 备份目录中的稳定版驾驶舱实现（通过唯一启动器使用） |

## 桌面模式相关入口

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| SuperQuantumNetwork/backup_current/run_desktop.py | SuperQuantumNetwork/backup_current | 备份目录中的桌面版启动器 |
| SuperQuantumNetwork/backup_current/supergod_desktop.py | SuperQuantumNetwork/backup_current | 备份目录中的桌面版实现 |

## 回测模式相关入口

| 文件名 | 路径 | 功能描述 |
|--------|------|----------|
| SuperQuantumNetwork/test_quantum_strategy.py | SuperQuantumNetwork | 量子爆发策略测试脚本 |
| SuperQuantumNetwork/test_enhanced_strategy.py | SuperQuantumNetwork | 量子爆发增强策略测试脚本 |
| quantum_backtest.py | 项目根目录 | 高级回测脚本，集成数据连接器和量子爆发策略 |
| enhanced_strategy_example.py | 项目根目录 | 量子爆发增强策略回测示例 |

## 推荐的入口点

根据不同需求，推荐使用以下入口点：

1. **驾驶舱模式** (首选交互方式)：
   ```bash
   python supergod_cockpit_launcher.py
   ```
   **这是唯一官方支持的驾驶舱模式启动方式**

2. **桌面版** (高级数据可视化)：
   ```bash
   cd SuperQuantumNetwork/backup_current && python run_desktop.py
   ```

3. **量子策略回测** (策略评估)：
   ```bash
   python run_quantum_strategy.py
   ```

4. **增强策略回测示例** (带示例数据)：
   ```bash
   python enhanced_strategy_example.py
   ```

## 注意事项

- 确保已安装所有必要的依赖，特别是PyQt5和数据分析相关库
- 驾驶舱模式是最稳定的交互方式，推荐日常使用
- 桌面版提供更多高级可视化功能，但可能需要更多系统资源
- 回测模式适用于策略评估和优化，不需要图形界面
