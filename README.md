# 超神量子系统

## 系统启动说明

本系统使用 `run.py` 作为唯一启动入口，请始终使用此文件启动系统。

### 基本用法

```bash
# 启动桌面模式（默认）
python run.py

# 启动控制台模式
python run.py --mode console

# 启动驾驶舱模式
python run.py --mode cockpit

# 启动服务器模式
python run.py --mode server
```

### 命令行参数

- `--mode` 或 `-m`: 选择运行模式
  - `desktop`: 桌面GUI模式（默认）
  - `console`: 控制台模式
  - `server`: 服务器模式
  - `cockpit`: 驾驶舱模式

- `--debug` 或 `-d`: 启用调试模式

- `--config` 或 `-c`: 指定配置文件路径
  - 默认: `config/system_config.json`

### 示例

```bash
# 启动桌面模式并开启调试
python run.py -d

# 启动控制台模式并使用自定义配置
python run.py -m console -c my_config.json
```

## 重要提示

请不要直接运行其他启动脚本（如 `launch_quantum_core.py`、`start_quantum.py` 等），这些脚本可能会在未来版本中发生变化。`run.py` 将保持稳定，作为系统的永久入口点。 