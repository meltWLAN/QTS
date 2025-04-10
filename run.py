#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱模式启动器
"""

import sys
import logging
from supergod_cockpit import main

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("supergod_cockpit.log")
        ]
    )
    
    logger = logging.getLogger("SupergodLauncher")
    logger.info("启动超神量子共生系统 - 驾驶舱模式")
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断，正在安全退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        sys.exit(1) 