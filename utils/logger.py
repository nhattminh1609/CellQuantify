import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import config

def setup_logger(log_dir=config.LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("CellCounter")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:  # already configured (avoid duplicate log lines)
        return logger

    log_format = "[%(asctime)s.%(msecs)03d] [%(levelname)-8s] [%(threadName)-15s] [%(module)s:%(lineno)d] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    app_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

app_logger = logging.getLogger("CellCounter")