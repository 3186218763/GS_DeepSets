import os
import logging


def setup_logging():
    """配置 logging"""
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)  # 创建日志文件夹，如果不存在的话

    # 创建一个 logger 实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置 logger 的日志级别为 DEBUG

    # 设置输出到 info.log 文件的 Handler
    info_name = "info.log"
    info_log_file = os.path.join(log_dir, info_name)
    info_handler = logging.FileHandler(info_log_file, mode='a', encoding='utf-8')
    info_handler.setLevel(logging.INFO)  # 设置 Handler 的日志级别为 INFO
    info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(info_handler)

    # 设置输出到 debug.log 文件的 Handler
    debug_name = "debug.log"
    debug_log_file = os.path.join(log_dir, debug_name)
    debug_handler = logging.FileHandler(debug_log_file, mode='a', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)  # 设置 Handler 的日志级别为 DEBUG
    debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(debug_handler)

    # 设置输出到 error.log 文件的 Handler
    error_name = "error.log"
    error_log_file = os.path.join(log_dir, error_name)
    error_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)  # 设置 Handler 的日志级别为 DEBUG
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(error_handler)

    return logger


logger = setup_logging()
if __name__ == '__main__':
    # 在需要的时候调用 setup_logging 函数进行日志配置
    logger = setup_logging()
    logger.info("dd")
