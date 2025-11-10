import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = './logs', level: int = logging.INFO):
    """设置日志配置"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 清除现有的处理器
    logging.getLogger().handlers.clear()
    
    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print(f"日志文件保存在: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    return logging.getLogger(name)

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'metrics_{timestamp}.log'
        
        self.setup_metrics_logger()
    
    def setup_metrics_logger(self):
        """设置指标日志记录器"""
        self.metrics_logger = logging.getLogger('metrics')
        self.metrics_logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not self.metrics_logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.metrics_logger.addHandler(handler)
    
    def log_metrics(self, epoch: int, metrics: dict):
        """记录训练指标"""
        metrics_str = f"Epoch {epoch:03d}"
        for key, value in metrics.items():
            metrics_str += f" - {key}: {value:.6f}"
        
        self.metrics_logger.info(metrics_str)
        print(metrics_str)