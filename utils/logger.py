#!/usr/bin/env python3
"""
日志管理模块

提供结构化日志记录、训练监控和错误追踪功能
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback

class TrainingLogger:
    """训练专用日志记录器
    
    功能：
    - 结构化日志记录
    - 训练指标监控
    - 错误详细追踪
    - 多级别日志输出
    """
    
    def __init__(self, 
                 name: str = 'training',
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 console_output: bool = True):
        """初始化训练日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
            level: 日志级别
            console_output: 是否输出到控制台
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加文件处理器
        self._setup_file_handlers()
        
        # 添加控制台处理器
        if console_output:
            self._setup_console_handler()
            
        # 训练指标存储
        self.metrics_history = []
        self.current_epoch_metrics = {}
        
        self.info(f"日志系统初始化完成，日志目录: {self.log_dir}")
        
    def _setup_file_handlers(self):
        """设置文件处理器"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 主日志文件
        main_log_file = self.log_dir / f'{self.name}_{timestamp}.log'
        file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志文件
        error_log_file = self.log_dir / f'{self.name}_errors_{timestamp}.log'
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
        
        # 指标日志文件（JSON格式）
        self.metrics_file = self.log_dir / f'{self.name}_metrics_{timestamp}.json'
        
    def _setup_console_handler(self):
        """设置控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 控制台使用简化格式
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """记录一般信息"""
        self._log_with_context(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """记录警告信息"""
        self._log_with_context(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """记录错误信息"""
        if exception:
            # 记录详细的异常信息
            exc_info = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
            kwargs.update(exc_info)
            
        self._log_with_context(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """记录严重错误信息"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
        
    def _log_with_context(self, level: int, message: str, **kwargs):
        """带上下文信息的日志记录"""
        if kwargs:
            # 将额外信息格式化为JSON字符串
            context = json.dumps(kwargs, ensure_ascii=False, default=str)
            full_message = f"{message} | Context: {context}"
        else:
            full_message = message
            
        self.logger.log(level, full_message)
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        self.current_epoch_metrics = {
            'epoch': epoch,
            'total_epochs': total_epochs,
            'start_time': datetime.now().isoformat(),
            'train_metrics': {},
            'val_metrics': {}
        }
        
        self.info(f"开始训练 Epoch {epoch+1}/{total_epochs}")
        
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **metrics):
        """记录epoch结束"""
        self.current_epoch_metrics.update({
            'end_time': datetime.now().isoformat(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics
        })
        
        # 保存到历史记录
        self.metrics_history.append(self.current_epoch_metrics.copy())
        
        # 保存指标到文件
        self._save_metrics()
        
        self.info(
            f"Epoch {epoch+1} 完成",
            train_loss=train_loss,
            val_loss=val_loss,
            **metrics
        )
        
    def log_batch_metrics(self, batch_idx: int, total_batches: int, 
                         loss: float, **metrics):
        """记录batch指标"""
        if batch_idx % 10 == 0:  # 每10个batch记录一次
            self.debug(
                f"Batch {batch_idx+1}/{total_batches}",
                loss=loss,
                **metrics
            )
            
    def log_model_info(self, model, total_params: int, trainable_params: int):
        """记录模型信息"""
        model_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }
        
        self.info("模型信息", **model_info)
        
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        self.info("训练配置", config=config)
        
    def log_gpu_memory(self):
        """记录GPU内存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                
                self.debug(
                    "GPU内存使用情况",
                    allocated_gb=f"{memory_allocated:.3f}",
                    reserved_gb=f"{memory_reserved:.3f}"
                )
        except ImportError:
            pass
            
    def log_checkpoint_save(self, checkpoint_path: str, epoch: int, 
                           loss: float, is_best: bool = False):
        """记录检查点保存"""
        self.info(
            f"检查点已保存: {checkpoint_path}",
            epoch=epoch,
            loss=loss,
            is_best=is_best
        )
        
    def log_early_stopping(self, patience: int, epochs_no_improve: int):
        """记录早停信息"""
        self.warning(
            f"触发早停机制，{epochs_no_improve}/{patience} epochs无改善"
        )
        
    def log_training_complete(self, total_time: float, best_loss: float):
        """记录训练完成"""
        self.info(
            "训练完成",
            total_time_seconds=total_time,
            total_time_formatted=f"{total_time/3600:.2f}小时",
            best_loss=best_loss,
            total_epochs=len(self.metrics_history)
        )
        
    def _save_metrics(self):
        """保存指标到JSON文件"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.error(f"保存指标文件失败: {e}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取训练指标摘要"""
        if not self.metrics_history:
            return {}
            
        train_losses = [m.get('train_loss', 0) for m in self.metrics_history if 'train_loss' in m]
        val_losses = [m.get('val_loss', 0) for m in self.metrics_history if 'val_loss' in m]
        
        summary = {
            'total_epochs': len(self.metrics_history),
            'best_train_loss': min(train_losses) if train_losses else None,
            'best_val_loss': min(val_losses) if val_losses else None,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
        }
        
        return summary
        
    def create_training_report(self) -> Dict[str, Any]:
        """创建训练报告"""
        summary = self.get_metrics_summary()
        
        report = {
            'total_epochs': summary.get('total_epochs', 0),
            'best_train_loss': summary.get('best_train_loss', None),
            'best_val_loss': summary.get('best_val_loss', None),
            'final_train_loss': summary.get('final_train_loss', None),
            'final_val_loss': summary.get('final_val_loss', None),
            'log_dir': str(self.log_dir),
            'metrics_file': str(self.metrics_file),
            'metrics_history': self.metrics_history
        }
        
        return report


def setup_logging(log_dir: Optional[str] = None, 
                 level: int = logging.INFO,
                 console_output: bool = True) -> TrainingLogger:
    """快速设置训练日志
    
    Args:
        log_dir: 日志目录
        level: 日志级别
        console_output: 是否输出到控制台
        
    Returns:
        TrainingLogger实例
    """
    return TrainingLogger(
        name='training',
        log_dir=log_dir,
        level=level,
        console_output=console_output
    )


# 全局日志记录器实例
_global_logger: Optional[TrainingLogger] = None

def get_logger() -> TrainingLogger:
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger

def set_global_logger(logger: TrainingLogger):
    """设置全局日志记录器"""
    global _global_logger
    _global_logger = logger