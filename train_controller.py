import os
import sys

# 添加父目录到sys.path以便导入train
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from config.experiment_config import ExperimentConfig
from experiment.experiment_runner import ExperimentRunner

if __name__ == '__main__':
    # 创建实验配置
    config = ExperimentConfig()
    
    # 创建实验运行器
    runner = ExperimentRunner(config)
    
    # 运行所有实验
    runner.run_experiments(train_model)