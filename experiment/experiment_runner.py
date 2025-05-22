import os
from tqdm import tqdm
from preprocess_edges import preprocess_edge_data

class ExperimentRunner:
    """
    实验运行器类，用于管理和执行实验流程
    """
    def __init__(self, config):
        """
        初始化实验运行器
        Args:
            config: ExperimentConfig实例，包含所有实验配置
        """
        self.config = config
        self.base_params = config.BASE_TRAIN_PARAMS
        self.training_runs = config.TRAINING_RUNS
    
    def preprocess_edge_data(self):
        """
        预处理所有实验所需的边缘检测数据
        """
        # 收集所有实验需要的边缘检测方法
        required_edge_methods = set()
        for run_config in self.training_runs:
            if run_config['edge_methods']:
                required_edge_methods.update(run_config['edge_methods'])
        
        required_edge_methods = list(required_edge_methods)
        if not required_edge_methods:
            return
        
        print(f"正在检查和预处理所需的边缘数据: {required_edge_methods}")
        
        # 处理训练数据的边缘
        print(f"正在处理训练数据的边缘检测，数据目录: {self.config.TRAIN_LR_DIR}...")
        preprocess_edge_data(
            input_dir=self.config.TRAIN_LR_DIR,
            edge_methods=required_edge_methods,
            device=self.base_params['device']
        )
        
        # 如果存在验证数据目录，处理验证数据的边缘
        if os.path.exists(self.config.VAL_LR_DIR):
            print(f"正在处理验证数据的边缘检测，数据目录: {self.config.VAL_LR_DIR}...")
            preprocess_edge_data(
                input_dir=self.config.VAL_LR_DIR,
                edge_methods=required_edge_methods,
                device=self.base_params['device']
            )
        else:
            print(f"警告: 未找到验证数据目录 {self.config.VAL_LR_DIR}，跳过验证数据的边缘预处理。")
    
    def run_experiments(self, train_func):
        """
        运行所有配置的实验
        Args:
            train_func: 训练函数，用于执行实际的模型训练
        """
        print("开始边缘检测实验...")
        
        # 预处理边缘数据
        self.preprocess_edge_data()
        
        # 执行每个实验配置
        for run_config in tqdm(self.training_runs, desc="执行实验", unit="experiment"):
            run_name = run_config['name']
            edge_methods = run_config['edge_methods']
            criterion = run_config['criterion']
            
            print(f"\n--- 正在运行实验: {run_name}，使用边缘检测方法: {edge_methods} ---")
            
            # 准备当前实验的训练参数
            current_train_params = self.base_params.copy()
            current_train_params['model_name'] = run_name
            current_train_params['edge_detection_methods'] = edge_methods
            current_train_params['criterion'] = criterion
            
            # 设置数据目录
            current_train_params['lr_data_dir'] = self.config.TRAIN_LR_DIR
            current_train_params['hr_data_dir'] = self.config.TRAIN_HR_DIR
            current_train_params['val_lr_data_dir'] = self.config.VAL_LR_DIR
            current_train_params['val_hr_data_dir'] = self.config.VAL_HR_DIR
            
            try:
                train_func(**current_train_params)
                print(f"实验 '{run_name}' 成功完成。")
            except Exception as e:
                print(f"运行实验 '{run_name}' 时发生错误: {e}")
        
        print("所有边缘检测实验已完成。")