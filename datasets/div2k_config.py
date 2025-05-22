"""DIV2K数据集配置"""

class DIV2KConfig:
    """DIV2K数据集配置类"""
    # 基础URL
    BASE_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
    
    # 数据集类型
    TYPES = {
        "hr": {
            "train": ["DIV2K_train_HR.zip"],
            "valid": ["DIV2K_valid_HR.zip"]
        },
        "lr_bicubic": {
            "train": [
                "DIV2K_train_LR_bicubic_X2.zip",
                "DIV2K_train_LR_bicubic_X3.zip",
                "DIV2K_train_LR_bicubic_X4.zip"
            ],
            "valid": [
                "DIV2K_valid_LR_bicubic_X2.zip",
                "DIV2K_valid_LR_bicubic_X3.zip",
                "DIV2K_valid_LR_bicubic_X4.zip"
            ]
        },
        "lr_unknown": {
            "train": [
                "DIV2K_train_LR_unknown_X2.zip",
                "DIV2K_train_LR_unknown_X3.zip",
                "DIV2K_train_LR_unknown_X4.zip"
            ],
            "valid": [
                "DIV2K_valid_LR_unknown_X2.zip",
                "DIV2K_valid_LR_unknown_X3.zip",
                "DIV2K_valid_LR_unknown_X4.zip"
            ]
        },
        "lr_special": {
            "train": [
                "DIV2K_train_LR_x8.zip",
                "DIV2K_train_LR_mild.zip",
                "DIV2K_train_LR_difficult.zip",
                "DIV2K_train_LR_wild.zip"
            ],
            "valid": [
                "DIV2K_valid_LR_x8.zip",
                "DIV2K_valid_LR_mild.zip",
                "DIV2K_valid_LR_difficult.zip",
                "DIV2K_valid_LR_wild.zip"
            ]
        }
    }
    
    @classmethod
    def get_download_urls(cls, data_types=None, splits=None):
        """获取下载URL列表
        
        Args:
            data_types: 数据类型列表，可选值：['hr', 'lr_bicubic', 'lr_unknown', 'lr_special']
            splits: 数据集划分列表，可选值：['train', 'valid']
            
        Returns:
            下载URL列表
        """
        if data_types is None:
            data_types = list(cls.TYPES.keys())
        if splits is None:
            splits = ['train', 'valid']
            
        urls = []
        for dtype in data_types:
            if dtype not in cls.TYPES:
                print(f"Warning: Unknown data type '{dtype}', skipping...")
                continue
            for split in splits:
                if split not in cls.TYPES[dtype]:
                    print(f"Warning: Split '{split}' not available for type '{dtype}', skipping...")
                    continue
                urls.extend([cls.BASE_URL + filename 
                           for filename in cls.TYPES[dtype][split]])
        return urls
    
    @classmethod
    def get_all_filenames(cls):
        """获取所有文件名列表"""
        filenames = []
        for dtype in cls.TYPES.values():
            for split in dtype.values():
                filenames.extend(split)
        return filenames