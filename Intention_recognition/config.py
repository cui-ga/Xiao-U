"""
医疗意图识别模型配置
"""
import os
from pathlib import Path

# ========== 1. 基础路径配置 ==========
# 项目根目录 (pythonProject1)
PROJECT_ROOT = Path(__file__).parent.parent
# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# ========== 2. 标签映射 (固定不变) ==========
# 您的13个医疗意图类别
LABEL_MAPPING = {
    "label2id": {
        "定义": 0,
        "病因": 1,
        "预防": 2,
        "临床表现(病症表现)": 3,
        "相关病症": 4,
        "治疗方法": 5,
        "所属科室": 6,
        "传染性": 7,
        "治愈率": 8,
        "禁忌": 9,
        "化验/体检方案": 10,
        "治疗时间": 11,
        "其他": 12
    },
    "id2label": {
        "0": "定义",
        "1": "病因",
        "2": "预防",
        "3": "临床表现(病症表现)",
        "4": "相关病症",
        "5": "治疗方法",
        "6": "所属科室",
        "7": "传染性",
        "8": "治愈率",
        "9": "禁忌",
        "10": "化验/体检方案",
        "11": "治疗时间",
        "12": "其他"
    }
}

# ========== 3. 模型训练配置 (针对MacBERT优化) ==========
MODEL_CONFIG = {
    # 使用哈工大发布的MacBERT模型，在中文任务上表现优异
    "model_name": "hfl/chinese-macbert-base",

    # 模型输出类别数 (13个意图)
    "num_labels": len(LABEL_MAPPING["label2id"]),

    # 文本处理参数
    "max_length": 128,  # 最大序列长度，适合医疗短文本
    "batch_size": 32,  # 批大小，平衡速度与内存

    # 优化器参数
    "learning_rate": 1e-5,  # 学习率，适合预训练模型微调
    "weight_decay": 0.01,  # 权重衰减，防止过拟合
    "warmup_ratio": 0.1,  # 学习率预热比例

    # 训练参数
    "epochs": 4,  # 训练轮数，从历史看4轮足够
    "random_seed": 42,  # 随机种子，确保可重复性
    "validation_split": 0.2,  # 验证集比例
}

# ========== 4. 路径配置 ==========
# 模型保存路径
MODEL_SAVE_DIR = Path(__file__).parent / "saved_models"
MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# 最佳模型保存路径
BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_model"

# 日志文件路径
LOG_FILE = Path(__file__).parent / "training.log"