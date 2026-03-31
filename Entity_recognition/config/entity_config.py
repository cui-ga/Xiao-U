import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EntityConfig:
    """实体识别配置类"""
    # Neo4j数据库配置
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"

    # 知识图谱节点标签（从您的图中提取）
    node_labels: List[str] = None

    # 实体类型映射
    entity_type_mapping: Dict[str, str] = None

    # 路径配置
    data_dir: str = "data/entities"
    cache_dir: str = "cache/entities"

    # 阈值配置
    min_entity_length: int = 2
    min_confidence: float = 0.6

    def __post_init__(self):
        """初始化默认值"""
        if self.node_labels is None:
            # 从您的图中获取的节点标签
            self.node_labels = [
                "疾病",  # Disease
                "症状",  # Symptom
                "药品",  # Drug
                "检查",  # Check
                "科室",  # Department
                "食物",  # Food
                "药企",  # Company
                "菜谱"  # Recipe
            ]

        if self.entity_type_mapping is None:
            # 中文标签到英文类型的映射
            self.entity_type_mapping = {
                "疾病": "DISEASE",
                "症状": "SYMPTOM",
                "药品": "DRUG",
                "检查": "CHECK",
                "科室": "DEPARTMENT",
                "食物": "FOOD",
                "药企": "COMPANY",
                "菜谱": "RECIPE"
            }

        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)