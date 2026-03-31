from neo4j import GraphDatabase
import json
import os
from typing import List, Dict, Any
from Entity_recognition.config.entity_config import EntityConfig


class KnowledgeGraphExtractor:
    """从知识图谱提取实体数据"""

    def __init__(self, config: EntityConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )

    def extract_all_entities(self) -> Dict[str, List[str]]:
        """从知识图谱提取所有实体"""
        entities_dict = {}

        print("开始从知识图谱提取实体...")

        with self.driver.session() as session:
            for label in self.config.node_labels:
                try:
                    # 查询该标签的所有节点
                    query = f"MATCH (n:`{label}`) RETURN n.name AS name"
                    result = session.run(query)

                    entity_names = []
                    for record in result:
                        name = record["name"]
                        if name and isinstance(name, str) and len(name.strip()) >= 2:
                            entity_names.append(name.strip())

                    if entity_names:
                        # 使用英文类型作为键
                        eng_type = self.config.entity_type_mapping.get(label, label.upper())
                        entities_dict[eng_type] = entity_names
                        print(f"  {label}({eng_type}): 提取了 {len(entity_names)} 个实体")

                except Exception as e:
                    print(f"  提取 {label} 时出错: {e}")
                    continue

            # 在同一个会话中提取关系
            self._extract_relationships(session)

        print(f"实体提取完成，共提取 {sum(len(v) for v in entities_dict.values())} 个实体")

        return entities_dict

    def _extract_relationships(self, session):
        """提取关系信息用于规则匹配"""
        relationships = []

        # 从您的图中提取的关系列表
        relationship_types = [
            "acompany_with",  # 伴随
            "belongs_to",  # 属于
            "cure_department",  # 治疗科室
            "do_eat",  # 宜吃
            "has_common_drug",  # 常用药品
            "has_symptom",  # 有症状
            "need_check",  # 需要检查
            "not_eat",  # 忌吃
            "production",  # 生产
            "recommand_drug",  # 推荐药品
            "recommand_recipes"  # 推荐菜谱
        ]

        for rel_type in relationship_types:
            try:
                # 获取关系示例
                query = f"""
                MATCH (a)-[r:`{rel_type}`]->(b)
                RETURN a.name AS source, b.name AS target, type(r) AS relation
                """
                result = session.run(query)

                examples = []
                for record in result:
                    examples.append({
                        "source": record["source"],
                        "target": record["target"],
                        "relation": record["relation"]
                    })

                if examples:
                    relationships.append({
                        "type": rel_type,
                        "examples": examples
                    })

            except Exception as e:
                print(f"  提取关系 {rel_type} 时出错: {e}")

        # 保存关系信息
        rel_path = os.path.join(self.config.data_dir, "relationships.json")
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(relationships, f, ensure_ascii=False, indent=2)

        print(f"关系提取完成，保存到: {rel_path}")

    def extract_properties(self) -> Dict[str, List[str]]:
        """提取属性信息"""
        properties = {}

        with self.driver.session() as session:
            # 获取疾病属性示例
            query = """
            MATCH (d:疾病)
            WHERE d.desc IS NOT NULL OR d.cause IS NOT NULL
            RETURN d.name AS name, d.desc AS description, d.cause AS cause
            LIMIT 20
            """

            result = session.run(query)
            disease_props = []

            for record in result:
                disease_props.append({
                    "name": record["name"],
                    "description": record["description"],
                    "cause": record["cause"]
                })

            if disease_props:
                properties["disease_properties"] = disease_props

        return properties

    def close(self):
        """关闭数据库连接"""
        self.driver.close()