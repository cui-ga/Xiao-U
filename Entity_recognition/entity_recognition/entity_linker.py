
from typing import List, Dict, Any
import Levenshtein
import os
import json

from neo4j import GraphDatabase


class EntityLinker:
    """实体链接器 - 链接到知识图谱标准实体"""

    def __init__(self, config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        self.entity_cache = self._load_entity_cache()

    def _load_entity_cache(self) -> Dict[str, List[Dict]]:
        """加载实体缓存"""
        cache_path = os.path.join(self.config.cache_dir, "entity_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def link(self, entities: List[Dict], text: str) -> List[Dict]:
        """链接实体到知识图谱"""
        linked_entities = []

        for entity in entities:
            linked_entity = self._link_single_entity(entity)
            if linked_entity:
                linked_entities.append(linked_entity)

        return linked_entities

    def _link_single_entity(self, entity: Dict) -> Dict:
        """链接单个实体"""
        entity_text = entity.get('text', '')
        entity_type = entity.get('type', '')

        if not entity_text or not entity_type:
            return entity

        # 1. 检查缓存
        cache_key = f"{entity_type}:{entity_text}"
        if cache_key in self.entity_cache:
            kg_entity = self.entity_cache[cache_key]
            entity['normalized_text'] = kg_entity.get('name', entity_text)
            entity['kg_id'] = kg_entity.get('id')
            entity['confidence'] = kg_entity.get('similarity', 0.9)
            entity['linked'] = True
            return entity

        # 2. 查询知识图谱
        kg_entity = self._query_kg(entity_text, entity_type)

        if kg_entity:
            # 更新缓存
            self.entity_cache[cache_key] = kg_entity
            self._save_entity_cache()

            entity['normalized_text'] = kg_entity.get('name', entity_text)
            entity['kg_id'] = kg_entity.get('id')
            entity['confidence'] = kg_entity.get('similarity', 0.9)
            entity['linked'] = True
        else:
            entity['linked'] = False
            entity['normalized_text'] = entity_text

        return entity

    def _query_kg(self, text: str, entity_type: str) -> Dict:
        """查询知识图谱"""
        # 将英文类型映射回中文标签
        type_mapping = {v: k for k, v in self.config.entity_type_mapping.items()}
        label = type_mapping.get(entity_type, "")

        if not label:
            return None

        try:
            with self.driver.session() as session:
                # 1. 精确匹配
                query = f"""
                MATCH (n:`{label}`)
                WHERE toLower(n.name) = toLower($name)
                RETURN n.name AS name, elementId(n) AS id, 1.0 AS similarity
                """

                result = session.run(query, name=text)
                record = result.single()

                if record:
                    return {
                        'name': record['name'],
                        'id': record['id'],
                        'similarity': 1.0
                    }

                # 2. 改进的模糊匹配
                # 使用CONTAINS预先筛选，返回30个候选实体
                query = f"""
                MATCH (n:`{label}`)
                WHERE n.name CONTAINS $name
                RETURN n.name AS name, elementId(n) AS id
                ORDER BY size(n.name)
                LIMIT 30
                """

                result = session.run(query, name=text)
                best_match = None
                best_similarity = 0

                for record in result:
                    kg_name = record['name']
                    if not kg_name:
                        continue

                    # 计算相似度
                    similarity = Levenshtein.ratio(text, kg_name)

                    if similarity > best_similarity and similarity > 0.7:
                        best_similarity = similarity
                        best_match = {
                            'name': kg_name,
                            'id': record['id'],
                            'similarity': similarity
                        }

                return best_match

        except Exception as e:
            print(f"查询知识图谱失败: {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        import Levenshtein

        # 基本编辑距离相似度
        levenshtein_sim = Levenshtein.ratio(text1, text2)

        # 包含关系加分
        contain_bonus = 0
        if text1 in text2 or text2 in text1:
            contain_bonus = 0.1

        # 长度相似度
        len_sim = 1 - abs(len(text1) - len(text2)) / max(len(text1), len(text2), 1)

        # 综合相似度
        total_similarity = levenshtein_sim * 0.6 + contain_bonus + len_sim * 0.3

        return min(total_similarity, 1.0)

    def _save_entity_cache(self):
        """保存实体缓存"""
        cache_path = os.path.join(self.config.cache_dir, "entity_cache.json")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.entity_cache, f, ensure_ascii=False, indent=2)

    def close(self):
        """关闭数据库连接"""
        self.driver.close()