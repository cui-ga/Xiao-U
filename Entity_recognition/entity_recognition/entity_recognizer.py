from typing import List, Dict, Any
from .dictionary_matcher import DictionaryMatcher
from .rule_matcher import RuleBasedMatcher
from .entity_linker import EntityLinker
from Entity_recognition.config.entity_config import EntityConfig
import time


class EntityRecognizer:
    """主实体识别器"""

    def __init__(self, config: EntityConfig):
        self.config = config

        # 初始化各模块
        self.dictionary_matcher = DictionaryMatcher(config)
        self.rule_matcher = RuleBasedMatcher(config)
        self.entity_linker = EntityLinker(config)

        print("实体识别器初始化完成")

    def recognize(self, text: str, use_linking: bool = True) -> List[Dict[str, Any]]:
        """识别文本中的实体"""
        start_time = time.time()
        entities = []

        # 1. 词典匹配
        dict_entities = self.dictionary_matcher.match(text)
        entities.extend(dict_entities)

        # 2. 规则匹配
        rule_entities = self.rule_matcher.match(text)
        entities.extend(rule_entities)

        # 3. 去重和合并
        entities = self._deduplicate_entities(entities)

        # 4. 实体链接
        if use_linking and entities:
            entities = self.entity_linker.link(entities, text)

        processing_time = time.time() - start_time

        # 添加元数据
        for entity in entities:
            entity['processing_time'] = processing_time
            entity['text_length'] = len(text)

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重和合并重叠实体"""
        if not entities:
            return []

        # 按起始位置排序
        entities.sort(key=lambda x: (x['start'], -x['end']))

        deduplicated = []
        prev_end = -1

        for entity in entities:
            # 检查是否重叠
            if entity['start'] < prev_end:
                # 重叠，选择置信度更高的
                if deduplicated and entity.get('confidence', 0) > deduplicated[-1].get('confidence', 0):
                    deduplicated[-1] = entity
                    prev_end = entity['end']
            else:
                deduplicated.append(entity)
                prev_end = entity['end']

        return deduplicated

    def close(self):
        """关闭资源"""
        self.entity_linker.close()