from typing import List, Dict, Any
from .dictionary_matcher import DictionaryMatcher
from .rule_matcher import RuleBasedMatcher
from Entity_recognition.config.entity_config import EntityConfig
import time


class EntityRecognizer:
    """主实体识别器

    注意：
    这个模块只负责“实体识别”，不在这里做实体链接。
    实体链接/规范化由独立的 Entity_linking.EntityNormalizer 在后续流程中完成。
    """

    def __init__(self, config: EntityConfig):
        self.config = config

        # 初始化识别模块
        self.dictionary_matcher = DictionaryMatcher(config)
        self.rule_matcher = RuleBasedMatcher(config)

        print("实体识别器初始化完成")

    def recognize(self, text: str, use_linking: bool = False) -> List[Dict[str, Any]]:
        """识别文本中的实体

        Args:
            text: 用户输入文本
            use_linking: 保留该参数是为了兼容旧调用；当前不在本模块做实体链接。

        Returns:
            原始实体候选列表，后续交给 EntityNormalizer 做规范化。
        """
        start_time = time.time()
        entities = []

        # 1. AC自动机词典匹配，优先级最高
        dict_entities = self.dictionary_matcher.match(text)
        entities.extend(dict_entities)

        # 2. 如果AC没有识别到实体，再启用jieba增强兜底
        # 这样就把jieba放在“AC自动机”和“系统简单保底识别”之间。
        if not dict_entities:
            jieba_entities = self.dictionary_matcher.match_by_jieba(text)
            entities.extend(jieba_entities)

        # 3. 规则匹配，用于补充疾病、症状、药品、检查、科室等模式
        rule_entities = self.rule_matcher.match(text)
        entities.extend(rule_entities)

        # 4. 去重和合并
        entities = self._deduplicate_entities(entities)

        processing_time = time.time() - start_time

        # 5. 添加元数据
        for entity in entities:
            entity['processing_time'] = processing_time
            entity['text_length'] = len(text)

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """去重和合并重叠实体"""
        if not entities:
            return []

        # 按起始位置排序；同起点时，优先长实体、再优先高置信度
        entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start']), -x.get('confidence', 0)))

        deduplicated = []

        for entity in entities:
            overlap_index = -1

            for i, existing in enumerate(deduplicated):
                # 判断是否重叠
                if not (entity['end'] <= existing['start'] or entity['start'] >= existing['end']):
                    overlap_index = i
                    break

            if overlap_index == -1:
                deduplicated.append(entity)
            else:
                existing = deduplicated[overlap_index]

                entity_len = entity['end'] - entity['start']
                existing_len = existing['end'] - existing['start']
                entity_conf = entity.get('confidence', 0)
                existing_conf = existing.get('confidence', 0)

                # 优先保留更长的实体；长度相同则保留置信度更高的
                if entity_len > existing_len or (entity_len == existing_len and entity_conf > existing_conf):
                    deduplicated[overlap_index] = entity

        deduplicated.sort(key=lambda x: x['start'])
        return deduplicated

    def close(self):
        """关闭资源。当前识别模块没有需要关闭的连接。"""
        pass
