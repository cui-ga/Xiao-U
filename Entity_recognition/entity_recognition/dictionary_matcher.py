import ahocorasick
import jieba
from typing import List, Dict, Any
import json
import os

from Entity_recognition.entity_recognition.kg_extractor import KnowledgeGraphExtractor


class DictionaryMatcher:
    """基于词典的实体匹配器"""

    def __init__(self, config, entity_dict_path=None):
        self.config = config

        # 加载实体词典
        if entity_dict_path and os.path.exists(entity_dict_path):
            with open(entity_dict_path, 'r', encoding='utf-8') as f:
                self.entity_dict = json.load(f)
        else:
            # 从知识图谱提取
            extractor = KnowledgeGraphExtractor(config)
            self.entity_dict = extractor.extract_all_entities()
            extractor.close()

        # 构建AC自动机
        self.automaton = self._build_automaton()

        # 加载jieba自定义词典
        self._load_jieba_dict()

    def _build_automaton(self):
        """构建AC自动机用于多模式匹配"""
        automaton = ahocorasick.Automaton()

        for entity_type, entities in self.entity_dict.items():
            for entity in entities:
                if entity and len(entity) >= self.config.min_entity_length:
                    automaton.add_word(entity, (entity, entity_type))

        automaton.make_automaton()
        print(f"AC自动机构建完成，包含 {len(automaton)} 个模式")
        return automaton

    def _load_jieba_dict(self):
        """为jieba加载自定义词典"""
        dict_lines = []

        for entity_type, entities in self.entity_dict.items():
            for entity in entities:
                if entity and len(entity) >= 2:
                    # 为医疗实体分配较高的词频
                    freq = 1000
                    # 根据实体类型分配词性
                    pos_map = {
                        "DISEASE": "n",  # 名词
                        "SYMPTOM": "n",  # 名词
                        "DRUG": "n",  # 名词
                        "CHECK": "n",  # 名词
                        "DEPARTMENT": "ns",  # 处所名词
                        "FOOD": "n",  # 名词
                        "COMPANY": "nt",  # 机构名词
                        "RECIPE": "n"  # 名词
                    }
                    pos = pos_map.get(entity_type, "n")
                    dict_lines.append(f"{entity} {freq} {pos}")

        # 写入临时文件
        temp_dict_path = os.path.join(self.config.cache_dir, "jieba_custom_dict.txt")
        with open(temp_dict_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dict_lines))

        # 加载到jieba
        jieba.load_userdict(temp_dict_path)

    def match(self, text: str) -> List[Dict[str, Any]]:
        """在文本中匹配实体"""
        entities = []

        if not hasattr(self, 'automaton'):
            return entities

        # 使用AC自动机进行匹配
        for end_index, (entity_text, entity_type) in self.automaton.iter(text):
            start_index = end_index - len(entity_text) + 1

            # 检查是否重叠
            overlap = False
            for existing in entities:
                if not (end_index < existing['start'] or start_index > existing['end'] - 1):
                    overlap = True
                    # 如果新实体更长，替换旧实体
                    if len(entity_text) > len(existing['text']):
                        entities.remove(existing)
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'start': start_index,
                            'end': end_index + 1,
                            'confidence': 0.95,
                            'source': 'dictionary'
                        })
                    break

            if not overlap:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'start': start_index,
                    'end': end_index + 1,
                    'confidence': 0.95,  # 词典匹配置信度高
                    'source': 'dictionary'
                })

        return entities