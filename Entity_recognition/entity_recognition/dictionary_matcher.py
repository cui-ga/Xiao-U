import ahocorasick
import jieba
from typing import List, Dict, Any
import json
import os
import re

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

        # 构建实体反查表：实体名 -> 实体类型，给jieba兜底使用
        self.entity_index = self._build_entity_index()

        # 构建AC自动机
        self.automaton = self._build_automaton()

        # 加载jieba自定义词典
        self._load_jieba_dict()

    def _build_entity_index(self) -> Dict[str, str]:
        """构建实体反查表，方便jieba兜底时判断分词结果是否为实体"""
        entity_index = {}

        for entity_type, entities in self.entity_dict.items():
            for entity in entities:
                if entity and len(entity) >= self.config.min_entity_length:
                    # 如果同一个实体出现在多个类型里，默认保留第一次出现的类型
                    if entity not in entity_index:
                        entity_index[entity] = entity_type

        return entity_index

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
                        "DISEASE": "n",      # 名词
                        "SYMPTOM": "n",      # 名词
                        "DRUG": "n",         # 名词
                        "CHECK": "n",        # 名词
                        "DEPARTMENT": "ns",  # 处所名词
                        "FOOD": "n",         # 名词
                        "COMPANY": "nt",     # 机构名词
                        "RECIPE": "n"        # 名词
                    }
                    pos = pos_map.get(entity_type, "n")
                    dict_lines.append(f"{entity} {freq} {pos}")

        # 写入临时文件
        temp_dict_path = os.path.join(self.config.cache_dir, "jieba_custom_dict.txt")
        os.makedirs(os.path.dirname(temp_dict_path), exist_ok=True)

        with open(temp_dict_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dict_lines))

        # 加载到jieba
        jieba.load_userdict(temp_dict_path)

    def match(self, text: str) -> List[Dict[str, Any]]:
        """在文本中使用AC自动机匹配实体"""
        entities = []

        if not text:
            return entities

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
                    # 如果新实体更长，替换旧实体（最长匹配优先）
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
                    'confidence': 0.95,
                    'source': 'dictionary'
                })

        return entities

    def match_by_jieba(self, text: str) -> List[Dict[str, Any]]:
        """
        使用jieba分词做增强兜底识别。

        说明：
        1. 只有上层在AC自动机没有识别到实体时才建议调用；
        2. 如果jieba切出的词在实体词典中，直接作为实体；
        3. 如果不在实体词典中，则根据医学常见表达和后缀猜测实体类型；
        4. 输出仍然是原始候选实体，后续交给独立的 Entity_linking.EntityNormalizer 规范化。
        """
        entities = []

        if not text:
            return entities

        tokens = []
        for word, start, end in jieba.tokenize(text):
            word = word.strip()
            if not word:
                continue
            tokens.append({
                'word': word,
                'start': start,
                'end': end
            })

        # 1. 单个分词候选
        for token in tokens:
            word = token['word']
            start = token['start']
            end = token['end']

            if self._is_noise_word(word):
                continue

            entity_type = ""
            confidence = 0.0

            # 分词结果刚好在实体词典中
            if word in self.entity_index:
                entity_type = self.entity_index[word]
                confidence = 0.75
            else:
                # 不在词典中，根据医学表达猜类型
                entity_type = self._guess_entity_type(word)
                confidence = 0.55

            if entity_type:
                entities.append({
                    'text': word,
                    'type': entity_type,
                    'start': start,
                    'end': end,
                    'confidence': confidence,
                    'source': 'jieba'
                })

        # 2. 组合相邻分词候选，例如“血糖 + 高”“肚子 + 疼”
        entities.extend(self._match_combined_jieba_tokens(tokens))

        # 3. 去重
        return self._deduplicate_jieba_entities(entities)

    def _match_combined_jieba_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合相邻jieba词，作为兜底候选实体"""
        entities = []

        if not tokens:
            return entities

        max_window = 3

        for i in range(len(tokens)):
            combined_word = ""
            start = tokens[i]['start']
            end = tokens[i]['end']

            for j in range(i, min(i + max_window, len(tokens))):
                # 只组合相邻或近似相邻的词，避免跨太远组合出奇怪短语
                if j > i and tokens[j]['start'] > end + 1:
                    break

                combined_word += tokens[j]['word']
                end = tokens[j]['end']

                if len(combined_word) < self.config.min_entity_length:
                    continue

                if self._is_noise_word(combined_word):
                    continue

                entity_type = ""
                confidence = 0.0

                if combined_word in self.entity_index:
                    entity_type = self.entity_index[combined_word]
                    confidence = 0.72
                else:
                    entity_type = self._guess_entity_type(combined_word)
                    confidence = 0.52

                if entity_type:
                    entities.append({
                        'text': combined_word,
                        'type': entity_type,
                        'start': start,
                        'end': end,
                        'confidence': confidence,
                        'source': 'jieba_combined'
                    })

        return entities

    def _guess_entity_type(self, word: str) -> str:
        """根据词形和医学常见表达粗略猜测实体类型"""
        if not word:
            return ""

        clean_word = word.strip("，。！？；：、,.!?;: ")
        if not clean_word or len(clean_word) < self.config.min_entity_length:
            return ""

        # 常见疾病/状态口语表达。这里给出类型只是为了让后续实体链接知道查哪个大类。
        disease_like_words = {
            "血糖高", "高血糖", "血压高", "高血压", "血脂高", "高血脂",
            "心梗", "脑梗", "甲亢", "甲减", "乙肝", "丙肝", "痛风"
        }
        if clean_word in disease_like_words:
            return "DISEASE"

        # 常见症状别称或口语表达
        symptom_words = {
            "发烧", "发热", "高烧", "低烧", "头疼", "头痛", "胃疼", "胃痛",
            "肚子疼", "肚子痛", "腹疼", "腹痛", "腰疼", "腰痛", "胸闷", "胸痛",
            "心慌", "心悸", "恶心", "呕吐", "腹泻", "拉肚子", "便秘", "咳嗽",
            "咳痰", "气喘", "呼吸困难", "流鼻涕", "鼻塞", "乏力", "头晕", "眩晕"
        }
        if clean_word in symptom_words:
            return "SYMPTOM"

        # 常见检查项目
        check_words = {
            "血常规", "尿常规", "肝功能", "肾功能", "血糖", "血脂", "血压",
            "心电图", "B超", "CT", "核磁共振", "磁共振", "胃镜", "肠镜",
            "彩超", "胸片", "X光"
        }
        if clean_word in check_words:
            return "CHECK"

        # 疾病类后缀
        if clean_word.endswith((
            "病", "症", "炎", "癌", "瘤", "综合征", "症候群",
            "感染", "结石", "中毒", "梗死", "硬化", "衰竭"
        )):
            return "DISEASE"

        # 症状类后缀
        if clean_word.endswith((
            "痛", "疼", "痒", "麻", "肿", "晕", "吐", "泻",
            "烧", "热", "咳", "喘", "血", "胀", "酸", "乏力"
        )):
            return "SYMPTOM"

        # 药品类后缀或常见药名结构
        if clean_word.endswith((
            "片", "丸", "胶囊", "颗粒", "口服液", "注射液",
            "膏", "贴", "剂", "霉素", "沙星", "洛尔", "地平",
            "普利", "沙坦", "他汀", "西林", "松", "芬", "酮", "定"
        )):
            return "DRUG"

        # 检查类后缀
        if clean_word.endswith((
            "检查", "检测", "化验", "扫描", "B超", "CT",
            "MRI", "X光", "心电图", "胃镜", "肠镜", "超声"
        )):
            return "CHECK"

        # 科室类后缀
        if clean_word.endswith(("科", "科室")):
            return "DEPARTMENT"

        # 食物类后缀
        if clean_word.endswith(("汤", "粥", "茶", "水", "果", "菜", "肉", "鱼", "奶", "蛋")):
            return "FOOD"

        return ""

    def _is_noise_word(self, word: str) -> bool:
        """过滤明显不是实体的词"""
        if not word:
            return True

        word = word.strip()

        if len(word) < self.config.min_entity_length:
            return True

        # 纯数字、纯字母、纯符号一般不作为实体
        if re.fullmatch(r"[0-9a-zA-Z_]+", word):
            return True

        if re.fullmatch(r"[\W_]+", word):
            return True

        stop_words = {
            "什么", "怎么", "怎样", "如何", "可以", "不能", "需要", "应该",
            "有没有", "是不是", "为什么", "哪个", "哪些", "多少", "一下",
            "这个", "这种", "那个", "那种", "患者", "医生", "医院", "治疗",
            "症状", "原因", "病因", "预防", "检查", "科室", "方法", "建议",
            "介绍", "知道", "帮我", "请问", "一下子", "吗", "呢", "吧"
        }

        if word in stop_words:
            return True

        return False

    def _deduplicate_jieba_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对jieba候选实体去重，保留更长、更高置信度的实体"""
        if not entities:
            return []

        entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start']), -x.get('confidence', 0)))

        deduplicated = []

        for entity in entities:
            overlap_index = -1

            for i, existing in enumerate(deduplicated):
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

                if entity_len > existing_len or (entity_len == existing_len and entity_conf > existing_conf):
                    deduplicated[overlap_index] = entity

        deduplicated.sort(key=lambda x: x['start'])
        return deduplicated
