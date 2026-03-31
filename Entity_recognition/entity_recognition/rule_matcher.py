import re
import jieba.posseg as pseg
from typing import List, Dict, Any


class RuleBasedMatcher:
    """基于规则的实体匹配器"""

    def __init__(self, config):
        self.config = config
        self.patterns = self._build_medical_patterns()

    def _build_medical_patterns(self):
        """构建医疗领域专用匹配规则"""
        patterns = {
            "DISEASE": [
                # 疾病名称模式
                r"(患有|得了|诊断出|确诊为|得了)([\u4e00-\u9fa5]{2,6})(病|症|炎|癌|瘤|痛|症候群|综合征)",
                r"([\u4e00-\u9fa5]{2,6})(病|症|炎|癌|瘤|痛|症候群|综合征)的?(症状|治疗|原因|预防|检查)",
                r"(什么是|啥是|介绍一下)([\u4e00-\u9fa5]{2,6})(病|症|炎|癌|瘤)",
                r"([\u4e00-\u9fa5]{2,6})(病|症|炎|癌|瘤)怎么(治|治疗|处理)",
            ],
            "SYMPTOM": [
                # 症状模式
                r"(出现|有|感到|觉得|经常)([\u4e00-\u9fa5]{2,6})(痛|痒|麻|肿|晕|吐|泻|烧|热|咳|喘|血|炎)",
                r"([\u4e00-\u9fa5]{2,6})(疼痛|发痒|麻木|肿胀|头晕|呕吐|腹泻|发烧|发热|咳嗽|气喘|出血|发炎)",
                r"(头痛|胃痛|腹痛|腰痛|关节痛|胸闷|心悸|恶心|呕吐|腹泻|便秘|发烧|咳嗽|气喘|呼吸困难)",
            ],
            "DRUG": [
                # 药品模式
                r"(服用|吃|用了|注射|打)([\u4e00-\u9fa5]{2,8})(片|丸|胶囊|颗粒|口服液|注射液|膏|贴|剂)",
                r"([\u4e00-\u9fa5]{2,8})(霉素|沙星|洛尔|地平|普利|沙坦|他汀|西林|松|芬|酮|定)",
                r"(阿司匹林|青霉素|头孢|布洛芬|对乙酰氨基酚|胰岛素|二甲双胍|硝酸甘油|维生素)",
            ],
            "CHECK": [
                # 检查项目模式
                r"(做|进行|检查|化验|测)([\u4e00-\u9fa5]{2,8})(检查|检测|化验|扫描|B超|CT|MRI|X光|心电图)",
                r"([\u4e00-\u9fa5]{2,8})(检查|检测|化验|扫描|B超|CT|核磁共振|X光|心电图|超声)",
                r"(血常规|尿常规|肝功能|肾功能|血糖|血脂|血压|心电图|B超|CT|核磁共振|胃镜)",
            ],
            "DEPARTMENT": [
                # 科室模式
                r"(挂|看|去|就诊于|咨询)([\u4e00-\u9fa5]{2,4})(科|科室)",
                r"([\u4e00-\u9fa5]{2,4})(科|科室)的?(医生|主任|专家)",
                r"(内科|外科|儿科|妇科|眼科|耳鼻喉科|皮肤科|口腔科|神经科|心血管科|内分泌科)",
            ],
            "FOOD": [
                # 食物模式
                r"(吃|喝|食用|饮用)([\u4e00-\u9fa5]{2,6})(汤|粥|茶|水|果|菜|肉|鱼)",
                r"(苹果|香蕉|橙子|西瓜|米饭|面条|猪肉|鸡肉|鱼肉|牛奶|鸡蛋)",
            ]
        }

        return patterns

    def match(self, text: str) -> List[Dict[str, Any]]:
        """使用规则匹配实体"""
        entities = []

        for entity_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                try:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        matched_text = match.group()

                        # 清理匹配文本
                        cleaned_text = self._clean_matched_text(matched_text, entity_type)

                        if len(cleaned_text) >= 2:
                            entities.append({
                                'text': cleaned_text,
                                'type': entity_type,
                                'start': match.start(),
                                'end': match.end(),
                                'confidence': 0.8,
                                'source': 'rule',
                                'pattern': pattern
                            })
                except Exception as e:
                    continue  # 跳过有问题的模式

        return entities

    def _clean_matched_text(self, text: str, entity_type: str) -> str:
        """清理匹配到的文本，只提取括号中的实体部分"""
        import re

        # 匹配括号中的内容
        pattern_map = {
            "DISEASE": r"[\u4e00-\u9fa5]{2,6}(?=病|症|炎|癌|瘤|痛|症候群|综合征)",
            "SYMPTOM": r"[\u4e00-\u9fa5]{2,6}(?=痛|痒|麻|肿|晕|吐|泻|烧|热|咳|喘|血|炎)",
            "DRUG": r"[\u4e00-\u9fa5]{2,8}(?=霉素|沙星|洛尔|地平|普利|沙坦|他汀|西林|松|芬|酮|定)",
            "CHECK": r"[\u4e00-\u9fa5]{2,8}(?=检查|检测|化验|扫描|B超|CT|MRI|X光|心电图)",
            "DEPARTMENT": r"[\u4e00-\u9fa5]{2,4}(?=科|科室)",
            "FOOD": r"[\u4e00-\u9fa5]{2,6}(?=汤|粥|茶|水|果|菜|肉|鱼)"
        }

        if entity_type in pattern_map:
            match = re.search(pattern_map[entity_type], text)
            if match:
                return match.group()

        return text