import os
import sys
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import Levenshtein
import re
from datetime import datetime


class EntityNormalizer:
    """实体链接/规范化模块 - 独立的处理步骤"""

    def __init__(self, config_path=None):
        """
        初始化实体规范化器

        Args:
            config_path: 配置文件路径或配置字典
        """
        # 项目根目录
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 加载配置
        self.config = self._load_config(config_path)

        # 连接知识图谱
        self.driver = self._connect_neo4j() if self.config.get('neo4j_uri') else None

        # 加载词典
        self.synonym_dict = self._load_synonym_dictionary()
        self.abbreviation_dict = self._load_abbreviation_dictionary()

        # 缓存
        self.cache = {}
        self.cache_file = os.path.join(self.project_root, 'Entity_linking', 'cache', 'entity_cache.pkl')
        self._load_cache()

        # 统计
        self.stats = {
            'total_processed': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'failed_matches': 0
        }

        print("实体链接/规范化模块初始化完成")
        print(f"  同义词词典: {len(self.synonym_dict)} 条")
        print(f"  缩写词典: {len(self.abbreviation_dict)} 条")
        print(f"  缓存大小: {len(self.cache)} 条")

    def _load_config(self, config_path):
        """加载配置"""
        default_config = {
            'neo4j_uri': 'bolt://127.0.0.1:7687',  # 改为 bolt 协议
            'neo4j_user': 'neo4j',
            'neo4j_password': '12345678',
            'neo4j_database': 'neo4j',
            'data_dir': os.path.join(self.project_root, 'Entity_linking', 'data'),
            'cache_dir': os.path.join(self.project_root, 'Entity_linking', 'cache'),
            'similarity_threshold': 0.8,
            'max_candidates': 20,
            'use_cache': True,
            'enable_debug': True
        }

        # 如果传入配置，合并
        if config_path and isinstance(config_path, dict):
            default_config.update(config_path)
        elif config_path and isinstance(config_path, str) and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            default_config.update(file_config)

        # 创建目录
        os.makedirs(default_config['data_dir'], exist_ok=True)
        os.makedirs(default_config['cache_dir'], exist_ok=True)

        return default_config

    def _connect_neo4j(self):
        """连接Neo4j"""
        try:
            uri = self.config['neo4j_uri']
            user = self.config['neo4j_user']
            password = self.config['neo4j_password']
            database = self.config.get('neo4j_database', 'neo4j')

            print(f"尝试连接 Neo4j:")
            print(f"  URI: {uri}")
            print(f"  用户: {user}")
            print(f"  数据库: {database}")

            driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                encrypted=False
            )

            # 测试连接
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value == 1:
                    print(f"✅ 成功连接到Neo4j数据库 '{database}'")
                    return driver
                else:
                    print(f"❌ Neo4j连接测试失败")
                    return None
        except Exception as e:
            print(f"❌ 连接Neo4j失败: {str(e)}")
            print("⚠️  将使用离线模式（仅使用本地词典）")
            return None

    def _load_synonym_dictionary(self) -> Dict[str, str]:
        """加载同义词词典"""
        synonym_path = os.path.join(self.config['data_dir'], 'synonyms.json')

        # 基础同义词词典
        synonyms = {
            "糖病": "糖尿病",
            "高血压病": "高血压",
            "冠心病": "冠状动脉性心脏病",
            "心梗": "心肌梗死",
            "脑梗": "脑梗死",
            "脑出血": "脑溢血",
            "高血糖": "糖尿病",
            "血癌": "白血病",
            "痨病": "肺结核",
            "慢阻肺": "慢性阻塞性肺疾病",
            "慢支": "慢性支气管炎",
            "心衰": "心力衰竭",
            "心绞痛": "冠心病",
            "中风": "脑卒中",
            "脑血栓": "脑梗死",
            "甲亢": "甲状腺功能亢进症",
            "甲减": "甲状腺功能减退症",
            "类风关": "类风湿关节炎",
            "痛风石": "痛风",
            "乙肝": "乙型病毒性肝炎",
            "丙肝": "丙型病毒性肝炎",
            "戊肝": "戊型病毒性肝炎",
            "水痘": "带状疱疹",
            "蛇胆疮": "带状疱疹",
            "飞蛇": "带状疱疹",
            "抑郁": "抑郁症",
            "焦虑": "焦虑症",
            "失眠": "睡眠障碍",
            "头疼": "头痛",
            "肚子疼": "腹痛",
            "拉肚子": "腹泻",
            "发高烧": "发热",
            "咳嗽咳痰": "咳嗽",
            "心慌": "心悸",
            "心口疼": "胸痛",
            "背心痛": "胸痛",
            "反酸水": "反酸",
            "烧心": "胃灼热",
            "打嗝": "嗳气",
            "打冷颤": "寒战",
            "怕冷": "畏寒",
            "没力气": "乏力",
            "出虚汗": "盗汗",
            "流鼻血": "鼻出血",
            "耳朵响": "耳鸣",
            "眼花": "视力模糊",
            "抽筋": "痉挛",
            "水肿": "浮肿",
            "扑热息痛": "对乙酰氨基酚",
            "安定": "地西泮",
            "心得安": "普萘洛尔",
            "开博通": "卡托普利",
            "络活喜": "氨氯地平",
            "拜新同": "硝苯地平控释片",
            "立普妥": "阿托伐他汀钙片",
            "格华止": "二甲双胍",
            "胰岛素针": "胰岛素注射液",
            "青霉素针剂": "青霉素",
            "头孢针": "头孢菌素",
            "阿莫西林胶囊": "阿莫西林",
            "布洛芬缓释胶囊": "布洛芬",
            "眼药水": "滴眼液",
            "阿司匹林片": "阿司匹林",
            "青霉素针": "青霉素",
            "头孢霉素": "头孢菌素",
            "CT扫描": "CT检查",
            "核磁": "磁共振成像",
            "心电图检查": "心电图",
            "拍片子": "X光检查",
            "做B超": "超声检查",
            "心脏彩超": "超声心动图",
            "做胃镜": "胃镜检查",
            "肠镜": "肠镜检查",
            "验血": "血液检查",
            "验尿": "尿液检查",
            "查血糖": "血糖检测",
            "量血压": "血压测量",
            "心内科": "心血管内科",
            "消化内科": "胃肠科",
            "神内": "神经内科",
            "脑外科": "神经外科",
            "肚子科": "消化内科",
            "不孕不育科": "生殖医学科",
            "肿瘤科": "肿瘤内科",
            "拍片室": "放射科"
        }

        if os.path.exists(synonym_path):
            try:
                with open(synonym_path, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                    if file_content:  # 检查文件是否非空
                        loaded_synonyms = json.loads(file_content)
                        # 合并，文件中的条目优先
                        synonyms.update(loaded_synonyms)
                        print(f"从文件加载同义词词典: {len(loaded_synonyms)} 条")
                        return synonyms
            except Exception as e:
                print(f"加载同义词词典文件失败: {e}, 将使用基础词典")

        # 保存基础词典
        with open(synonym_path, 'w', encoding='utf-8') as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)

        print(f"创建基础同义词词典: {synonym_path}")
        return synonyms

    def _load_abbreviation_dictionary(self) -> Dict[str, str]:
        """加载缩写词典"""
        abbreviation_path = os.path.join(self.config['data_dir'], 'abbreviations.json')

        # 基础缩写词典
        abbreviations = {
            "CT": "计算机断层扫描",
            "MRI": "磁共振成像",
            "B超": "B型超声",
            "X光": "X射线",
            "ECG": "心电图",
            "血糖": "血液葡萄糖",
            "血压": "动脉血压",
            "血脂": "血液脂质",
            "US": "超声检查",
            "PET-CT": "正电子发射计算机断层扫描",
            "DR": "数字化X射线摄影",
            "DSA": "数字减影血管造影",
            "EEG": "脑电图",
            "EMG": "肌电图",
            "PACS": "影像归档和通信系统",
            "WBC": "白细胞计数",
            "RBC": "红细胞计数",
            "Hb": "血红蛋白",
            "PLT": "血小板计数",
            "ALT": "丙氨酸氨基转移酶",
            "AST": "天冬氨酸氨基转移酶",
            "Cr": "肌酐",
            "BUN": "尿素氮",
            "HDL-C": "高密度脂蛋白胆固醇",
            "LDL-C": "低密度脂蛋白胆固醇",
            "CVD": "心血管疾病",
            "CHD": "冠心病",
            "COPD": "慢性阻塞性肺疾病",
            "DM": "糖尿病",
            "HTN": "高血压",
            "ICU": "重症监护室",
            "ER": "急诊室",
            "OTC": "非处方药",
            "Rx": "处方药",
            "PO": "口服",
            "IV": "静脉注射",
            "IM": "肌肉注射",
            "ASA": "阿司匹林",
            "NS": "生理盐水",
            "mg": "毫克",
            "g": "克",
            "ml": "毫升",
            "L": "升",
            "bid": "每日两次",
            "tid": "每日三次",
            "qd": "每日一次"
        }

        if os.path.exists(abbreviation_path):
            try:
                with open(abbreviation_path, 'r', encoding='utf-8') as f:
                    file_content = f.read().strip()
                    if file_content:  # 检查文件是否非空
                        loaded_abbreviations = json.loads(file_content)
                        # 合并，文件中的条目优先
                        abbreviations.update(loaded_abbreviations)
                        print(f"从文件加载缩写词典: {len(loaded_abbreviations)} 条")
                        return abbreviations
            except Exception as e:
                print(f"加载缩写词典文件失败: {e}, 将使用基础词典")

        # 保存基础词典
        with open(abbreviation_path, 'w', encoding='utf-8') as f:
            json.dump(abbreviations, f, ensure_ascii=False, indent=2)

        print(f"创建基础缩写词典: {abbreviation_path}")
        return abbreviations

    def normalize_entities(self, entities: List[Dict[str, Any]], query_text: str = "") -> List[Dict[str, Any]]:
        """
        规范化实体列表

        Args:
            entities: 来自实体识别模块的原始实体列表
            query_text: 原始查询文本（用于上下文）

        Returns:
            规范化后的实体列表
        """
        self.stats['total_processed'] += len(entities)
        normalized_entities = []

        for entity in entities:
            try:
                # 1. 规范化单个实体
                normalized_entity = self._normalize_single_entity(entity, query_text)

                if normalized_entity:
                    normalized_entities.append(normalized_entity)
                else:
                    # 规范化失败，保留原始实体但标记
                    entity['normalized'] = False
                    entity['normalization_error'] = '无法规范化'
                    normalized_entities.append(entity)
                    self.stats['failed_matches'] += 1

            except Exception as e:
                # 异常处理
                if self.config['enable_debug']:
                    print(f"规范化实体失败: {entity.get('text', '')} - {e}")
                entity['normalized'] = False
                entity['normalization_error'] = str(e)
                normalized_entities.append(entity)
                self.stats['failed_matches'] += 1

        return normalized_entities

    def _normalize_single_entity(self, entity: Dict[str, Any], context: str = "") -> Optional[Dict[str, Any]]:
        """
        规范化单个实体

        步骤：
        1. 文本清洗
        2. 同义词替换
        3. 缩写扩展
        4. 知识图谱链接
        5. 置信度计算
        """
        original_text = entity.get('text', '')
        entity_type = entity.get('type', '')

        if not original_text or not entity_type:
            return None

        # 1. 文本清洗
        cleaned_text = self._clean_entity_text(original_text)

        # 2. 同义词替换
        synonym_normalized = self._apply_synonyms(cleaned_text, entity_type)

        # 3. 缩写扩展
        expanded_text = self._expand_abbreviations(synonym_normalized)

        # 4. 知识图谱链接
        kg_result = self._link_to_knowledge_graph(expanded_text, entity_type, original_text)

        # 5. 构建规范化实体
        normalized_entity = entity.copy()

        if kg_result and kg_result.get('success'):
            # 成功链接到知识图谱
            normalized_entity.update({
                'normalized_text': kg_result['normalized_name'],
                'kg_id': kg_result.get('kg_id'),
                'kg_name': kg_result.get('normalized_name'),
                'similarity': kg_result.get('similarity', 0.0),
                'normalized': True,
                'normalization_method': kg_result.get('method', 'exact'),
                'original_text': original_text,
                'cleaned_text': cleaned_text,
                'synonym_normalized': synonym_normalized,
                'expanded_text': expanded_text
            })

            if kg_result.get('method') == 'exact':
                self.stats['exact_matches'] += 1
            else:
                self.stats['fuzzy_matches'] += 1

        else:
            # 链接失败，使用扩展后的文本
            normalized_entity.update({
                'normalized_text': expanded_text,
                'kg_id': None,
                'kg_name': expanded_text,
                'similarity': 0.0,
                'normalized': False,
                'normalization_method': 'dictionary_only',
                'original_text': original_text,
                'cleaned_text': cleaned_text,
                'synonym_normalized': synonym_normalized,
                'expanded_text': expanded_text,
                'normalization_error': kg_result.get('error', '未知错误') if kg_result else '无结果'
            })
            self.stats['failed_matches'] += 1

        # 添加时间戳
        normalized_entity['normalized_at'] = datetime.now().isoformat()

        return normalized_entity

    def _clean_entity_text(self, text: str) -> str:
        """清洗实体文本"""
        # 去除标点符号
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        # 去除多余空格
        text = ' '.join(text.split())
        return text.strip()

    def _apply_synonyms(self, text: str, entity_type: str) -> str:
        """应用同义词替换"""
        # 精确匹配
        if text in self.synonym_dict:
            return self.synonym_dict[text]

        # 尝试部分匹配
        for synonym, standard in self.synonym_dict.items():
            if synonym in text and len(synonym) >= 2:
                # 替换文本中的同义词
                text = text.replace(synonym, standard)
                return text

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """扩展缩写"""
        if text in self.abbreviation_dict:
            return self.abbreviation_dict[text]
        return text

    def _link_to_knowledge_graph(self, text: str, entity_type: str, original_text: str) -> Dict[str, Any]:
        """
        链接到知识图谱

        返回格式：
        {
            'success': bool,
            'normalized_name': str,
            'kg_id': str,
            'similarity': float,
            'method': str,  # 'exact', 'fuzzy', 'synonym', 'abbreviation'
            'error': str
        }
        """
        # 检查缓存
        cache_key = f"{entity_type}:{text}"
        if self.config['use_cache'] and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self.config.get('enable_debug', False):
                print(f"  调试: 从缓存获取: {cache_key}")
            return cached_result

        result = {
            'success': False,
            'normalized_name': text,
            'kg_id': None,
            'similarity': 0.0,
            'method': 'none',
            'error': '未连接知识图谱'
        }

        if not self.driver:
            result['error'] = '未连接知识图谱'
            if self.config.get('enable_debug', False):
                print(f"  调试: 无数据库连接，使用离线模式")
            return result

        try:
            # 创建会话时指定数据库
            database = self.config.get('neo4j_database', 'neo4j')

            if self.config.get('enable_debug', False):
                print(f"  调试: 尝试链接实体 '{text}' (类型: {entity_type}) 到数据库 '{database}'")

            with self.driver.session(database=database) as session:
                # 1. 精确匹配
                if self.config.get('enable_debug', False):
                    print(f"  调试: 尝试精确匹配...")

                exact_match = self._query_exact_match(session, text, entity_type)
                if exact_match:
                    if self.config.get('enable_debug', False):
                        print(f"  调试: 精确匹配成功: {exact_match['name']}")

                    result.update({
                        'success': True,
                        'normalized_name': exact_match['name'],
                        'kg_id': exact_match['id'],
                        'similarity': 1.0,
                        'method': 'exact',
                        'error': None
                    })
                    # 只在成功时缓存
                    self._update_cache(cache_key, result)
                    return result

                # 2. 模糊匹配
                if self.config.get('enable_debug', False):
                    print(f"  调试: 精确匹配失败，尝试模糊匹配...")

                fuzzy_match = self._query_fuzzy_match(session, text, entity_type)
                if fuzzy_match and fuzzy_match['similarity'] >= self.config['similarity_threshold']:
                    if self.config.get('enable_debug', False):
                        print(f"  调试: 模糊匹配成功: {fuzzy_match['name']} (相似度: {fuzzy_match['similarity']:.2f})")

                    result.update({
                        'success': True,
                        'normalized_name': fuzzy_match['name'],
                        'kg_id': fuzzy_match['id'],
                        'similarity': fuzzy_match['similarity'],
                        'method': 'fuzzy',
                        'error': None
                    })
                    # 只在成功时缓存
                    self._update_cache(cache_key, result)
                    return result

                # 3. 同义词匹配
                if self.config.get('enable_debug', False):
                    print(f"  调试: 模糊匹配失败，尝试同义词匹配...")

                synonym_match = self._query_by_synonyms(session, text, entity_type)
                if synonym_match:
                    if self.config.get('enable_debug', False):
                        print(f"  调试: 同义词匹配成功: {synonym_match['name']}")

                    result.update({
                        'success': True,
                        'normalized_name': synonym_match['name'],
                        'kg_id': synonym_match['id'],
                        'similarity': synonym_match.get('similarity', 0.9),
                        'method': 'synonym',
                        'error': None
                    })
                    # 只在成功时缓存
                    self._update_cache(cache_key, result)
                    return result

                result['error'] = '在知识图谱中未找到匹配实体'
                if self.config.get('enable_debug', False):
                    print(f"  调试: 所有匹配方法都失败")

        except Exception as e:
            error_msg = f"查询异常: {str(e)}"
            result['error'] = error_msg
            if self.config.get('enable_debug', False):
                print(f"  调试: 查询异常: {error_msg}")
            # 失败时不缓存，以便下次重试

        # 注意：失败时不缓存，只有成功时才缓存
        return result

    def _query_exact_match(self, session, text: str, entity_type: str) -> Optional[Dict]:
        """精确匹配查询"""
        label_map = {
            'DISEASE': '疾病',
            'SYMPTOM': '症状',
            'DRUG': '药品',
            'CHECK': '检查',
            'DEPARTMENT': '科室',
            'FOOD': '食物',
            'COMPANY': '药企',
            'RECIPE': '菜谱'
        }

        label = label_map.get(entity_type, entity_type)

        # 添加调试
        if self.config.get('enable_debug', False):
            print(f"    调试: 查询标签 '{label}' 中的实体 '{text}'")

        query = f"""
        MATCH (n:`{label}`)
        WHERE toLower(n.name) = toLower($name)
        RETURN n.name AS name, elementId(n) AS id
        LIMIT 1
        """

        if self.config.get('enable_debug', False):
            print(f"    调试: 执行查询: {query}")
            print(f"    调试: 参数: name={text}")

        try:
            result = session.run(query, name=text)
            record = result.single()

            if record:
                if self.config.get('enable_debug', False):
                    print(f"    调试: 找到记录: {record['name']}")
                return {'name': record['name'], 'id': record['id']}
            else:
                if self.config.get('enable_debug', False):
                    print(f"    调试: 未找到记录")

        except Exception as e:
            if self.config.get('enable_debug', False):
                print(f"    调试: 查询异常: {type(e).__name__}: {str(e)}")

        return None

    def _query_fuzzy_match(self, session, text: str, entity_type: str) -> Optional[Dict]:
        """模糊匹配查询"""
        label_map = {
            'DISEASE': '疾病',
            'SYMPTOM': '症状',
            'DRUG': '药品',
            'CHECK': '检查',
            'DEPARTMENT': '科室',
            'FOOD': '食物',
            'COMPANY': '药企',
            'RECIPE': '菜谱'
        }

        label = label_map.get(entity_type, entity_type)

        # 获取候选实体
        query = f"""
        MATCH (n:`{label}`)
        WHERE n.name CONTAINS $text OR $text CONTAINS n.name
        RETURN n.name AS name, elementId(n) AS id
        LIMIT {self.config['max_candidates']}
        """

        if self.config.get('enable_debug', False):
            print(f"    调试: 模糊查询: {query}")
            print(f"    调试: 参数: text={text}")

        result = session.run(query, text=text)
        best_match = None
        best_similarity = 0

        for record in result:
            kg_name = record['name']
            if not kg_name:
                continue

            similarity = Levenshtein.ratio(text, kg_name)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'name': kg_name,
                    'id': record['id'],
                    'similarity': similarity
                }

        if self.config.get('enable_debug', False) and best_match:
            print(f"    调试: 模糊匹配最佳: {best_match['name']} (相似度: {best_match['similarity']:.3f})")

        return best_match

    def _query_by_synonyms(self, session, text: str, entity_type: str) -> Optional[Dict]:
        """通过同义词查询"""
        # 获取可能的同义词
        synonyms = []
        for syn, std in self.synonym_dict.items():
            if text in syn or syn in text:
                synonyms.append(std)

        if not synonyms:
            if self.config.get('enable_debug', False):
                print(f"    调试: 无同义词可用")
            return None

        if self.config.get('enable_debug', False):
            print(f"    调试: 尝试同义词: {synonyms}")

        # 对每个同义词进行精确匹配
        for synonym in synonyms:
            exact_match = self._query_exact_match(session, synonym, entity_type)
            if exact_match:
                exact_match['similarity'] = 0.9  # 同义词匹配的置信度
                if self.config.get('enable_debug', False):
                    print(f"    调试: 同义词匹配成功: {synonym} -> {exact_match['name']}")
                return exact_match

        if self.config.get('enable_debug', False):
            print(f"    调试: 同义词匹配失败")

        return None

    def _calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """计算增强的相似度"""
        import Levenshtein

        # 1. 编辑距离相似度
        levenshtein_sim = Levenshtein.ratio(text1, text2)

        # 2. 包含关系加分
        contain_bonus = 0
        if text1 in text2 or text2 in text1:
            contain_bonus = 0.2

        # 3. 长度相似度
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2), 1)

        # 综合相似度
        total_similarity = (levenshtein_sim * 0.5 + contain_bonus + len_ratio * 0.3)

        return min(total_similarity, 1.0)

    def _update_cache(self, key: str, value: Dict):
        """更新缓存 - 只在成功时调用此方法"""
        self.cache[key] = value
        # 定期保存缓存
        if len(self.cache) % 100 == 0:
            self._save_cache()

    def _load_cache(self):
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                # 清除旧的错误缓存
                self._clean_error_cache()
            except Exception as e:
                print(f"加载缓存失败: {e}")
                self.cache = {}

    def _clean_error_cache(self):
        """清理错误缓存"""
        keys_to_remove = []
        for key, value in self.cache.items():
            if not value.get('success', False):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            print(f"清理了 {len(keys_to_remove)} 个错误缓存")

    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"保存缓存失败: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.stats['total_processed']
        if total == 0:
            success_rate = 0
        else:
            success_rate = (self.stats['exact_matches'] + self.stats['fuzzy_matches']) / total * 100

        return {
            'total_processed': total,
            'exact_matches': self.stats['exact_matches'],
            'fuzzy_matches': self.stats['fuzzy_matches'],
            'failed_matches': self.stats['failed_matches'],
            'success_rate': f"{success_rate:.2f}%",
            'cache_size': len(self.cache)
        }

    def close(self):
        """关闭资源"""
        self._save_cache()
        if self.driver:
            self.driver.close()
        print("实体链接/规范化模块已关闭")