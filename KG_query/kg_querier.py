import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from neo4j import GraphDatabase, Session

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kg_query.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class KnowledgeGraphQuerier:
    """知识图谱查询器 - 增强版"""

    def __init__(self, config: Dict[str, Any]):
        """初始化知识图谱查询器

        Args:
            config: 配置字典，包含neo4j连接信息
        """
        self.config = config
        self.driver = None
        self.cypher_templates = {}
        self.query_cache = {}
        self.query_count = 0
        self.cache_hits = 0
        self.cache_size = config.get('cache_size', 100)
        self.enable_debug = config.get('enable_debug', False)
        self.enable_fuzzy_match = config.get('enable_fuzzy_match', True)
        self.max_query_time = config.get('max_query_time', 10)  # 秒

        # Neo4j连接参数
        self.neo4j_uri = config.get('neo4j_uri', 'bolt://localhost:7687')
        self.neo4j_user = config.get('neo4j_user', 'neo4j')
        self.neo4j_password = config.get('neo4j_password', '12345678')
        self.database = config.get('database', 'neo4j')

        # 初始化连接
        self._connect_neo4j()
        self._load_cypher_templates()

    def _connect_neo4j(self) -> None:
        """连接到Neo4j数据库"""
        try:
            logger.info(f"正在连接 Neo4j: {self.neo4j_uri}")
            logger.info(f"数据库: {self.database}")

            # 创建驱动
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=30 * 60,  # 30分钟
                max_connection_pool_size=50,
                connection_acquisition_timeout=2 * 60,  # 2分钟
                encrypted=False
            )

            # 验证连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as connection_test")
                test_value = result.single()["connection_test"]

                if test_value == 1:
                    logger.info("✅ Neo4j连接成功")

                    # 获取知识图谱统计信息
                    self._get_graph_statistics()
                else:
                    logger.warning("⚠️ Neo4j连接测试返回异常值")

        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            self.driver = None
            raise Exception(f"无法连接到Neo4j数据库: {e}")

    def _get_graph_statistics(self) -> None:
        """获取知识图谱统计信息"""
        try:
            with self.driver.session(database=self.database) as session:
                # 查询节点标签
                result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                record = result.single()
                if record and "labels" in record:
                    logger.info(f"📊 知识图谱节点标签: {', '.join(record['labels'])}")

                # 查询关系类型
                result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as rel_types")
                record = result.single()
                if record and "rel_types" in record:
                    logger.info(f"📊 知识图谱关系类型: {', '.join(record['rel_types'])}")

                # 查询节点数量
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                logger.info(f"📊 知识图谱节点总数: {node_count}")

                # 查询关系数量
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                logger.info(f"📊 知识图谱关系总数: {rel_count}")

        except Exception as e:
            logger.warning(f"无法获取完整知识图谱统计信息: {e}")

    def _load_cypher_templates(self) -> None:
        """加载Cypher查询模板"""
        try:
            # 可能的模板文件路径
            template_paths = [
                os.path.join("KG_query", "data", "cypher_templates.json"),
                os.path.join("data", "cypher_templates.json"),
                "cypher_templates.json",
                os.path.join(os.path.dirname(__file__), "data", "cypher_templates.json")
            ]

            for path in template_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        self.cypher_templates = json.load(f)

                    logger.info(f"✅ 加载Cypher模板成功: {len(self.cypher_templates)} 个")
                    logger.debug(f"模板路径: {path}")
                    return

            # 如果找不到模板文件，创建默认模板
            logger.warning("⚠️ 未找到Cypher模板文件，将使用内置默认模板")
            self.cypher_templates = self._create_default_templates()

        except json.JSONDecodeError as e:
            logger.error(f"❌ Cypher模板文件JSON格式错误: {e}")
            self.cypher_templates = self._create_default_templates()
        except Exception as e:
            logger.error(f"❌ 加载Cypher模板失败: {e}")
            self.cypher_templates = self._create_default_templates()

    def _create_default_templates(self) -> Dict:
        """创建默认Cypher模板（如果找不到模板文件时使用）"""
        default_templates = {
            "疾病基本信息": {
                "description": "查询疾病的基本信息",
                "cypher": "MATCH (d:疾病 {name: $entity_name}) RETURN d.name AS 疾病名称, d.symptom AS 症状, d.cause AS 病因, d.cure_way AS 治疗方法, d.prevent AS 预防措施, d.cure_department AS 治疗科室, d.desc AS 疾病描述, d.cure_lasttime AS 治疗周期, d.cured_prob AS 治愈率, d.easy_get AS 易感人群, d.get_prob AS 发病率",
                "parameters": ["entity_name"],
                "result_format": "single"
            },
            "临床表现(病症表现)": {
                "description": "查询疾病的症状表现",
                "cypher": "MATCH (d:疾病 {name: $entity_name}) RETURN d.name AS 疾病名称, d.symptom AS 症状",
                "parameters": ["entity_name"],
                "result_format": "single"
            }
        }

        # 确保目录存在
        os.makedirs("KG_query/data", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # 保存默认模板
        default_path = "KG_query/data/cypher_templates.json"
        with open(default_path, 'w', encoding='utf-8') as f:
            json.dump(default_templates, f, ensure_ascii=False, indent=2)

        logger.info(f"📁 已创建默认模板文件: {default_path}")
        return default_templates

    def query_by_intent(self, intent: str, entities: List[Dict], query_text: str = "", use_cache: bool = True) -> Dict[
        str, Any]:
        """根据意图查询知识图谱

        Args:
            intent: 查询意图
            entities: 实体列表
            query_text: 原始查询文本
            use_cache: 是否使用缓存

        Returns:
            查询结果字典
        """
        self.query_count += 1

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return {
                "success": False,
                "error": "Neo4j连接未建立",
                "data": {},
                "execution_time": 0,
                "cache_hit": False
            }

        # 生成缓存键
        cache_key = self._generate_cache_key(intent, entities, query_text)

        if use_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            logger.debug(f"从缓存获取查询结果: {cache_key}")
            return {
                "success": True,
                "data": self.query_cache[cache_key],
                "execution_time": 0,
                "cached": True,
                "cache_hit": True
            }

        start_time = time.time()

        try:
            # 获取Cypher模板
            template = self.cypher_templates.get(intent)
            if not template:
                logger.warning(f"未找到意图 '{intent}' 的查询模板")
                return {
                    "success": False,
                    "error": f"未找到意图 '{intent}' 的查询模板",
                    "available_intents": list(self.cypher_templates.keys()),
                    "data": {},
                    "execution_time": time.time() - start_time,
                    "cache_hit": False
                }

            cypher_query = template.get("cypher", "").strip()
            parameters = template.get("parameters", [])
            result_format = template.get("result_format", "single")

            if not cypher_query:
                logger.error(f"意图 '{intent}' 的Cypher查询为空")
                return {
                    "success": False,
                    "error": f"意图 '{intent}' 的Cypher查询为空",
                    "data": {},
                    "execution_time": time.time() - start_time,
                    "cache_hit": False
                }

            # 构建查询参数
            query_params = self._build_query_params(intent, entities, query_text, parameters)

            if self.enable_debug:
                logger.debug(f"执行Cypher查询: {intent}")
                logger.debug(f"查询语句: {cypher_query}")
                logger.debug(f"查询参数: {query_params}")
                logger.debug(f"参数模板: {parameters}")
                logger.debug(f"实体列表: {entities}")

            # 执行查询
            with self.driver.session(database=self.database) as session:
                # 设置查询超时
                result = session.run(cypher_query, **query_params)

                # 根据结果格式处理
                if result_format == "list":
                    data = []
                    for record in result:
                        data.append(self._clean_record(dict(record)))
                else:  # single
                    record = result.single()
                    if record:
                        data = self._clean_record(dict(record))
                    else:
                        data = {}

                execution_time = time.time() - start_time

                # 如果查询时间过长，记录警告
                if execution_time > 5:  # 超过5秒
                    logger.warning(f"查询 '{intent}' 耗时较长: {execution_time:.3f}秒")
                else:
                    logger.info(f"查询 '{intent}' 耗时: {execution_time:.3f}秒")

                # 缓存结果
                if data and use_cache and execution_time < 0.5:  # 只缓存快速查询结果
                    self._add_to_cache(cache_key, data)

                return {
                    "success": True,
                    "data": data,
                    "execution_time": execution_time,
                    "cached": False,
                    "cache_hit": False
                }

        except Exception as e:
            error_msg = f"Cypher查询执行失败: {str(e)}"
            logger.error(f"{error_msg} - 意图: {intent}, 参数: {entities}")

            # 尝试模糊匹配
            if self.enable_fuzzy_match and "entity_name" in intent.lower():
                fuzzy_result = self._try_fuzzy_match(intent, entities, query_text)
                if fuzzy_result["success"]:
                    return fuzzy_result

            return {
                "success": False,
                "error": error_msg,
                "data": {},
                "execution_time": time.time() - start_time,
                "cache_hit": False
            }

    def _clean_record(self, record: Dict) -> Dict:
        """清理记录，处理特殊值"""
        cleaned = {}
        for key, value in record.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, list):
                # 处理列表中的None值
                cleaned[key] = [item for item in value if item is not None]
            else:
                cleaned[key] = value
        return cleaned

    def _try_fuzzy_match(self, intent: str, entities: List[Dict], query_text: str) -> Dict[str, Any]:
        """尝试模糊匹配实体"""
        if not entities or len(entities) == 0:
            return {"success": False, "error": "无实体进行模糊匹配"}

        entity = entities[0]
        entity_name = entity.get('entity_name') or entity.get('text', '')

        if not entity_name:
            return {"success": False, "error": "实体名称为空"}

        logger.info(f"尝试对实体 '{entity_name}' 进行模糊匹配")

        # 尝试模糊查询实体
        fuzzy_queries = [
            ("疾病模糊查询", "MATCH (d:疾病) WHERE d.name CONTAINS $keyword RETURN d.name LIMIT 5"),
            ("症状模糊查询", "MATCH (s:症状) WHERE s.name CONTAINS $keyword RETURN s.name LIMIT 5"),
            ("药品模糊查询", "MATCH (d:药品) WHERE d.name CONTAINS $keyword RETURN d.name LIMIT 5"),
            ("通用模糊查询", "MATCH (n) WHERE n.name CONTAINS $keyword RETURN labels(n)[0] AS label, n.name AS name LIMIT 5")
        ]

        for query_name, cypher in fuzzy_queries:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(cypher, keyword=entity_name)
                    matches = [dict(record) for record in result]

                    if matches:
                        logger.info(f"模糊查询 '{query_name}' 找到 {len(matches)} 个匹配")
                        return {
                            "success": True,
                            "data": {
                                "original_entity": entity_name,
                                "fuzzy_matches": matches,
                                "query_type": query_name
                            },
                            "execution_time": 0,
                            "cached": False,
                            "cache_hit": False,
                            "fuzzy_match": True
                        }
            except Exception as e:
                logger.warning(f"模糊查询 '{query_name}' 失败: {e}")

        return {"success": False, "error": "模糊匹配未找到相关实体"}

    def _generate_cache_key(self, intent: str, entities: List[Dict], query_text: str) -> str:
        """生成缓存键"""
        entity_strs = []
        for entity in entities:
            # 提取实体的关键信息
            entity_name = entity.get('entity_name') or entity.get('text', '')
            entity_type = entity.get('type', '')
            if entity_name:
                entity_strs.append(f"{entity_type}:{entity_name}")

        # 添加查询文本
        if query_text:
            entity_strs.append(f"query:{query_text[:20]}")

        if not entity_strs:
            entity_strs.append("no_entity")

        return f"{intent}:{':'.join(sorted(entity_strs))}"

    def _build_query_params(self, intent: str, entities: List[Dict], query_text: str, parameters: List[str]) -> Dict[
        str, Any]:
        """构建查询参数"""
        params = {}

        # 提取实体信息
        entity_names = []
        entity_types = []

        for entity in entities:
            entity_name = entity.get('entity_name') or entity.get('text', '')
            entity_type = entity.get('type', '')

            if entity_name:
                entity_names.append(entity_name)
                entity_types.append(entity_type)

        # 根据参数名填充参数
        for param in parameters:
            if param == "entity_name" and entity_names:
                params["entity_name"] = entity_names[0]
            elif param == "entity_names" and entity_names:
                params["entity_names"] = entity_names
            elif param == "keyword" and query_text:
                # 用于模糊查询
                params["keyword"] = query_text
            elif param == "disease_name" and entity_names and ("疾病" in entity_types or not entity_types):
                params["disease_name"] = entity_names[0]
            elif param == "symptom_name" and entity_names and ("症状" in entity_types or not entity_types):
                params["symptom_name"] = entity_names[0]
            elif param == "start_name" and len(entity_names) >= 1:
                params["start_name"] = entity_names[0]
            elif param == "end_name" and len(entity_names) >= 2:
                params["end_name"] = entity_names[1]
            elif param == "query_text":
                params["query_text"] = query_text
            elif param == "limit":
                params["limit"] = 20
            else:
                # 如果没有匹配的参数，使用实体名称
                if entity_names and param not in params:
                    params[param] = entity_names[0]

        # 如果没有实体，尝试从查询文本中提取
        if not params.get("entity_name") and query_text and "entity_name" in parameters:
            # 简单提取，实际应用中可能需要更复杂的NLP处理
            params["entity_name"] = query_text.strip()

        return params

    def _add_to_cache(self, cache_key: str, data: Dict) -> None:
        """添加结果到缓存"""
        if len(self.query_cache) >= self.cache_size:
            # 简单LRU：移除第一个元素
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
            logger.debug(f"缓存已满，移除: {oldest_key}")

        self.query_cache[cache_key] = data
        logger.debug(f"添加到缓存: {cache_key}")

    def query_direct(self, cypher_query: str, params: Dict = None, timeout: int = 30) -> List[Dict]:
        """直接执行Cypher查询

        Args:
            cypher_query: Cypher查询语句
            params: 查询参数
            timeout: 超时时间（秒）

        Returns:
            查询结果列表
        """
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return []

        try:
            with self.driver.session(database=self.database) as session:
                if params:
                    result = session.run(cypher_query, **params)
                else:
                    result = session.run(cypher_query)

                return [self._clean_record(dict(record)) for record in result]
        except Exception as e:
            logger.error(f"直接查询执行失败: {e}")
            return []

    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """获取实体信息（自动检测实体类型）"""
        if not entity_name:
            return {"error": "实体名称为空"}

        results = {}

        # 尝试在不同类型的节点中查找
        entity_types = ["疾病", "症状", "药品", "食物", "菜谱", "检查", "科室", "药企"]

        for entity_type in entity_types:
            cypher = f"MATCH (n:{entity_type} {{name: $entity_name}}) RETURN n"
            data = self.query_direct(cypher, {"entity_name": entity_name})

            if data:
                results[entity_type] = data

        return results

    def search_entities(self, keyword: str, limit: int = 10) -> List[Dict]:
        """搜索实体（跨所有类型）"""
        if not keyword:
            return []

        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        RETURN labels(n)[0] AS label, n.name AS name, n
        LIMIT $limit
        """

        return self.query_direct(query, {"keyword": keyword, "limit": limit})

    def get_disease_network(self, disease_name: str) -> Dict[str, Any]:
        """获取疾病的关系网络"""
        query = """
        MATCH (d:疾病 {name: $disease_name})
        OPTIONAL MATCH (d)-[:has_symptom]->(s:症状)
        OPTIONAL MATCH (d)-[:recommand_drug]->(drug:药品)
        OPTIONAL MATCH (d)-[:acompany_with]->(comp:疾病)
        OPTIONAL MATCH (d)-[:not_eat]->(avoid:食物)
        OPTIONAL MATCH (d)-[:do_eat]->(good:食物)
        OPTIONAL MATCH (d)-[:need_check]->(check:检查)
        OPTIONAL MATCH (d)-[:recommand_recipes]->(recipe:菜谱)
        OPTIONAL MATCH (d)-[:cure_department]->(dept:科室)
        RETURN d.name AS 疾病名称,
               collect(DISTINCT s.name) AS 症状,
               collect(DISTINCT drug.name) AS 推荐药品,
               collect(DISTINCT comp.name) AS 并发症,
               collect(DISTINCT avoid.name) AS 忌食,
               collect(DISTINCT good.name) AS 宜食,
               collect(DISTINCT check.name) AS 相关检查,
               collect(DISTINCT recipe.name) AS 推荐食谱,
               collect(DISTINCT dept.name) AS 科室
        """

        result = self.query_direct(query, {"disease_name": disease_name})
        if result:
            return result[0]
        return {}

    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        stats = {}

        try:
            # 查询节点数量
            node_query = """
            CALL db.labels() YIELD label
            CALL {
              WITH label
              MATCH (n)
              WHERE label IN labels(n)
              RETURN count(n) as count
            }
            RETURN label, count
            ORDER BY label
            """
            nodes = self.query_direct(node_query)
            stats['node_counts'] = nodes

            # 查询关系数量
            rel_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL {
              WITH relationshipType
              MATCH ()-[r]->()
              WHERE type(r) = relationshipType
              RETURN count(r) as count
            }
            RETURN relationshipType, count
            ORDER BY relationshipType
            """
            relationships = self.query_direct(rel_query)
            stats['relationship_counts'] = relationships

            # 总节点数
            total_nodes = sum(item['count'] for item in nodes)
            # 总关系数
            total_rels = sum(item['count'] for item in relationships)

            stats['total_nodes'] = total_nodes
            stats['total_relationships'] = total_rels
            stats['node_types'] = len(nodes)
            stats['relationship_types'] = len(relationships)

            logger.info(f"知识图谱统计: {stats['node_types']} 种节点类型, {stats['relationship_types']} 种关系类型")
            logger.info(f"总节点数: {total_nodes}, 总关系数: {total_rels}")

        except Exception as e:
            logger.error(f"获取知识图谱统计失败: {e}")
            stats['error'] = str(e)

        return stats

    def get_template_info(self) -> Dict[str, Any]:
        """获取模板信息"""
        return {
            "template_count": len(self.cypher_templates),
            "available_intents": list(self.cypher_templates.keys()),
            "cache_size": len(self.query_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits
        }

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if self.query_count == 0:
            return 0.0
        return self.cache_hits / self.query_count

    def clear_cache(self) -> None:
        """清除查询缓存"""
        self.query_cache.clear()
        self.query_count = 0
        self.cache_hits = 0
        logger.info("查询缓存已清空")

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                return test_value == 1
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

    def get_available_intents(self) -> List[str]:
        """获取可用的查询意图列表"""
        return list(self.cypher_templates.keys())

    def get_intent_description(self, intent: str) -> str:
        """获取意图的描述"""
        template = self.cypher_templates.get(intent, {})
        return template.get("description", "无描述")

    def execute_template(self, intent: str, params: Dict = None) -> Dict[str, Any]:
        """直接执行模板查询"""
        template = self.cypher_templates.get(intent)
        if not template:
            return {
                "success": False,
                "error": f"未找到意图 '{intent}' 的查询模板"
            }

        cypher = template.get("cypher", "")
        result_format = template.get("result_format", "single")

        try:
            data = self.query_direct(cypher, params or {})

            if result_format == "single":
                result_data = data[0] if data else {}
            else:
                result_data = data

            return {
                "success": True,
                "data": result_data,
                "execution_time": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def close(self) -> None:
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("知识图谱查询器连接已关闭")


def test_querier():
    """测试查询器"""
    config = {
        'neo4j_uri': 'bolt://127.0.0.1:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': '12345678',
        'database': 'neo4j',
        'enable_debug': True,
        'enable_fuzzy_match': True
    }

    querier = None
    try:
        querier = KnowledgeGraphQuerier(config)

        # 测试用例
        test_cases = [
            {
                "query": "糖尿病有什么症状？",
                "intent": "临床表现(病症表现)",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "糖尿病怎么治疗？",
                "intent": "治疗方法",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "高血压应该看什么科？",
                "intent": "治疗科室",
                "entities": [{"text": "高血压", "type": "疾病", "entity_name": "高血压"}]
            },
            {
                "query": "糖尿病可以吃什么？",
                "intent": "建议食物",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "糖尿病不能吃什么？",
                "intent": "食物禁忌",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "糖尿病推荐什么药？",
                "intent": "推荐药品",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "糖尿病需要做什么检查？",
                "intent": "相关检查",
                "entities": [{"text": "糖尿病", "type": "疾病", "entity_name": "糖尿病"}]
            },
            {
                "query": "胃痛是什么症状？",
                "intent": "症状信息",
                "entities": [{"text": "胃痛", "type": "症状", "entity_name": "胃痛"}]
            },
            {
                "query": "查询疾病列表",
                "intent": "疾病列表",
                "entities": []
            },
            {
                "query": "查询知识图谱概况",
                "intent": "知识图谱概况",
                "entities": []
            }
        ]

        print("=" * 80)
        print("知识图谱查询器测试")
        print("=" * 80)

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n测试 {i}: {test_case['query']}")
            print("-" * 40)

            result = querier.query_by_intent(
                intent=test_case['intent'],
                entities=test_case['entities'],
                query_text=test_case['query']
            )

            print(f"意图: {test_case['intent']}")
            print(f"成功: {result.get('success', False)}")
            if not result.get('success'):
                print(f"错误: {result.get('error', '未知错误')}")
            else:
                print(f"耗时: {result.get('execution_time', 0):.3f}秒")
                print(f"缓存命中: {result.get('cache_hit', False)}")

                data = result.get('data', {})
                if data:
                    print("查询结果:")
                    for key, value in data.items():
                        if isinstance(value, list):
                            items = [str(item) for item in value[:5]]  # 只显示前5个
                            display_value = f"[{', '.join(items)}]" + ("..." if len(value) > 5 else "")
                            print(f"  {key}: {display_value}")
                        elif isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print("数据: 无结果")

        # 测试模板信息
        template_info = querier.get_template_info()
        print(f"\n" + "=" * 80)
        print("模板信息:")
        print(f"  模板数量: {template_info['template_count']}")
        print(f"  总查询数: {template_info['total_queries']}")
        print(f"  缓存命中率: {template_info['cache_hit_rate']:.2%}")
        print(f"  可用意图: {len(template_info['available_intents'])} 个")

        # 显示前10个意图
        print("\n前10个查询意图:")
        for i, intent in enumerate(template_info['available_intents'][:10], 1):
            description = querier.get_intent_description(intent)
            print(f"  {i:2d}. {intent}: {description}")

        # 测试疾病网络查询
        print(f"\n" + "=" * 80)
        print("测试疾病网络查询:")

        test_diseases = ["脊疳", "小儿消化性溃疡", "真菌性角膜炎"]
        for disease in test_diseases:
            print(f"\n疾病网络: {disease}")
            network = querier.get_disease_network(disease)
            if network:
                print(f"  症状: {network.get('症状', [])[:3]}")
                print(f"  推荐药品: {network.get('推荐药品', [])[:3]}")
                print(f"  并发症: {network.get('并发症', [])[:3]}")
            else:
                print(f"  未找到疾病: {disease}")

        # 测试实体搜索
        print(f"\n" + "=" * 80)
        print("测试实体搜索:")

        search_keywords = ["糖尿", "胃", "头痛"]
        for keyword in search_keywords:
            print(f"\n搜索关键词: '{keyword}'")
            results = querier.search_entities(keyword, limit=3)
            for result in results:
                print(f"  {result.get('label', '未知')}: {result.get('name', '未知')}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if querier:
            querier.close()


if __name__ == "__main__":
    test_querier()