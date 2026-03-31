import os
import sys
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MedicalQAIntegratedSystem:
    """医疗问答集成系统（优化版）"""

    def __init__(self, config_path: str = None):
        """
        初始化完整的医疗问答系统，集成所有模块，包括RAG和生成器。

        Args:
            config_path: 主配置文件路径。如果为None，则使用默认配置。
        """
        # 1. 确定项目根目录并加载配置
        self.project_root = Path(__file__).parent.parent
        self.config = self._load_config(config_path)

        # 2. 初始化各模块引用
        self.intent_recognizer = None
        self.entity_recognizer = None
        self.entity_normalizer = None
        self.kg_querier = None
        self.query_formatter = None
        self.rag_retriever = None
        self.rag_generator = None
        self.rag_enabled = False
        self.deepseek_enabled = False
        self.dialogue_manager = None
        self._initialize_dialogue_manager()

        # 3. 初始化统计信息
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'kg_only_answers': 0,
            'rag_only_answers': 0,
            'deepseek_answers': 0,
            'fallback_answers': 0,
            'system_intent_answers': 0
        }

        # 4. 超时设置
        self.kg_timeout = 5.0
        self.rag_timeout = 8.0
        self.deepseek_timeout = 30.0
        self.use_deepseek = True  # 控制开关

        # 5. 按顺序初始化所有模块
        logger.info("=" * 60)
        logger.info("开始初始化医疗问答集成系统（优化版）")
        logger.info("=" * 60)

        try:
            # 第一阶段：快速初始化核心模块
            self._initialize_core_modules()

            # 第二阶段：初始化RAG（较慢）
            self._initialize_rag_modules()

            # 第三阶段：初始化DeepSeek生成器
            self._initialize_deepseek_generator()

            logger.info("✅ 系统初始化完成")
            logger.info(f"   意图识别: {'✅' if self.intent_recognizer else '❌'}")
            logger.info(f"   实体识别: {'✅' if self.entity_recognizer else '❌'}")
            logger.info(f"   实体链接: {'✅' if self.entity_normalizer else '❌'}")
            logger.info(f"   KG查询器: {'✅' if self.kg_querier else '❌'}")
            logger.info(f"   RAG模块: {'✅' if self.rag_enabled else '❌'}")
            logger.info(f"   DeepSeek生成器: {'✅' if self.deepseek_enabled else '❌'}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            # 即使部分模块失败，也继续运行
            if not (self.kg_querier or self.rag_enabled):
                logger.error("⚠️ 核心模块（KG和RAG）都初始化失败，系统可能无法正常工作")
            else:
                logger.info("⚠️ 部分模块初始化失败，但系统仍可运行")

    def _initialize_dialogue_manager(self):
        """初始化对话管理器"""
        try:
            from QA_system.dialogue_manager import DialogueManager
            self.dialogue_manager = DialogueManager(self.config)
            if self.dialogue_manager.is_enabled():
                logger.info("✅ 对话管理器加载成功")
            else:
                logger.info("⚠️ 对话管理器已禁用")
        except ImportError as e:
            logger.warning(f"无法导入对话管理器: {e}，将禁用多轮对话功能")
            self.dialogue_manager = None
        except Exception as e:
            logger.error(f"初始化对话管理器失败: {e}")
            self.dialogue_manager = None

    def _initialize_core_modules(self):
        """初始化核心模块（快速）"""
        logger.info("阶段1: 初始化核心模块...")

        # 1. 意图识别
        try:
            intent_module_path = self.project_root / "Intention_recognition"
            if not intent_module_path.exists():
                logger.warning(f"意图识别模块目录不存在: {intent_module_path}")
                return

            sys.path.insert(0, str(intent_module_path))
            from predict_intention import MedicalIntentPredictor

            model_path = self.config["modules"]["intent_recognition"]["model_path"]
            if not Path(model_path).exists():
                logger.warning(f"意图识别模型路径不存在: {model_path}，将使用关键词匹配。")
                return

            self.intent_recognizer = MedicalIntentPredictor(model_path)
            logger.info(f"✅ 意图识别模块加载成功: {model_path}")
        except ImportError as e:
            logger.warning(f"无法导入MedicalIntentPredictor: {e}，将使用关键词匹配。")
        except Exception as e:
            logger.error(f"初始化意图识别模块失败: {e}")
            self.intent_recognizer = None

        # 2. 实体识别
        try:
            entity_config = self.config["modules"]["entity_recognition"]
            if not entity_config.get("enabled", True):
                logger.info("实体识别模块已禁用")
                return

            entity_recog_path = self.project_root / "Entity_recognition"
            if not entity_recog_path.exists():
                logger.warning(f"实体识别模块目录不存在: {entity_recog_path}")
                return

            sys.path.insert(0, str(entity_recog_path))
            from Entity_recognition.entity_recognition.entity_recognizer import EntityRecognizer
            from Entity_recognition.config.entity_config import EntityConfig

            config_params = entity_config.get("config_params", {})
            entity_config_obj = EntityConfig(**config_params)
            self.entity_recognizer = EntityRecognizer(entity_config_obj)
            logger.info("✅ 实体识别模块加载成功")
        except Exception as e:
            logger.error(f"❌ 实体识别模块初始化失败: {e}")
            self.entity_recognizer = None

        # 3. 实体链接
        try:
            linking_config = self.config["modules"]["entity_linking"]
            if not linking_config.get("enabled", True):
                logger.info("实体链接模块已禁用")
                return

            entity_linking_path = self.project_root / "Entity_linking"
            if not entity_linking_path.exists():
                logger.warning(f"实体链接模块目录不存在: {entity_linking_path}")
                return

            sys.path.insert(0, str(entity_linking_path))
            from Entity_linking.entity_normalizer import EntityNormalizer

            config_path = linking_config.get("config_path")
            if not config_path or not Path(config_path).exists():
                logger.warning(f"实体链接配置文件不存在: {config_path}")
                return

            self.entity_normalizer = EntityNormalizer(config_path)

            # 兼容性处理
            if not hasattr(self.entity_normalizer, 'normalize'):
                if hasattr(self.entity_normalizer, 'normalize_entities'):
                    self.entity_normalizer.normalize = self.entity_normalizer.normalize_entities
                else:
                    logger.warning("实体链接模块没有normalize或normalize_entities方法")

            logger.info(f"✅ 实体链接模块加载成功")
        except Exception as e:
            logger.error(f"❌ 实体链接模块初始化失败: {e}")
            self.entity_normalizer = None

        # 4. 知识图谱查询器
        try:
            kg_config = {
                'neo4j_uri': 'bolt://127.0.0.1:7687',
                'neo4j_user': 'neo4j',
                'neo4j_password': '12345678',
                'database': 'neo4j',
                'enable_debug': True,
                'enable_fuzzy_match': True
            }

            # 修改这一行：添加 KG_query. 前缀
            from KG_query.kg_querier import KnowledgeGraphQuerier
            self.kg_querier = KnowledgeGraphQuerier(kg_config)

            # 测试连接
            if hasattr(self.kg_querier, 'test_connection'):
                if self.kg_querier.test_connection():
                    logger.info(f"✅ 知识图谱查询器加载成功，连接测试通过")
                else:
                    logger.error(f"❌ 知识图谱连接测试失败")
                    self.kg_querier = None
            else:
                logger.info(f"✅ 知识图谱查询器加载成功（跳过连接测试）")

        except ImportError as e:
            logger.error(f"导入知识图谱查询模块失败: {e}")
            logger.error(f"请确保KG_query目录存在，并且kg_querier.py中有KnowledgeGraphQuerier类")
            self.kg_querier = None
        except Exception as e:
            logger.error(f"❌ 初始化知识图谱查询器失败: {e}")
            self.kg_querier = None

    def _initialize_rag_modules(self):
        """初始化RAG模块（较慢，可后台进行）"""
        logger.info("阶段2: 初始化RAG模块...")

        rag_config = self.config["modules"]["rag"]
        if not rag_config.get("enabled", True):
            logger.info("⚠️ RAG模块已禁用")
            return

        try:
            rag_module_path = self.project_root / "RAG"
            if not rag_module_path.exists():
                logger.warning(f"RAG模块目录不存在: {rag_module_path}")
                return

            sys.path.insert(0, str(rag_module_path))

            # 尝试导入RAG检索器
            try:
                from RAG.retriever.retriever import RAGRetriever

                rag_config_dict = {
                    'embedding_model': rag_config.get("embedding_model", "moka-ai/m3e-base"),
                    'reranker_model': rag_config.get("reranker_model", "BAAI/bge-reranker-base"),
                    'vector_db_path': rag_config.get("vector_db_path", str(self.project_root / "RAG" / "vector_db")),
                    'top_k_initial': rag_config.get("top_k_initial", 30),
                    'top_k_final': rag_config.get("top_k_final", 3),
                    'similarity_threshold': rag_config.get("similarity_threshold", 0.001),
                    'enable_reranking': rag_config.get("enable_reranking", True),
                    'max_rerank_docs': rag_config.get("max_rerank_docs", 8)
                }

                logger.info("正在初始化RAG检索器...")
                start_time = time.time()
                self.rag_retriever = RAGRetriever(rag_config_dict)
                init_time = time.time() - start_time
                logger.info(f"✅ RAG检索器初始化成功，耗时: {init_time:.2f}秒")

                # 预热检索器
                if rag_config.get("warmup_on_start", True):
                    self._warm_up_rag_models()

                self.rag_enabled = True
                logger.info("✅ RAG模块已启用")

            except ImportError as e:
                logger.error(f"无法导入RAGRetriever: {e}")
                self.rag_enabled = False

        except Exception as e:
            logger.error(f"❌ 初始化RAG模块失败: {e}")
            self.rag_enabled = False

    def _initialize_deepseek_generator(self):
        """初始化DeepSeek生成器（优化版）"""
        if not self.use_deepseek or not self.rag_enabled:
            logger.info("⚠️ DeepSeek生成器已禁用或RAG未启用")
            return

        logger.info("阶段3: 初始化DeepSeek生成器（优化版）...")

        try:
            from RAG.generator.deepseek_integration import DeepSeekGenerator

            deepseek_config = {
                'api_base': 'http://localhost:11434/v1',
                'api_key': 'ollama',
                'model': 'deepseek-r1:1.5b',
                'max_tokens': 1024,
                'temperature': 0.1,
                'top_p': 0.7,
                'timeout': 60,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.1,
                'max_retries': 2
            }

            logger.info(f"正在初始化DeepSeek生成器 (模型: {deepseek_config['model']})...")
            self.rag_generator = DeepSeekGenerator(deepseek_config)

            # 简化连接测试
            if self.rag_generator.test_connection():
                logger.info(f"✅ DeepSeek生成器初始化成功 (模型: {deepseek_config['model']})")
                self.deepseek_enabled = True
            else:
                logger.warning("⚠️ DeepSeek连接测试失败，但可能仍可使用")
                self.deepseek_enabled = True  # 不立即禁用

        except Exception as e:
            logger.error(f"DeepSeek生成器初始化失败: {e}")
            self.rag_generator = None
            self.deepseek_enabled = False

    def _warm_up_rag_models(self):
        """预热RAG模型"""
        if not self.rag_retriever:
            return

        logger.info("开始预热RAG模型...")
        try:
            warmup_start = time.time()
            warmup_results = self.rag_retriever.retrieve("预热", top_k=1)
            warmup_time = time.time() - warmup_start
            logger.info(f"✅ 检索器预热完成，耗时: {warmup_time:.2f}秒")
        except Exception as e:
            logger.warning(f"预热过程出现异常: {e}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载系统配置文件"""
        default_config = {
            "modules": {
                "intent_recognition": {
                    "enabled": True,
                    "model_path": str(self.project_root / "Intention_recognition" / "saved_models" / "best_model"),
                    "top_k": 1
                },
                "entity_recognition": {
                    "enabled": True,
                    "config_class": "Entity_recognition.config.entity_config.EntityConfig",
                    "config_params": {
                        "neo4j_uri": "bolt://127.0.0.1:7687",
                        "neo4j_user": "neo4j",
                        "neo4j_password": "12345678"
                    }
                },
                "entity_linking": {
                    "enabled": True,
                    "config_path": str(self.project_root / "Entity_linking" / "config.json")
                },
                "kg_query": {
                    "enabled": True,
                    "config_path": str(self.project_root / "KG_query" / "config.json")
                },
                "rag": {
                    "enabled": True,
                    "embedding_model": "moka-ai/m3e-base",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "vector_db_path": str(self.project_root / "RAG" / "vector_db"),
                    "top_k_initial": 20,
                    "top_k_final": 2,
                    "similarity_threshold": 0.001,
                    "enable_reranking": True,
                    "max_rerank_docs": 6,
                    "warmup_on_start": True
                }
            },
            "system": {
                "max_entities": 5,
                "timeout_seconds": 10,
                "log_level": "INFO"
            },
            "answer_logic": {
                "kg_min_length": 25,  # KG答案最小长度阈值
                "enable_fallback": True
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._deep_update(default_config, user_config)
                logger.info(f"已加载用户配置文件: {config_path}")
            except Exception as e:
                logger.error(f"加载用户配置文件失败，将使用默认配置: {e}")

        return default_config

    def _deep_update(self, base: Dict, update: Dict):
        """深度更新字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def _get_kg_answer_text(self, kg_result: Dict[str, Any], intent: str = None) -> tuple:
        """从KG结果中提取答案文本和是否有效的标志"""
        if not kg_result or not kg_result.get('success', False):
            logger.debug("KG查询失败或无结果")
            return "", False, 0

        kg_data = kg_result.get('data', {})
        if not kg_data:
            logger.debug("KG数据为空")
            return "", False, 0

        # 根据意图获取对应的字段名
        field_map = {
            "临床表现(病症表现)": "症状",
            "治疗方法": "治疗方法",
            "所属科室": "治疗科室",
            "化验/体检方案": "相关检查",
            "定义": "定义",
            "传染性": "传染性",
            "治愈率": "治愈率",
            "治疗时间": "治疗时间",
            "病因": "病因",
            "相关病症": "相关病症",
            "建议食物": "宜食食物",
            "食物禁忌": "忌食食物",
            "推荐药品": "推荐药品",
            "相关检查": "相关检查",
            "症状信息": "症状信息",
            "疾病列表": "疾病列表"
        }

        # 获取目标字段名
        target_field = field_map.get(intent, "result")
        result_value = kg_data.get(target_field, '')

        # 如果目标字段不存在，尝试从数据中找第一个非'疾病名称'的字段
        if not result_value and intent:
            for key, value in kg_data.items():
                if key != "疾病名称" and value:
                    result_value = value
                    break
            else:
                result_value = kg_data.get("result", '')

        # 处理不同类型的返回值
        if isinstance(result_value, list):
            if len(result_value) == 0:
                logger.debug(f"KG结果列表为空: 字段={target_field}, 意图={intent}")
                return "", False, 0
            # 过滤空值并连接
            filtered_values = [str(item) for item in result_value if item and str(item).strip()]
            if len(filtered_values) == 0:
                logger.debug(f"KG结果列表过滤后为空: 字段={target_field}, 意图={intent}")
                return "", False, 0
            answer_text = "、".join(filtered_values)
        elif isinstance(result_value, str):
            if not result_value.strip() or result_value in ["暂无相关信息", "未找到相关信息", "未知"]:
                logger.debug(f"KG结果字符串无效: {result_value}")
                return "", False, 0
            answer_text = result_value.strip()
        else:
            answer_text = str(result_value) if result_value else ""

        has_answer = len(answer_text.strip()) > 0
        answer_length = len(answer_text.strip())

        logger.debug(f"KG答案: 意图={intent}, 字段={target_field}, 长度={answer_length}, 内容={answer_text[:50]}...")

        return answer_text, has_answer, answer_length

    def _extract_rag_content(self, rag_results: List[Dict]) -> str:
        """从RAG结果中提取内容"""
        if not rag_results:
            logger.debug("RAG结果为空")
            return ""

        extracted_info = []
        for i, result in enumerate(rag_results[:2], 1):
            content = result.get('content', '')
            if not content or len(content) < 20:
                continue

            # 尝试提取问题和答案
            lines = content.split('\n')
            question = ""
            answer = ""

            for line in lines:
                line = line.strip()
                if line.startswith('问题：') and not question:
                    question = line[3:100]
                elif ('答案' in line or '回答' in line) and not answer:
                    if '】' in line:
                        answer = line.split('】', 1)[-1].strip()[:150]
                    elif ':' in line:
                        answer = line.split(':', 1)[-1].strip()[:150]
                    else:
                        answer = line[:150]
                    break

            if answer and len(answer) > 10:
                extracted_info.append(f"相关病例{i}: {answer}")
            elif content:
                snippet = content[:200] + ('...' if len(content) > 200 else '')
                extracted_info.append(f"医学信息{i}: {snippet}")

        if extracted_info:
            logger.debug(f"提取到 {len(extracted_info)} 条RAG信息")
            return "\n".join(extracted_info)

        logger.debug("未能从RAG结果中提取有效信息")
        return ""

    def _generate_answer_with_deepseek(self, kg_answer: str, rag_content: str, query: str) -> Optional[str]:
        """使用DeepSeek生成答案 - 优化版本"""
        if not self.rag_generator or not self.deepseek_enabled:
            logger.debug("DeepSeek生成器未启用")
            return None

        try:
            # 构建更详细的上下文
            context_parts = []

            # 添加KG知识
            if kg_answer and len(kg_answer.strip()) > 5:
                context_parts.append(f"【知识图谱信息】{kg_answer}")

            # 添加RAG信息
            if rag_content and len(rag_content.strip()) > 5:
                context_parts.append(f"【相关病例信息】{rag_content}")

            # 合并上下文
            context = "\n\n".join(context_parts) if context_parts else ""

            if not context:
                logger.debug("DeepSeek生成失败：上下文为空")
                return None

            # 构建更明确的提示词
            prompt = f"""请基于以下医学信息，专业、清晰、直接地回答用户的问题。
参考信息：
{context}

用户问题：{query}

请用中文提供结构清晰的回答，注意：
1. 直接回答问题，不要复述问题或参考信息
2. 分点说明（如需），但避免使用Markdown格式
3. 确保信息的准确性和安全性
4. 如果信息不足，可以补充通用的医学建议
5. 不要重复相同内容
6. 不要包含“【相关医学信息】”、“问题”、“答案”等内部标记，直接开始回答。

回答："""

            logger.info("正在使用DeepSeek生成答案...")
            start_time = time.time()

            # 使用生成器生成答案
            try:
                answer = self.rag_generator.generate(prompt, context, max_tokens=1024)
                generation_time = time.time() - start_time

                if answer and len(answer.strip()) > 30:  # 降低长度阈值
                    logger.info(f"✅ DeepSeek生成成功，长度: {len(answer)}，耗时: {generation_time:.2f}秒")
                    return answer.strip()
                else:
                    logger.warning(f"DeepSeek生成长度过短或无内容: {len(answer) if answer else 0}")
                    return None

            except Exception as e:
                logger.error(f"DeepSeek生成异常: {e}")
                return None

        except Exception as e:
            logger.error(f"DeepSeek生成过程异常: {e}")
            return None

    def _format_rag_fallback_answer(self, rag_content: str) -> str:
        """格式化RAG后备答案"""
        if not rag_content or len(rag_content.strip()) < 20:
            return "抱歉，暂时没有找到相关信息。"

        friendly_prefix = "这方面内容，小U正在努力学习，先给你相关病例，供你参考：\n\n"
        return friendly_prefix + rag_content

    def _check_system_intent(self, query):
        """检查是否是系统相关的意图"""
        system_intents = {
            "问候": [
                "你好", "您好", "hi", "hello", "嗨", "早上好", "下午好", "晚上好","哈喽","小u","小u小u",
                "你好，小u","您好小u","您好，小u","在吗","有人吗", "你好啊", "你好呀", "您好呀"
            ],
            "名字": [
                "你叫什么", "你是谁", "你的名字", "你叫啥", "你是",
                "叫什么名字", "称呼你什么", "怎么称呼你", "你叫什么名字"
            ],
            "功能": [
                "你能做什么", "你能干什么", "你有什么功能","你能为我做什么"
                "你可以帮我什么", "你会什么", "你的能力",
                "你可以做什么", "有什么功能", "你会干什么",
                "你有什么用", "你的作用", "你能帮什么忙"
            ],
            "感谢": [
                "谢谢", "多谢", "感谢", "谢谢你", "thx", "thanks",
                "感谢你", "谢谢您", "非常感谢", "十分感谢", "太感谢了"
            ],
            "结束": [
                "再见", "拜拜", "退出", "结束", "结束对话",
                "不聊了", "bye", "goodbye", "下次聊", "下次再说",
                "先这样", "先到这里", "结束咨询", "结束问诊"
            ],
            "状态": [
                "你好吗", "你怎么样", "在吗", "在不在", "忙不忙",
                "在忙吗", "有空吗", "在线吗", "有人吗", "在吗在吗"
            ],
            "帮助": [
                "帮助", "怎么用", "如何使用", "使用说明", "help",
                "使用方法", "使用指南", "怎么操作", "如何操作", "使用教程"
            ],
            "版本": [
                "你是什么版本", "版本号", "version", "什么版本",
                "当前版本", "版本信息", "你是哪个版本", "什么型号"
            ],
            "开发者": [
                "谁开发的", "开发者是谁", "谁创造的", "谁开发的你",
                "你的作者", "谁设计的", "谁创造的你", "你的创造者"
            ],
        }

        query_lower = query.lower().strip()
        for intent, keywords in system_intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        return None

    def _handle_system_intent(self, intent: str, query: str) -> str:
        """处理系统意图，返回预设回复"""

        responses = {
            "问候": [
                "你好！我是医疗助手小U，有什么医疗问题可以帮您解答？",
                "您好，我是小U，您的医疗健康助手。有什么可以帮您的吗？",
                "嗨！我是小U，随时为您提供医疗健康咨询。"
            ],
            "名字": [
                "我叫小U，是一个专业的医疗AI助手，很高兴为您服务！",
                "我是小U，您的医疗健康顾问，专门帮助解答医疗相关问题。",
                "我是小U，一个专注于医疗健康领域的AI助手。"
            ],
            "功能": [
                """我可以帮助您了解以下医疗健康信息：
    • 疾病症状与临床表现
    • 治疗方法与药物使用
    • 预防措施与健康管理
    • 就医注意事项
    • 饮食与生活建议
    • 以及各种医疗健康问题咨询""",
                """作为医疗AI助手，我的主要功能包括：
    1. 疾病症状分析
    2. 治疗方法咨询
    3. 预防保健建议
    4. 用药注意事项
    5. 就医指导
    6. 日常健康管理

    有什么具体的医疗问题，我都可以帮您解答！"""
            ],
            "感谢": [
                "不客气，这是我应该做的！有其他医疗问题也可以问我哦。",
                "很高兴能帮到您！祝您身体健康！",
                "谢谢您的认可！我会继续努力提供更好的医疗建议。"
            ],
            "结束": [
                "再见，祝您身体健康！有任何医疗问题随时可以找我。",
                "感谢咨询，小U随时为您服务！祝您健康！",
                "好的，再见！记得保持健康的生活方式哦！"
            ],
            "状态": [
                "我很好，随时准备为您提供医疗咨询！有什么问题吗？",
                "我状态很好，随时可以为您解答医疗健康问题。",
                "我在线，随时为您服务！"
            ],
            "帮助": [
                """使用帮助：
    1. 直接问我医疗问题，如"糖尿病有什么症状？"
    2. 我会结合知识图谱和医学文献为您解答
    3. 可以问疾病、症状、治疗、预防等各种医疗问题
    4. 如果需要更详细的信息，可以告诉我"详细一点"

    试试问我一个医疗问题吧！""",
                """如何使用我？
    • 直接提问：如"高血压怎么预防？"
    • 具体描述：如"我感冒了，应该吃什么药？"
    • 多轮对话：我会记住上下文
    • 追问细节：可以说"能再详细一点吗？"

    有什么具体的医疗问题吗？"""
            ],
            "版本": [
                "我是医疗问答系统小UV1.0，专注于医疗健康领域的AI助手。",
                "当前版本：医疗助手小UV1.0，持续优化中！",
                "版本号：小UV1.0，具备医疗知识图谱、文献检索和AI生成能力。"
            ],
            "开发者": [
                "我是由医疗AI团队开发的医疗健康助手，专门为解决医疗咨询问题而设计。",
                "我的开发团队专注于医疗AI技术，致力于提供准确、可靠的医疗健康建议。",
                "我是由专业的医疗AI团队开发的，结合了知识图谱、文献检索和AI生成技术。"
            ],
        }

        import random
        if intent in responses:
            return random.choice(responses[intent])
        else:
            return "我是医疗助手小U，有什么医疗问题可以帮您？"

    def _format_final_answer(self, raw_answer: str, answer_source: str, intent: str, query: str) -> str:
        """
        彻底移除答案中的内部处理标记，保持答案原有格式。
        优化版本：提高清理效率和覆盖更多标记模式。
        """
        if not raw_answer or raw_answer.strip() == "":
            return "抱歉，暂时没有找到相关信息。"

        import re

        formatted = raw_answer

        # 1. 首先处理最明显的重复块模式
        # 匹配模式：【相关医学信息】【相关病例信息】医学信息1: 问题：xxx 答案1...
        duplicate_block_pattern = r'【相关医学信息】.*?【相关病例信息】.*?医学信息\d+:.*?答案\d+.*?'
        formatted = re.sub(duplicate_block_pattern, '', formatted, flags=re.DOTALL)

        # 2. 移除所有方括号标记（包括可能的多行情况）
        bracket_patterns = [
            r'【相关医学信息】', r'【知识图谱信息】', r'【相关病例信息】',
            r'【新话题】', r'【回答】',
        ]

        for pattern in bracket_patterns:
            formatted = re.sub(pattern, '', formatted, flags=re.MULTILINE)

        # 3. 移除各种问答格式标记
        qa_patterns = [
            # 基础问答格式
            r'^问题[:：].*$',  # 以"问题："开头的行
            r'^答案\d*[:：].*$',  # 以"答案1："开头的行

            # Markdown加粗格式
            r'\*\*问题[:：].*?\*\*',  # **问题：xxx**
            r'\*\*答案\d*[:：]?\*\*',  # **答案1** 或 **答案：**

            # 带编号的信息标记
            r'医学信息\d+[:：].*$',
            r'相关病例\d+[:：].*$',
        ]

        for pattern in qa_patterns:
            formatted = re.sub(pattern, '', formatted, flags=re.MULTILINE)

        # 4. 移除DeepSeek生成的特殊标记
        deepseek_patterns = [
            r'用户问题[:：].*',
            r'用户查询[:：].*',
            r'参考信息[:：].*',
            r'上下文信息[:：].*',
            r'原始查询[:：].*',
            r'查询意图[:：].*',
        ]

        for pattern in deepseek_patterns:
            formatted = re.sub(pattern, '', formatted, flags=re.MULTILINE)

        # 5. 处理重复内容 - 这是关键优化
        # 如果内容有明显的重复段落，只保留第一个出现的版本
        lines = formatted.split('\n')
        if len(lines) > 3:  # 只有当有多行时才进行去重
            # 构建一个简单的重复检测
            seen_sentences = set()
            unique_lines = []

            for line in lines:
                line_strip = line.strip()
                if not line_strip or len(line_strip) < 10:  # 跳过空行和短行
                    unique_lines.append(line)
                    continue

                # 尝试提取句子的核心部分（移除数字、标点等）
                core_sentence = re.sub(r'\d+', '', line_strip)  # 移除数字
                core_sentence = re.sub(r'[^\w\u4e00-\u9fff]', '', core_sentence)  # 只保留中文和字母

                if core_sentence and core_sentence not in seen_sentences:
                    seen_sentences.add(core_sentence)
                    unique_lines.append(line)
                elif core_sentence and core_sentence in seen_sentences:
                    # 这是重复内容，跳过
                    continue
                else:
                    unique_lines.append(line)

            # 如果去重后行数显著减少，使用去重后的结果
            if len(unique_lines) < len(lines) * 0.7:  # 如果减少了30%以上
                formatted = '\n'.join(unique_lines)
            else:
                formatted = '\n'.join(lines)

        # 6. 清理多余空白和空行
        # 合并连续空格
        formatted = re.sub(r' +', ' ', formatted)
        # 合并多个连续空行为最多两个空行
        formatted = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted)
        # 移除首尾空白
        formatted = formatted.strip()

        # 7. 如果清理后结果为空，返回一个友好的消息
        if not formatted or len(formatted) < 10:  # 太短的答案
            # 尝试从原始答案中提取一个简短的版本
            if raw_answer and len(raw_answer) > 20:
                # 提取前200个字符，然后清理
                short_answer = raw_answer[:200]
                # 移除明显的内部标记
                short_answer = re.sub(r'【.*?】', '', short_answer)
                short_answer = re.sub(r'\*\*.*?\*\*', '', short_answer)
                short_answer = short_answer.strip()
                if short_answer and len(short_answer) > 20:
                    return short_answer + ("..." if len(short_answer) == 200 else "")

            return "暂时没有找到相关信息。"

        return formatted

    def process_query(self, user_query: str, session_id: str = None) -> Dict[str, Any]:
        """
        处理用户查询的完整流程（支持多轮对话）
        思维导图逻辑：
        1. 若KG答案>25字，直接输出KG答案
        2. 若KG答案不足(<25字)或者KG没有在知识图谱搜索到答案，启动RAG
        3. 若RAG中deepseek生成失败：
           - 若KG能在知识图谱查到答案就直接用KG的答案，不用管答案字数，直接输出
           - 若KG没有搜索到知识图谱的答案，就用RAG中的医疗库的答案，并加上友好提示
        4. 若RAG中deepseek生成成功，直接输出答案
        Args:
            user_query: 用户查询
            session_id: 会话ID，用于多轮对话。如果为None，使用默认会话ID

        Returns:
            处理结果字典
        """
        start_time = time.time()
        self.stats['total_queries'] += 1

        # 确定会话ID
        if session_id is None:
            session_id = self.config.get("dialogue", {}).get("default_session_id", "default_user")

        result = {
            'success': False,
            'query': user_query,
            'original_query': user_query,  # 保存原始查询
            'processed_query': user_query,  # 处理后的查询
            'answer': '抱歉，我暂时无法回答这个问题。',
            'raw_answer': '',  # 新增：保存原始答案
            'intent': None,
            'entities': [],
            'normalized_entities': [],
            'processing_steps': {},
            'errors': [],
            'warnings': [],
            'response_time': 0,
            'timestamp': datetime.now().isoformat(),
            'answer_source': 'unknown',
            'kg_used': False,
            'rag_used': False,
            'deepseek_used': False,
            'session_id': session_id,  # 添加会话ID
            'has_context': False,  # 是否有对话上下文
            'context_info': {}  # 上下文信息
        }

        logger.info(f"🔍 开始处理查询 [会话: {session_id}]: {user_query}")

        try:
            # ========== 第一步：先检查系统意图 ==========
            system_intent = self._check_system_intent(user_query)
            if system_intent:
                logger.info(f"🎯 检测到系统意图: {system_intent}")
                response = self._handle_system_intent(system_intent, user_query)

                result['success'] = True
                result['raw_answer'] = response
                result['intent'] = system_intent
                result['answer_source'] = 'system_intent'
                # 格式化答案
                result['answer'] = self._format_final_answer(
                    raw_answer=response,
                    answer_source='system_intent',
                    intent=system_intent,
                    query=user_query
                )
                self.stats['system_intent_answers'] += 1
                self.stats['successful_queries'] += 1

                # 如果是结束意图，结束对话
                if system_intent == "结束" and self.dialogue_manager:
                    self.dialogue_manager.end_dialogue(session_id)

                # 计算响应时间
                response_time = time.time() - start_time
                result['response_time'] = response_time

                logger.info(f"✅ 系统意图处理完成. 耗时: {response_time:.3f}s, 意图: {system_intent} [系统]")
                return result

            # ========== 第二步：对话管理预处理 ==========
            processed_query = user_query
            dialogue_state = None
            has_context = False

            if self.dialogue_manager and self.dialogue_manager.is_enabled():
                # 处理用户查询（指代消解）
                processed_query, dialogue_state = self.dialogue_manager.process_user_query(
                    session_id, user_query, system_intent
                )

                if processed_query != user_query:
                    logger.info(f"🔄 对话管理: 查询已重写 '{user_query}' -> '{processed_query}'")
                    result['processed_query'] = processed_query

                # 获取上下文信息
                context_info = self.dialogue_manager.get_dialogue_context(session_id)
                result['has_context'] = context_info.get('has_context', False)
                result['context_info'] = context_info

            # ========== 第三步：医疗问答流程 ==========
            # 步骤1: 意图识别（使用处理后的查询）
            intent_result = self._recognize_intent(processed_query)
            result['intent'] = intent_result['intent']
            result['processing_steps']['intent'] = intent_result
            logger.info(f"✅ 意图识别: {intent_result['intent']}")

            # 步骤2: 实体识别
            entity_result = self._recognize_entities(processed_query)
            result['entities'] = entity_result['entities']
            result['processing_steps']['entity_recognition'] = entity_result
            logger.info(f"✅ 实体识别: 识别到 {len(entity_result['entities'])} 个实体")

            # 步骤3: 实体链接
            if result['entities'] and self.entity_normalizer:
                linking_result = self._link_entities(result['entities'], processed_query)
                result['normalized_entities'] = linking_result['entities']
                result['processing_steps']['entity_linking'] = linking_result

                normalized_count = sum(1 for e in linking_result.get('entities', []) if e.get('normalized', False))
                logger.info(f"✅ 实体链接: 规范化 {normalized_count}/{len(linking_result.get('entities', []))} 个实体")

                for entity in result['normalized_entities']:
                    if 'kg_id' in entity:
                        entity['entity_id'] = entity['kg_id']
                    if 'kg_name' in entity:
                        entity['entity_name'] = entity['kg_name']
            else:
                result['normalized_entities'] = result['entities']

            # 步骤4: 对话管理 - 查询丰富化
            final_query = processed_query
            final_entities = result['normalized_entities'].copy()

            if self.dialogue_manager and self.dialogue_manager.is_enabled() and dialogue_state:
                # 用上下文信息丰富查询
                enriched_query, supplementary_entities = self.dialogue_manager.enrich_query_for_modules(
                    processed_query, dialogue_state, result['intent']
                )

                if enriched_query != processed_query:
                    logger.info(f"📝 查询丰富化: '{processed_query}' -> '{enriched_query}'")
                    final_query = enriched_query

                # 添加上下文补充的实体
                if supplementary_entities:
                    final_entities.extend(supplementary_entities)
                    logger.info(f"📝 添加上下文实体: {len(supplementary_entities)} 个")

            # 步骤5: 知识图谱查询
            logger.info(f"🔍 知识图谱查询: 意图={result['intent']}, 实体数={len(final_entities)}")
            kg_result = self._query_knowledge_graph(
                result['intent'],
                final_entities,
                final_query
            )
            result['processing_steps']['kg_query'] = kg_result
            logger.info(f"✅ 知识图谱查询: {'成功' if kg_result.get('success') else '失败'}")

            # 步骤6: 提取KG答案信息
            kg_answer_text, kg_has_answer, kg_answer_length = self._get_kg_answer_text(kg_result, result['intent'])
            logger.info(f"📊 KG答案: 有答案={kg_has_answer}, 长度={kg_answer_length}")

            # 步骤7: 应用思维导图逻辑
            kg_min_length = self.config.get("answer_logic", {}).get("kg_min_length", 25)
            final_answer = ""
            answer_source = "unknown"

            # 逻辑1: 若KG答案>25字，直接输出KG答案
            if kg_has_answer and kg_answer_length > kg_min_length:
                logger.info(f"🎯 逻辑1: KG答案长度({kg_answer_length})>{kg_min_length}，直接使用KG答案")
                final_answer = kg_answer_text
                answer_source = "kg_only"
                result['kg_used'] = True
                self.stats['kg_only_answers'] += 1

            # 逻辑2: 若KG答案不足(<25字)或者KG没有在知识图谱搜索到答案，启动RAG
            elif (kg_has_answer and kg_answer_length <= kg_min_length) or not kg_has_answer:
                logger.info(f"🔄 逻辑2: KG答案不足({kg_answer_length}字)或无答案，启动RAG")

                if not self.rag_enabled or not self.rag_retriever:
                    logger.warning("RAG模块未启用，使用现有KG答案")
                    if kg_has_answer:
                        final_answer = kg_answer_text
                        answer_source = "kg_fallback"
                        result['kg_used'] = True
                        self.stats['kg_only_answers'] += 1
                    else:
                        final_answer = "抱歉，暂时没有找到相关信息。"
                        answer_source = "no_info"
                else:
                    # 为RAG构建增强的查询
                    rag_query = final_query
                    if self.dialogue_manager and self.dialogue_manager.is_enabled() and dialogue_state:
                        rag_query = self.dialogue_manager.build_retrieval_context(final_query, dialogue_state)

                    # 启动RAG检索
                    rag_start_time = time.time()
                    try:
                        rag_results = self.rag_retriever.retrieve(rag_query, top_k=2)
                        rag_time = time.time() - rag_start_time

                        if rag_results:
                            logger.info(f"🔍 RAG检索到 {len(rag_results)} 个文档，耗时: {rag_time:.2f}秒")
                            rag_content = self._extract_rag_content(rag_results)
                            result['rag_used'] = True

                            # 尝试使用DeepSeek生成答案
                            deepseek_answer = self._generate_answer_with_deepseek(kg_answer_text, rag_content,
                                                                                  final_query)

                            # 逻辑4: 若RAG中deepseek生成成功，直接输出答案
                            if deepseek_answer:
                                logger.info("🤖 逻辑4: DeepSeek生成成功，使用生成答案")
                                final_answer = deepseek_answer
                                answer_source = "deepseek_generated"
                                result['deepseek_used'] = True
                                self.stats['deepseek_answers'] += 1

                            # 逻辑3: 若RAG中deepseek生成失败
                            else:
                                logger.info("⚠️ DeepSeek生成失败，使用后备方案")

                                # 逻辑3.1: 若KG能在知识图谱查到答案就直接用KG的答案
                                if kg_has_answer:
                                    logger.info("📊 逻辑3.1: 使用KG答案作为后备")
                                    final_answer = kg_answer_text
                                    answer_source = "kg_fallback"
                                    result['kg_used'] = True
                                    self.stats['kg_only_answers'] += 1

                                # 逻辑3.2: 若KG没有搜索到知识图谱的答案，就用RAG中的医疗库的答案
                                else:
                                    logger.info("📄 逻辑3.2: KG无答案，使用RAG医疗库答案")
                                    final_answer = self._format_rag_fallback_answer(rag_content)
                                    answer_source = "rag_fallback"
                                    self.stats['rag_only_answers'] += 1
                        else:
                            logger.info("📭 RAG未检索到相关文档")
                            if kg_has_answer:
                                final_answer = kg_answer_text
                                answer_source = "kg_fallback_no_rag"
                                result['kg_used'] = True
                                self.stats['kg_only_answers'] += 1
                            else:
                                final_answer = "抱歉，暂时没有找到相关信息。"
                                answer_source = "no_info_both"
                                self.stats['fallback_answers'] += 1

                    except Exception as rag_error:
                        logger.error(f"RAG检索失败: {rag_error}")
                        if kg_has_answer:
                            final_answer = kg_answer_text
                            answer_source = "kg_error_fallback"
                            result['kg_used'] = True
                            self.stats['kg_only_answers'] += 1
                        else:
                            final_answer = "抱歉，检索相关信息时发生错误。"
                            answer_source = "rag_error"
            else:
                # 如果KG有答案但长度<=25，且RAG不可用，使用KG答案
                if kg_has_answer:
                    final_answer = kg_answer_text
                    answer_source = "kg_only_short"
                    result['kg_used'] = True
                    self.stats['kg_only_answers'] += 1
                else:
                    final_answer = "抱歉，暂时没有找到相关信息。"
                    answer_source = "no_info"
                    self.stats['fallback_answers'] += 1

            # 保存原始答案
            result['raw_answer'] = final_answer
            result['answer_source'] = answer_source
            result['success'] = True
            self.stats['successful_queries'] += 1

            # 格式化最终答案
            result['answer'] = self._format_final_answer(
                raw_answer=final_answer,
                answer_source=answer_source,
                intent=result['intent'],
                query=user_query
            )

            # ========== 第四步：更新对话状态 ==========
            if self.dialogue_manager and self.dialogue_manager.is_enabled():
                turn_data = {
                    'user_query': user_query,  # 使用原始查询
                    'intent': result['intent'],
                    'entities': result['entities'],
                    'normalized_entities': result['normalized_entities'],
                    'system_response': result['answer'],  # 使用格式化后的答案
                    'answer_source': answer_source,
                    'response_time': time.time() - start_time,
                    'success': result['success'],
                    'errors': result['errors']
                }
                self.dialogue_manager.update_dialogue_state(session_id, turn_data)

        except Exception as e:
            error_msg = f"处理流程发生未预期异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
            result['success'] = False
            result['answer'] = "抱歉，处理您的问题时出现了错误，请稍后再试。"
            self.stats['failed_queries'] += 1

        # 计算响应时间
        response_time = time.time() - start_time
        result['response_time'] = response_time

        # 更新平均响应时间
        if self.stats['total_queries'] > 0:
            self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time)
                    / self.stats['total_queries']
            )

        # 日志输出
        status = "✅" if result['success'] else "❌"
        source_marker = {
            "system_intent": "[系统]",
            "kg_only": "[KG]",
            "kg_fallback": "[KG备]",
            "rag_fallback": "[RAG]",
            "deepseek_generated": "[DS]",
            "kg_only_short": "[KG短]",
            "kg_fallback_no_rag": "[KG无RAG]",
            "kg_error_fallback": "[KG错误备]",
            "rag_error": "[RAG错误]",
            "no_info": "[无]",
            "no_info_both": "[无]"
        }.get(result['answer_source'], "[未知]")

        context_marker = "[有上下文]" if result.get('has_context') else ""
        logger.info(
            f"{status} 处理完成. 耗时: {response_time:.3f}s, 会话: {session_id}, 意图: {result['intent']} {source_marker} {context_marker}")
        return result

    def _recognize_intent(self, text: str) -> Dict[str, Any]:
        """识别查询意图"""
        if self.intent_recognizer:
            try:
                prediction = self.intent_recognizer.predict(text, top_k=1)
                if isinstance(prediction, list) and len(prediction) > 0:
                    pred = prediction[0]
                else:
                    pred = prediction
                intent = pred.get('predicted_intent', '其他')
                confidence = pred.get('confidence', 0.0)
                return {
                    'success': True,
                    'intent': intent,
                    'confidence': confidence,
                    'method': 'model',
                    'error': None
                }
            except Exception as e:
                logger.error(f"意图识别模型调用失败: {e}")
        return self._recognize_intent_by_keywords(text)

    def _recognize_intent_by_keywords(self, text: str) -> Dict[str, Any]:
        """通过关键词匹配识别意图"""
        keyword_map = {
            "临床表现(病症表现)": ["症状", "表现", "症候", "有什么症状", "哪些症状", "早期症状", "常见症状"],
            "所属科室": ["看什么科", "挂什么科", "哪个科室", "什么科室", "就诊", "应该看"],
            "治疗方法": ["怎么治", "如何治疗", "治疗方法", "怎么治疗", "医治", "治疗方案", "怎么治好"],
            "化验/体检方案": ["检查", "化验", "体检", "做什么检查", "检测", "需要检查"],
            "定义": ["是什么", "什么叫", "什么是", "定义", "意思"],
            "传染性": ["传染", "传染性", "会传染", "传染吗", "传染途径"],
            "治愈率": ["治愈率", "能治好吗", "治好", "治愈", "能治愈吗"],
            "治疗时间": ["治疗时间", "多久能好", "多长时间", "疗程", "需要多久"],
            "病因": ["病因", "原因", "为什么", "怎么回事", "什么原因引起"],
            "相关病症": ["并发症", "相关疾病", "一起的病", "合并症", "伴随症状"],
            "禁忌": ["禁忌", "不能", "避免", "忌", "不可以", "禁止"],
            "预防": ["预防", "怎么预防", "如何预防", "防止", "预防措施"]
        }
        for intent, keywords in keyword_map.items():
            for keyword in keywords:
                if keyword in text:
                    return {
                        'success': True,
                        'intent': intent,
                        'confidence': 0.7,
                        'method': 'keyword',
                        'matched_keyword': keyword
                    }
        return {
            'success': True,
            'intent': '其他',
            'confidence': 0.5,
            'method': 'keyword',
            'error': None
        }

    def _recognize_entities(self, text: str) -> Dict[str, Any]:
        """识别文本中的实体"""
        if self.entity_recognizer:
            try:
                entities = self.entity_recognizer.recognize(text, use_linking=False)
                return {
                    'success': True,
                    'entities': entities,
                    'method': 'model',
                    'count': len(entities)
                }
            except Exception as e:
                logger.error(f"实体识别失败: {e}")
        return self._simple_entity_recognition(text)

    def _simple_entity_recognition(self, text: str) -> Dict[str, Any]:
        """简单的实体识别后备方案"""
        disease_keywords = ["糖尿病", "高血压", "感冒", "发烧", "头痛", "咳嗽", "肺炎", "心脏病",
                            "胃炎", "肝炎", "哮喘", "肺炎", "胃炎", "肾炎", "关节炎"]
        entities = []
        for disease in disease_keywords:
            if disease in text:
                entities.append({
                    'text': disease,
                    'type': '疾病',
                    'start': text.find(disease),
                    'end': text.find(disease) + len(disease)
                })
        return {
            'success': len(entities) > 0,
            'entities': entities,
            'method': 'simple_keyword',
            'count': len(entities)
        }

    def _link_entities(self, entities: List[Dict], text: str) -> Dict[str, Any]:
        """对识别出的实体进行链接和规范化"""
        if self.entity_normalizer and entities:
            try:
                # 调用实体链接模块
                if hasattr(self.entity_normalizer, 'normalize_entities'):
                    normalized_entities = self.entity_normalizer.normalize_entities(entities, text)
                elif hasattr(self.entity_normalizer, 'normalize'):
                    normalized_entities = self.entity_normalizer.normalize(entities, text)
                else:
                    logger.error("实体链接模块没有可用的normalize方法")
                    for entity in entities:
                        entity['normalized'] = False
                    return {
                        'success': False,
                        'entities': entities,
                        'method': 'none',
                        'success_count': 0,
                        'total_count': len(entities)
                    }

                success_count = sum(1 for e in normalized_entities if e.get('normalized', False))
                return {
                    'success': success_count > 0,
                    'entities': normalized_entities,
                    'method': 'model',
                    'success_count': success_count,
                    'total_count': len(normalized_entities)
                }
            except Exception as e:
                logger.error(f"实体链接失败: {e}")
        # 如果实体链接失败，返回原始实体
        for entity in entities:
            entity['normalized'] = False
        return {
            'success': False,
            'entities': entities,
            'method': 'none',
            'success_count': 0,
            'total_count': len(entities)
        }

    def _query_knowledge_graph(self, intent: str, entities: List[Dict], text: str) -> Dict[str, Any]:
        """查询知识图谱 - 修复实体格式"""
        if self.kg_querier:
            try:
                # 确保实体格式正确
                processed_entities = []
                for entity in entities:
                    # 复制实体
                    processed_entity = entity.copy()

                    # 确保有entity_id字段
                    if 'entity_id' not in processed_entity and 'kg_id' in processed_entity:
                        processed_entity['entity_id'] = processed_entity['kg_id']
                    elif 'entity_id' not in processed_entity:
                        processed_entity['entity_id'] = None

                    # 确保有entity_name字段
                    if 'entity_name' not in processed_entity:
                        if 'kg_name' in processed_entity:
                            processed_entity['entity_name'] = processed_entity['kg_name']
                        elif 'normalized_text' in processed_entity:
                            processed_entity['entity_name'] = processed_entity['normalized_text']
                        elif 'text' in processed_entity:
                            processed_entity['entity_name'] = processed_entity['text']
                        else:
                            processed_entity['entity_name'] = "未知"

                    processed_entities.append(processed_entity)

                # 调试日志
                logger.debug(f"传递给KG查询器的实体数量: {len(processed_entities)}")
                for i, entity in enumerate(processed_entities):
                    logger.debug(
                        f"实体{i}: entity_id='{entity.get('entity_id')}', entity_name='{entity.get('entity_name')}'")

                result = self.kg_querier.query_by_intent(intent, processed_entities, text)

                # 调试：检查返回结果
                logger.debug(f"KG查询结果: {result}")
                if result.get('success') and result.get('data'):
                    data = result.get('data', {})
                    logger.debug(f"KG数据字段: {list(data.keys())}")
                    for key, value in data.items():
                        if value:
                            logger.debug(f"  {key}: {value}")

                return result
            except Exception as e:
                logger.error(f"知识图谱查询失败: {e}")
        return self._mock_kg_query(intent, entities, text)

    def _mock_kg_query(self, intent: str, entities: List[Dict], text: str) -> Dict[str, Any]:
        """模拟知识图谱查询结果"""
        disease_name = "未知疾病"
        for entity in entities:
            if entity.get('type') in ['疾病', 'DISEASE'] or '疾病' in str(entity.get('type', '')):
                disease_name = entity.get('text', disease_name)
                break
        mock_data = {
            "糖尿病": {
                "临床表现(病症表现)": ["多饮、多尿、多食、体重下降", "视力模糊", "疲劳乏力"],
                "所属科室": ["内分泌科", "糖尿病专科"],
                "治疗方法": ["饮食控制", "运动疗法", "药物治疗"],
                "化验/体检方案": ["空腹血糖检测", "糖化血红蛋白检测"],
                "定义": "糖尿病是一种以高血糖为特征的代谢性疾病。",
                "传染性": "糖尿病不是传染病，不会在人与人之间传播。",
                "治愈率": "目前糖尿病无法完全根治，但可通过治疗良好控制。",
                "病因": ["遗传因素", "环境因素", "不良生活习惯"],
                "相关病症": ["高血压", "糖尿病视网膜病变", "糖尿病肾病"],
                "禁忌": ["高糖食物", "高脂食物", "过度饮酒"],
                "预防": ["健康饮食", "规律运动", "控制体重", "定期体检"]
            },
            "高血压": {
                "所属科室": ["心血管内科", "全科"],
                "临床表现(病症表现)": ["头晕", "头痛", "心悸", "耳鸣"],
                "治疗方法": ["药物治疗", "低盐饮食", "规律运动"],
                "定义": "高血压是一种动脉血压持续升高的慢性疾病。"
            },
            "感冒": {
                "所属科室": ["呼吸内科", "全科"],
                "临床表现(病症表现)": ["流鼻涕", "打喷嚏", "咳嗽", "喉咙痛"],
                "治疗方法": ["休息", "多喝水", "对症治疗"],
                "传染性": "感冒具有传染性，主要通过飞沫传播。"
            }
        }
        data = mock_data.get(disease_name, {})
        result_value = data.get(intent, f"关于{disease_name}的{intent}信息暂未收录。")
        return {
            'success': True,
            'confidence': 0.8,
            'data': {
                'disease_name': disease_name,
                'intent': intent,
                'result': result_value,
                'entities': [e.get('text', '') for e in entities]
            },
            'execution_time': 0.05,
            'timestamp': datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统运行统计信息"""
        total = self.stats['total_queries']
        if total == 0:
            success_rate = 0
        else:
            success_rate = self.stats['successful_queries'] / total * 100

        return {
            'total_queries': total,
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'success_rate': f"{success_rate:.2f}%",
            'avg_response_time': f"{self.stats['avg_response_time']:.3f}秒",
            'kg_only_answers': self.stats['kg_only_answers'],
            'rag_only_answers': self.stats['rag_only_answers'],
            'deepseek_answers': self.stats['deepseek_answers'],
            'fallback_answers': self.stats['fallback_answers'],
            'system_intent_answers': self.stats['system_intent_answers']
        }

    def close(self):
        """关闭系统，释放资源"""
        logger.info("正在关闭医疗问答系统...")
        if hasattr(self.kg_querier, 'close'):
            self.kg_querier.close()
        if hasattr(self.entity_normalizer, 'close'):
            self.entity_normalizer.close()
        logger.info("系统已关闭。")


def quick_test():
    """快速测试集成系统"""
    print("=" * 60)
    print("医疗问答集成系统 - 快速测试（优化版）")
    print("=" * 60)

    system = MedicalQAIntegratedSystem()

    test_queries = [
        "你叫什么",
        "小U",
        "你可以为我做什么",
        "糖尿病有什么症状？",
        "糖尿病的最新治疗方法是什么？",
        "高血压应该怎么预防？",
        "感冒了怎么办？",
        "胃痛应该注意什么？",
        "再见",
    ]
    # 测试查询

    for query in test_queries:
        print(f"\n[用户]: {query}")
        result = system.process_query(query)

        if result['success']:
            answer = result['answer']

            print(f"[系统]: {answer}")
            print(f"      意图: {result['intent']}, 成功: {result['success']}, 耗时: {result['response_time']:.3f}s")
            print(f"      答案来源: {result.get('answer_source', 'unknown')}")

        if result.get('kg_used'):
            print(f"      📊 使用KG知识")
        if result.get('rag_used'):
            print(f"      🔍 使用RAG检索")
        if result.get('deepseek_used'):
            print(f"      🤖 使用DeepSeek生成")

        if result.get('errors'):
            print(f"      错误: {result['errors']}")

    print("\n" + "=" * 60)
    stats = system.get_statistics()
    print("测试统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    system.close()


if __name__ == "__main__":
    quick_test()