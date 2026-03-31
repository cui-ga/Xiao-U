"""
RAG主模块
"""
import logging
from typing import Dict, Any, List
from .config.rag_config import RETRIEVE_CONFIG, DEEPSEEK_CONFIG, PROMPT_TEMPLATES
from .retriever.retriever import RAGRetriever
from .generator.deepseek_integration import DeepSeekGenerator
from .generator.prompt_templates import PromptManager
from .knowledge_fuser.fuser import KnowledgeFuser
from .data_loader.cmedqa2_loader import CMEDQA2Loader
from .data_loader.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class RAGModule:
    """RAG主模块"""

    def __init__(self, config_path: str = None):
        # 加载配置
        from .config.rag_config import (
            EMBEDDING_MODEL, RERANKER_MODEL, DEEPSEEK_MODEL,
            VECTOR_DB_PATH, CMEDQA2_DATA_PATH
        )

        self.config = {
            'embedding_model': EMBEDDING_MODEL,
            'reranker_model': RERANKER_MODEL,
            'deepseek_model': DEEPSEEK_MODEL,
            'vector_db_path': str(VECTOR_DB_PATH),
            'cmedqa2_data_path': str(CMEDQA2_DATA_PATH),
            **RETRIEVE_CONFIG
        }

        # 初始化组件
        self.retriever = None
        self.generator = None
        self.prompt_manager = None
        self.fuser = None
        self.initialized = False

    def initialize(self) -> bool:
        """初始化RAG模块"""
        try:
            logger.info("开始初始化RAG模块...")

            # 初始化检索器
            self.retriever = RAGRetriever(self.config)

            # 初始化生成器
            generator_config = {
                **DEEPSEEK_CONFIG,
                'model': self.config['deepseek_model']
            }
            self.generator = DeepSeekGenerator(generator_config)

            # 初始化提示管理器
            self.prompt_manager = PromptManager(PROMPT_TEMPLATES)

            # 初始化知识融合器
            self.fuser = KnowledgeFuser(self.config)

            # 检查是否需要构建索引
            if not self.retriever.vector_store.load():
                logger.info("构建新的向量索引...")
                self._build_index()

            self.initialized = True
            logger.info("RAG模块初始化完成")
            return True

        except Exception as e:
            logger.error(f"RAG模块初始化失败: {e}")
            return False

    def _build_index(self) -> None:
        """构建向量索引"""
        # 加载cMedQA2数据
        loader = CMEDQA2Loader(self.config['cmedqa2_data_path'])
        documents = loader.load_data()

        if documents:
            # 处理文档
            processor = DocumentProcessor()
            processed_docs = processor.process_documents(documents)

            # 添加到向量存储
            self.retriever.add_documents(processed_docs)
        else:
            logger.warning("cMedQA2数据为空，无法构建索引")

    def process_query(self, user_query: str, kg_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理用户查询

        Args:
            user_query: 用户查询
            kg_result: 知识图谱结果

        Returns:
            处理结果
        """
        if not self.initialized:
            return {
                'success': False,
                'error': 'RAG模块未初始化',
                'answer': None
            }

        try:
            # 1. 检索相关文档
            retrieved_docs = self.retriever.retrieve(user_query)

            # 2. 融合知识
            fused_context = self.fuser.fuse(kg_result, retrieved_docs, user_query)

            # 3. 生成提示
            prompt = self.prompt_manager.get_prompt(
                'knowledge_fusion',
                kg_context=fused_context.get('kg_context', ''),
                rag_context=fused_context.get('rag_context', ''),
                question=user_query
            )

            # 4. 生成回答
            answer = self.generator.generate(prompt, context=None)

            return {
                'success': True,
                'answer': answer,
                'retrieved_docs': retrieved_docs,
                'used_kg': fused_context.get('has_kg_info', False),
                'used_rag': fused_context.get('has_rag_info', False),
                'fused_context': fused_context
            }

        except Exception as e:
            logger.error(f"RAG处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': None
            }

    def augment_knowledge(self, user_query: str, kg_answer: str) -> str:
        """
        增强现有知识图谱答案

        Args:
            user_query: 用户查询
            kg_answer: 知识图谱的原始答案

        Returns:
            增强后的答案
        """
        if not self.initialized:
            return kg_answer

        try:
            # 检索补充信息
            retrieved_docs = self.retriever.retrieve(user_query)

            if not retrieved_docs:
                return kg_answer

            # 生成提示
            prompt = self.prompt_manager.get_prompt(
                'knowledge_augment',
                original_answer=kg_answer,
                additional_info="\n".join([doc.get('content', '')[:500] for doc in retrieved_docs[:3]]),
                question=user_query
            )

            # 生成增强答案
            augmented_answer = self.generator.generate(prompt)

            return augmented_answer

        except Exception as e:
            logger.error(f"知识增强失败: {e}")
            return kg_answer