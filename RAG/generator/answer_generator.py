# RAG/generator/answer_generator.py
"""
答案生成器 - 整合检索、重排序、生成流程
"""
from typing import Dict, Any, List, Optional
import logging
from ..retriever.retriever import RAGRetriever
from ..generator.deepseek_integration import DeepSeekGenerator
from ..generator.prompt_templates import PromptManager
from ..knowledge_fuser.fuser import KnowledgeFuser

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """答案生成器（RAG流程协调器）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化答案生成器

        Args:
            config: 配置字典
        """
        self.config = config

        # 初始化组件
        self.retriever = None
        self.generator = None
        self.prompt_manager = None
        self.fuser = None

        self._initialize_components()

    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 1. 检索器
            self.retriever = RAGRetriever(self.config)
            logger.info("检索器初始化完成")

            # 2. 生成器（DeepSeek）
            generator_config = self.config.get('generator_config', {})
            self.generator = DeepSeekGenerator(generator_config)
            logger.info("生成器初始化完成")

            # 3. 提示管理器
            prompt_templates = self.config.get('prompt_templates', {})
            self.prompt_manager = PromptManager(prompt_templates)
            logger.info("提示管理器初始化完成")

            # 4. 知识融合器
            self.fuser = KnowledgeFuser(self.config.get('fusion_config', {}))
            logger.info("知识融合器初始化完成")

        except Exception as e:
            logger.error(f"初始化组件失败: {e}")
            raise

    def generate_answer(self,
                        query: str,
                        kg_result: Optional[Dict[str, Any]] = None,
                        use_reranking: bool = True) -> Dict[str, Any]:
        """
        生成答案（完整的RAG流程）

        Args:
            query: 用户查询
            kg_result: 知识图谱结果
            use_reranking: 是否使用重排序

        Returns:
            生成结果字典
        """
        result = {
            'success': False,
            'answer': '',
            'retrieved_docs': [],
            'used_kg': False,
            'used_rag': False,
            'prompt': '',
            'error': None
        }

        try:
            # 1. 检索相关文档
            logger.info(f"开始检索: {query[:50]}...")
            retrieved_docs = self.retriever.retrieve(query)

            if not retrieved_docs:
                logger.warning("未检索到相关文档")
                if kg_result and kg_result.get('success'):
                    # 只使用知识图谱结果
                    result['answer'] = self._format_kg_answer(kg_result)
                    result['used_kg'] = True
                    result['success'] = True
                else:
                    result['error'] = "未找到相关信息"
                return result

            result['retrieved_docs'] = retrieved_docs
            result['used_rag'] = True

            # 2. 知识融合
            logger.info("进行知识融合...")
            fused_context = self.fuser.fuse(kg_result, retrieved_docs, query)
            result['used_kg'] = fused_context.get('has_kg_info', False)

            # 3. 生成提示
            logger.info("生成提示...")
            prompt = self.prompt_manager.get_prompt(
                'knowledge_fusion',
                kg_context=fused_context.get('kg_context', ''),
                rag_context=fused_context.get('rag_context', ''),
                question=query
            )
            result['prompt'] = prompt[:500] + "..." if len(prompt) > 500 else prompt

            # 4. 生成答案
            logger.info("调用大模型生成答案...")
            answer = self.generator.generate(prompt)

            if not answer or len(answer.strip()) < 10:
                logger.warning("生成的答案过短或为空")
                result['error'] = "生成答案失败"
                return result

            result['answer'] = answer
            result['success'] = True
            logger.info("答案生成成功")

        except Exception as e:
            logger.error(f"生成答案失败: {e}", exc_info=True)
            result['error'] = str(e)

        return result

    def _format_kg_answer(self, kg_result: Dict[str, Any]) -> str:
        """格式化知识图谱答案"""
        if not kg_result or not kg_result.get('success'):
            return "抱歉，我暂时无法回答这个问题。"

        data = kg_result.get('data', {})
        disease = data.get('disease_name', '')
        result = data.get('result', '')

        if isinstance(result, list):
            result_str = "、".join(result)
        else:
            result_str = str(result)

        if disease and result_str:
            return f"{disease}：{result_str}"
        elif result_str:
            return result_str
        else:
            return "相关信息不足，建议咨询专业医生。"

    def batch_generate(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量生成答案"""
        results = []
        for query in queries:
            result = self.generate_answer(query, **kwargs)
            results.append(result)
        return results