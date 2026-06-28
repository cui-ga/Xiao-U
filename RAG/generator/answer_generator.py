"""
答案生成器 - 整合检索、重排序、知识融合、提示词构造、模型生成流程
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
        self.config = config or {}

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

            # 2. 生成器
            generator_config = self.config.get('generator_config', {})
            self.generator = DeepSeekGenerator(generator_config)
            logger.info("生成器初始化完成")

            # 3. 提示管理器
            prompt_templates = self.config.get('prompt_templates', {})
            self.prompt_manager = PromptManager(prompt_templates)
            logger.info("提示管理器初始化完成")

            # 4. 知识融合器
            fusion_config = self.config.get('fusion_config', {})
            self.fuser = KnowledgeFuser(fusion_config)
            logger.info("知识融合器初始化完成")

        except Exception as e:
            logger.error(f"初始化组件失败: {e}", exc_info=True)
            raise

    def generate_answer(
            self,
            query: str,
            kg_result: Optional[Dict[str, Any]] = None,
            use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        生成答案（完整RAG流程）

        Args:
            query: 用户查询
            kg_result: 知识图谱结果
            use_reranking: 是否使用重排序。当前实际由 RAGRetriever 配置控制，此参数保留兼容。

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
            # 1. RAG 检索
            logger.info(f"开始检索: {query[:50]}...")
            retrieved_docs = self.retriever.retrieve(query)

            # 2. 如果没有检索到文档，尝试只使用 KG
            if not retrieved_docs:
                logger.warning("未检索到相关RAG文档")

                if kg_result and kg_result.get('success'):
                    kg_answer = self._format_kg_answer(kg_result)

                    # 只使用 KG 时，也交给 PromptManager + DeepSeek 润色
                    context = f"【知识图谱信息】\n{kg_answer}"

                    messages = self.prompt_manager.build_medical_messages(
                        question=query,
                        context=context
                    )

                    result['prompt'] = self._preview_messages(messages)
                    answer = self.generator.generate(messages=messages)

                    if answer and len(answer.strip()) >= 10:
                        result['answer'] = answer
                    else:
                        # 如果模型生成失败，则直接返回格式化 KG 答案
                        result['answer'] = kg_answer

                    result['used_kg'] = True
                    result['success'] = True
                    return result

                result['error'] = "未找到相关信息"
                return result

            result['retrieved_docs'] = retrieved_docs
            result['used_rag'] = True

            # 3. KG + RAG 知识融合
            logger.info("进行知识融合...")
            fused_context = self.fuser.fuse(kg_result, retrieved_docs, query)

            result['used_kg'] = fused_context.get('has_kg_info', False)

            # 4. 构造统一上下文
            context_text = self._build_context_text(fused_context)

            # 5. 使用 PromptManager 构造 messages
            logger.info("使用PromptManager生成messages...")
            messages = self.prompt_manager.build_medical_messages(
                question=query,
                context=context_text
            )

            result['prompt'] = self._preview_messages(messages)

            # 6. 调用 DeepSeekGenerator 生成答案
            logger.info("调用大模型生成答案...")
            answer = self.generator.generate(messages=messages)

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

    def _build_context_text(self, fused_context: Dict[str, Any]) -> str:
        """
        将知识融合结果整理成给 PromptManager 使用的 context 文本。
        """
        if not fused_context:
            return ""

        parts = []

        kg_context = fused_context.get('kg_context', '')
        rag_context = fused_context.get('rag_context', '')

        if kg_context:
            parts.append(f"【知识图谱信息】\n{kg_context}")

        if rag_context:
            parts.append(f"【RAG检索信息】\n{rag_context}")

        return "\n\n".join(parts).strip()

    def _preview_messages(self, messages: List[Dict[str, str]], max_len: int = 500) -> str:
        """
        保存 prompt 预览，避免 result 中内容过长。
        """
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text_parts.append(f"[{role}]\n{content}")

        preview = "\n\n".join(text_parts)
        return preview[:max_len] + "..." if len(preview) > max_len else preview

    def _format_kg_answer(self, kg_result: Dict[str, Any]) -> str:
        """
        格式化知识图谱答案。
        """
        if not kg_result or not kg_result.get('success'):
            return "抱歉，我暂时无法回答这个问题。"

        data = kg_result.get('data', {})
        disease = data.get('disease_name', '')
        kg_data = data.get('result', '')

        if isinstance(kg_data, list):
            result_str = "、".join(str(item) for item in kg_data)
        else:
            result_str = str(kg_data)

        if disease and result_str:
            return f"{disease}：{result_str}"

        if result_str:
            return result_str

        return "相关信息不足，建议咨询专业医生。"

    def batch_generate(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        批量生成答案。
        """
        results = []

        for query in queries:
            result = self.generate_answer(query, **kwargs)
            results.append(result)

        return results
