"""
提示模板管理器
统一负责所有提示词构造。
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """提示模板管理器"""

    def __init__(self, templates: Dict[str, str] = None):
        """
        初始化提示管理器

        Args:
            templates: 外部自定义提示模板字典
        """
        default_templates = self._get_default_templates()

        # 如果外部传入模板，则覆盖或补充默认模板
        if templates:
            default_templates.update(templates)

        self.templates = default_templates

    def _get_default_templates(self) -> Dict[str, str]:
        """获取默认提示模板"""
        return {
            # -------------------------------------------------
            # 1. DeepSeek 医疗助手 system prompt
            # 从 deepseek_integration.py 中迁移过来，作为统一模板管理
            # -------------------------------------------------
            "deepseek_medical_system": """你是一个专业的医疗AI助手，小U。请基于用户的问题和相关医学信息，给出准确、完整、专业的回答。

回答要求：
1. 参考信息中可能包含"问题："、"答案："等格式标记，请忽略这些标记，只关注医学内容本身
2. 针对用户的具体问题，用自然、直接的语言回答，不要复述参考信息或模仿其格式
3. 从参考信息中提取关键的医学知识点，用自己的话组织成完整的回答
4. 使用专业术语但解释清晰，让非专业人士也能理解
5. 如果信息适合，可以分点说明，但不要使用"答案1"、"答案2"这样的编号

特别注意事项：
- 不要包含"问题："、"答案："、"参考信息："等内部标记
- 不要重复相同的内容
- 如果信息不充分，可以基于医学常识补充，但要说明这是补充信息
- 保持回答的连贯性和完整性
- 医疗建议不能替代医生诊断，如症状严重、持续不缓解或涉及用药调整，应建议及时就医""",

            # -------------------------------------------------
            # 2. 有 RAG / KG 上下文时使用
            # -------------------------------------------------
            "deepseek_medical_qa": """用户问题：{question}

参考医学信息：
{context}

【回答要求】
1. 针对用户的具体问题"{question}"，给出直接的回答
2. 忽略参考信息中的格式标记，只提取有用的医学知识点
3. 用自然、专业的语言组织回答，不要模仿参考信息的格式
4. 回答要完整、准确、结构清晰
5. 如果信息冲突，优先采用知识图谱信息
6. 如涉及诊断、用药或病情加重，应提醒用户咨询专业医生

请开始你的专业回答：""",

            # -------------------------------------------------
            # 3. 没有上下文时使用
            # -------------------------------------------------
            "deepseek_medical_no_context": """【任务说明】
你是一位专业的医疗助手，需要回答用户关于医疗健康的问题。

【用户的具体问题】
{question}

【回答要求】
请基于你的医学知识，用专业、准确的语言回答上述问题。回答要完整、清晰。
如果信息不足，请说明不能完全确定，并建议用户咨询专业医生。

请开始你的专业回答：""",

            # -------------------------------------------------
            # 4. 原有普通 RAG 医疗问答模板，保留
            # -------------------------------------------------
            "medical_qa": """你是一个专业的医疗AI助手。请根据以下医学知识和用户问题，提供准确、专业的回答。

相关医学知识：
{context}

用户问题：{question}

请按照以下格式回答：
1. 首先给出直接回答
2. 然后提供相关医学解释
3. 最后给出注意事项或建议

回答时请：
- 使用专业但易懂的语言
- 如果信息不足，请明确指出
- 避免绝对化的表述
- 始终建议咨询专业医生

你的回答：""",

            # -------------------------------------------------
            # 5. KG + RAG 融合模板，保留备用
            # -------------------------------------------------
            "knowledge_fusion": """请将以下知识图谱信息和检索到的医学知识融合，回答用户问题：

知识图谱信息：
{kg_context}

相关医学文献：
{rag_context}

用户问题：{question}

请整合以上信息，给出全面的回答，要求：
1. 以知识图谱信息为主要依据
2. 用医学文献信息进行补充说明
3. 如果信息冲突，优先采用知识图谱信息
4. 标注信息来源

你的回答：""",

            # -------------------------------------------------
            # 6. 答案增强模板，保留备用
            # -------------------------------------------------
            "knowledge_augment": """你是一个医疗AI助手，需要基于以下原始答案和相关医学文献，提供一个更全面、准确的回答。

原始答案：
{original_answer}

相关医学文献（补充信息）：
{additional_info}

用户问题：{question}

请根据以上信息，提供一个增强版的回答，要求：
1. 保留原始答案的核心信息
2. 补充医学文献中的相关细节
3. 如果补充信息与原始答案有差异，以保守、安全的方式说明
4. 保持回答的专业性和可读性

增强后的回答：""",

            # -------------------------------------------------
            # 7. 简单兜底模板
            # -------------------------------------------------
            "simple_qa": """请根据以下信息回答问题：

相关信息：
{context}

问题：{question}

请直接、准确地回答："""
        }

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        获取填充后的提示词文本

        Args:
            template_name: 模板名称
            **kwargs: 模板变量

        Returns:
            填充后的提示词文本
        """
        if template_name not in self.templates:
            logger.warning(f"提示模板 '{template_name}' 不存在，使用 simple_qa 模板")
            template = self.templates.get(
                "simple_qa",
                "{context}\n\n问题：{question}\n\n回答："
            )
        else:
            template = self.templates[template_name]

        try:
            prompt = template.format(**kwargs)

            # 如果填充后仍然存在明显的占位符，给出警告
            if "{" in prompt and "}" in prompt:
                logger.warning(f"模板 '{template_name}' 可能未完全填充")

            return prompt

        except KeyError as e:
            logger.error(f"填充模板 '{template_name}' 时缺少参数: {e}")
            return (
                f"信息：{kwargs.get('context', '')}\n\n"
                f"问题：{kwargs.get('question', '')}\n\n"
                f"请回答："
            )

    def get_messages(
        self,
        user_template_name: str,
        system_template_name: str = "deepseek_medical_system",
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        构造 OpenAI / Ollama 兼容的 messages。

        Args:
            user_template_name: user prompt 模板名
            system_template_name: system prompt 模板名
            **kwargs: 模板变量

        Returns:
            messages 列表
        """
        system_prompt = self.get_prompt(system_template_name, **kwargs)
        user_prompt = self.get_prompt(user_template_name, **kwargs)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def build_medical_messages(self, question: str, context: str = None) -> List[Dict[str, str]]:
        """
        根据是否存在上下文，自动构造医疗问答 messages。

        Args:
            question: 用户问题
            context: KG / RAG 参考医学信息

        Returns:
            messages 列表
        """
        if context and len(context.strip()) > 10:
            return self.get_messages(
                "deepseek_medical_qa",
                question=question,
                context=context
            )

        return self.get_messages(
            "deepseek_medical_no_context",
            question=question,
            context=""
        )

    def add_template(self, name: str, template: str) -> None:
        """添加新模板"""
        self.templates[name] = template
        logger.info(f"已添加新模板: {name}")

    def list_templates(self) -> list:
        """列出所有模板名称"""
        return list(self.templates.keys())
