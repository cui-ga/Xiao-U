# RAG/generator/prompt_templates.py
"""
提示模板管理器
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PromptManager:
    """提示模板管理器"""

    def __init__(self, templates: Dict[str, str] = None):
        """
        初始化提示管理器

        Args:
            templates: 提示模板字典
        """
        self.templates = templates or self._get_default_templates()

    def _get_default_templates(self) -> Dict[str, str]:
        """获取默认提示模板"""
        return {
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

            "simple_qa": """请根据以下信息回答问题：

相关信息：
{context}

问题：{question}

请直接、准确地回答："""
        }

    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        获取填充后的提示

        Args:
            template_name: 模板名称
            **kwargs: 模板参数

        Returns:
            填充后的提示文本
        """
        if template_name not in self.templates:
            logger.warning(f"提示模板 '{template_name}' 不存在，使用默认模板")
            template = self.templates.get("simple_qa", "{context}\n\n问题：{question}\n\n回答：")
        else:
            template = self.templates[template_name]

        try:
            # 填充模板
            prompt = template.format(**kwargs)

            # 验证填充结果
            if "{" in prompt and "}" in prompt:
                logger.warning(f"模板 '{template_name}' 可能未完全填充")

            return prompt

        except KeyError as e:
            logger.error(f"填充模板 '{template_name}' 时缺少参数: {e}")
            # 尝试使用简单格式
            return f"信息：{kwargs.get('context', '')}\n\n问题：{kwargs.get('question', '')}\n\n请回答："

    def add_template(self, name: str, template: str) -> None:
        """添加新模板"""
        self.templates[name] = template
        logger.info(f"已添加新模板: {name}")

    def list_templates(self) -> list:
        """列出所有模板"""
        return list(self.templates.keys())