"""
DeepSeek / Ollama 生成器
只负责模型 API 调用、重试、返回结果清洗。
提示词统一由 prompt_templates.py 中的 PromptManager 构造。
"""

import requests
import logging
import time
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DeepSeekGenerator:
    """DeepSeek生成器 - 只负责API调用，不再负责提示词构造"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DeepSeek 生成器

        Args:
            config: 配置字典
        """
        self.config = config or {}

        self.api_base = self.config.get('api_base', 'http://localhost:11434/v1')
        self.api_key = self.config.get('api_key', 'ollama')
        self.model = self.config.get('model', 'deepseek-r1:1.5b')

        self.max_tokens = self.config.get('max_tokens', 1024)
        self.temperature = self.config.get('temperature', 0.1)
        self.top_p = self.config.get('top_p', 0.7)
        self.timeout = self.config.get('timeout', 50)

        self.frequency_penalty = self.config.get('frequency_penalty', 0.1)
        self.presence_penalty = self.config.get('presence_penalty', 0.1)
        self.max_retries = self.config.get('max_retries', 2)

        logger.info(
            f"初始化DeepSeek生成器 "
            f"(模型: {self.model}, API: {self.api_base}, 超时: {self.timeout}秒)"
        )

    def generate(
        self,
        prompt: str = None,
        messages: List[Dict[str, str]] = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """
        调用大模型生成回答。

        注意：
        - 推荐传入 messages，因为提示词应由 PromptManager 构造。
        - 如果只传 prompt，则把 prompt 作为 user message 发送。
        - 本类不再负责拼接 system_prompt、context 等提示词内容。

        Args:
            prompt: 已经构造好的完整用户提示词
            messages: OpenAI / Ollama 兼容 messages
            max_tokens: 最大生成 token 数

        Returns:
            生成后的文本；失败返回 None
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        if messages is None:
            if not prompt or not prompt.strip():
                logger.error("generate() 缺少 prompt 或 messages")
                return None

            messages = [
                {"role": "user", "content": prompt}
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    f"正在调用DeepSeek API "
                    f"(尝试 {attempt + 1}/{self.max_retries + 1})，模型: {self.model}"
                )
                start_time = time.time()

                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

                elapsed = time.time() - start_time

                if response.status_code == 200:
                    response_data = response.json()

                    if 'choices' not in response_data or len(response_data['choices']) == 0:
                        logger.error("API返回格式错误：缺少 choices 字段")
                        if attempt < self.max_retries:
                            time.sleep(1)
                            continue
                        return None

                    message = response_data['choices'][0].get('message', {})
                    content = message.get('content', '')

                    if not content or not content.strip():
                        logger.warning("API返回空内容")
                        if attempt < self.max_retries:
                            time.sleep(1)
                            continue
                        return None

                    clean_content = self._clean_generated_text(content)

                    logger.info(
                        f"✅ DeepSeek生成成功，耗时: {elapsed:.2f}秒，"
                        f"长度: {len(clean_content)}字符"
                    )

                    if len(clean_content) < 10:
                        logger.warning(
                            f"生成内容过短: {len(clean_content)}字符，但仍直接返回"
                        )

                    return clean_content.strip()

                elif response.status_code == 503:
                    logger.warning(
                        f"服务暂时不可用 503，准备重试 "
                        f"({attempt + 1}/{self.max_retries + 1})"
                    )
                    if attempt < self.max_retries:
                        time.sleep(2)
                        continue
                    return None

                else:
                    logger.error(
                        f"API调用失败: {response.status_code} - "
                        f"{response.text[:200]}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"DeepSeek API请求超时 timeout={self.timeout}s")
                if attempt < self.max_retries:
                    logger.info("正在重试超时请求...")
                    time.sleep(2)
                    continue
                return None

            except requests.exceptions.ConnectionError as e:
                logger.error(f"无法连接到DeepSeek API: {e}")
                if attempt < self.max_retries:
                    logger.info("正在重试连接...")
                    time.sleep(3)
                    continue
                return None

            except Exception as e:
                logger.error(f"DeepSeek生成异常: {e}", exc_info=True)
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                return None

        return None

    def _clean_generated_text(self, text: str) -> str:
        """
        清理生成文本，移除不适合展示的开头和 DeepSeek-R1 思考标签。
        """
        if not text:
            return ""

        cleaned = text.strip()

        # 1. 清理 DeepSeek-R1 可能输出的 <think>...</think>
        cleaned = re.sub(
            r'<think>.*?</think>',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        # 2. 清理中文“思考过程”标签
        cleaned = re.sub(
            r'思考过程[:：].*?(?=\n\n|$)',
            '',
            cleaned,
            flags=re.DOTALL
        )

        # 3. 清理常见开头废话
        patterns = [
            r'^思考[:：].*?\n',
            r'^推理[:：].*?\n',
            r'^首先，.*?\n',
            r'^然后，.*?\n',
            r'^最后，.*?\n',
            r'^作为一个AI.*?\n',
            r'^作为一名.*?\n',
            r'^好的，.*?\n',
            r'^当然，.*?\n',
        ]

        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)

        # 4. 移除内部标记，避免把 RAG 原始格式输出给用户
        internal_markers = [
            "参考信息：",
            "参考医学信息：",
            "用户问题：",
            "请开始你的专业回答：",
            "【回答要求】"
        ]

        for marker in internal_markers:
            cleaned = cleaned.replace(marker, "")

        # 5. 合并过多空行
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)

        return cleaned.strip()

    def test_connection(self) -> bool:
        """测试 API 连接"""
        try:
            response = requests.get(
                f"{self.api_base}/models",
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                logger.info("✅ DeepSeek API连接正常")
                return True

            logger.error(f"❌ DeepSeek API连接失败: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"❌ DeepSeek连接测试失败: {e}")
            return False
