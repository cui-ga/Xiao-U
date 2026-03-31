# RAG/generator/deepseek_integration.py
import requests
import logging
import time
import re
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class DeepSeekGenerator:
    """优化的DeepSeek生成器 - 专注质量而非速度"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化优化的DeepSeek生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.api_base = config.get('api_base', 'http://localhost:11434/v1')
        self.api_key = config.get('api_key', 'ollama')
        self.model = config.get('model', 'deepseek-r1:1.5b')
        self.max_tokens = config.get('max_tokens', 1024)  # 增加token数
        self.temperature = config.get('temperature', 0.1)
        self.top_p = config.get('top_p', 0.7)
        self.timeout = config.get('timeout', 50)  # 增加到40秒
        self.frequency_penalty = config.get('frequency_penalty', 0.1)
        self.presence_penalty = config.get('presence_penalty', 0.1)
        self.max_retries = config.get('max_retries', 2)  # 重试次数

        logger.info(f"初始化优化的DeepSeek生成器 (模型: {self.model}, 超时: {self.timeout}秒)")

    def generate(self, prompt: str, context: str = None, max_tokens: int = None) -> Optional[str]:
        """
        生成高质量回答

        Args:
            prompt: 用户问题
            context: 相关上下文
            max_tokens: 最大token数

        Returns:
            生成的回答
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        # 构建优化的系统提示
        system_prompt = """你是一个专业的医疗AI助手，小U。请基于用户的问题和相关医学信息，给出准确、完整、专业的回答。

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
- 保持回答的连贯性和完整性"""

        # 构建完整的提示词
        if context and len(context.strip()) > 10:
            # 优化上下文描述，明确告诉模型忽略格式
            user_content = f"""用户问题：{prompt}

参考医学信息：
{context}

【回答要求】
1. 针对用户的具体问题"'{prompt}'"，给出直接的回答
2. 忽略参考信息中的格式标记，只提取有用的医学知识点
3. 用自然、专业的语言组织回答，不要模仿参考信息的格式
4. 回答要完整、准确、结构清晰

请开始你的专业回答："""
        else:
            user_content = f"""【任务说明】
    你是一位专业的医疗助手，需要回答用户关于医疗健康的问题。

    【用户的具体问题】
    {prompt}

    【回答要求】
    请基于你的医学知识，用专业、准确的语言回答上述问题。回答要完整、清晰。

    请开始你的专业回答："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
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

        # 带重试机制的API调用
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"正在调用DeepSeek API (尝试{attempt + 1}/{self.max_retries + 1})，模型: {self.model}")
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
                        logger.error(f"API返回格式错误: 缺少choices字段")
                        if attempt < self.max_retries:
                            time.sleep(1)  # 等待1秒后重试
                            continue
                        return None

                    content = response_data['choices'][0]['message']['content']

                    if not content or not content.strip():
                        logger.warning("API返回空内容")
                        if attempt < self.max_retries:
                            time.sleep(1)
                            continue
                        return None

                    # 清理内容
                    clean_content = self._clean_generated_text(content)
                    logger.info(f"✅ DeepSeek生成成功，耗时: {elapsed:.2f}秒，长度: {len(clean_content)}字符")

                    # 检查内容是否过短，但只记录警告，不重试
                    if len(clean_content) < 10:
                        logger.warning(f"生成内容过短: {len(clean_content)}字符，但将直接返回")

                    return clean_content.strip()  # 直接返回，不重试

                elif response.status_code == 503:  # 服务暂时不可用
                    logger.warning(f"服务暂时不可用 (503)，正在重试... (尝试{attempt + 1}/{self.max_retries + 1})")
                    if attempt < self.max_retries:
                        time.sleep(2)  # 等待2秒
                        continue
                else:
                    logger.error(f"API调用失败: {response.status_code} - {response.text[:200]}")
                    if attempt < self.max_retries:
                        time.sleep(1)
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.error(f"DeepSeek API请求超时 (timeout={self.timeout}s)")
                if attempt < self.max_retries:
                    logger.info(f"正在重试... (尝试{attempt + 1}/{self.max_retries + 1})")
                    time.sleep(2)
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                logger.error(f"无法连接到DeepSeek API: {e}")
                if attempt < self.max_retries:
                    logger.info(f"正在重试连接... (尝试{attempt + 1}/{self.max_retries + 1})")
                    time.sleep(3)
                    continue
                return None
            except Exception as e:
                logger.error(f"DeepSeek生成异常: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                return None

        return None

    def _clean_generated_text(self, text: str) -> str:
        """清理生成的文本，移除中间思考过程"""
        if not text:
            return ""

        # 简单的清理：移除常见的思考标记
        patterns = [
            r'^思考[:：].*?\n',
            r'^推理[:：].*?\n',
            r'^首先，.*?\n',
            r'^然后，.*?\n',
            r'^最后，.*?\n',
            r'^作为一个AI.*?\n',
            r'^作为一名.*?\n',
            r'^好的，.*?\n',
        ]

        cleaned = text.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)

        # 移除多余的空行
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)

        return cleaned.strip()

    def test_connection(self) -> bool:
        """测试API连接 - 简化版"""
        try:
            # 只进行基本的连接测试
            response = requests.get(
                f"{self.api_base}/models",
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                logger.info("✅ DeepSeek API连接正常")
                return True
            else:
                logger.error(f"❌ DeepSeek API连接失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ DeepSeek连接测试失败: {e}")
            return False