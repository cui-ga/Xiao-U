# RAG/data_loader/document_processor.py
"""
文档处理器 - 清洗、分块、预处理文档
"""
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'max_chunk_size': 1000,  # 最大分块大小
            'chunk_overlap': 200,  # 块之间重叠大小
            'min_chunk_size': 50,  # 最小分块大小
            'clean_patterns': [
                r'\s+',  # 多个空白字符
                r'[^\w\u4e00-\u9fff\s,.!?;:，。！？；：、\-()（）]',  # 保留中英文和常用标点
            ]
        }

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理文档列表：清洗、分块、标准化

        Args:
            documents: 原始文档列表

        Returns:
            处理后的文档列表
        """
        processed_docs = []

        for i, doc in enumerate(documents):
            try:
                # 1. 清洗内容
                cleaned_content = self._clean_text(doc.get('content', ''))
                if not cleaned_content or len(cleaned_content) < 20:
                    logger.debug(f"文档 {i} 内容过短，跳过: {doc.get('id', 'unknown')}")
                    continue

                # 2. 分块（如果内容过长）
                chunks = self._chunk_text(cleaned_content)

                # 3. 为每个块创建独立的文档
                for chunk_idx, chunk in enumerate(chunks):
                    processed_doc = {
                        'id': f"{doc.get('id', f'doc_{i}')}_chunk_{chunk_idx}",
                        'content': chunk,
                        'original_id': doc.get('id'),
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'metadata': {
                            **doc.get('metadata', {}),
                            'original_question': doc.get('question', ''),
                            'answer_count': doc.get('answer_count', 0),
                            'chunk_size': len(chunk),
                            'processed': True
                        }
                    }
                    processed_docs.append(processed_doc)

            except Exception as e:
                logger.error(f"处理文档 {i} 失败: {e}")
                continue

        logger.info(f"文档处理完成: {len(documents)} -> {len(processed_docs)} 个处理后的文档块")
        return processed_docs

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str):
            return ""

        cleaned = text

        # 应用清洗模式
        for pattern in self.config.get('clean_patterns', []):
            if pattern == r'\s+':
                # 合并空白字符
                cleaned = re.sub(pattern, ' ', cleaned)
            else:
                # 移除不需要的字符
                cleaned = re.sub(pattern, '', cleaned)

        # 去除首尾空白
        cleaned = cleaned.strip()

        return cleaned

    def _chunk_text(self, text: str) -> List[str]:
        """文本分块"""
        if len(text) <= self.config['max_chunk_size']:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # 计算块结束位置
            end = start + self.config['max_chunk_size']

            if end >= text_length:
                # 最后一块
                chunk = text[start:]
                if len(chunk) >= self.config['min_chunk_size']:
                    chunks.append(chunk)
                break

            # 尝试在句子边界处分割
            sentence_end = self._find_sentence_boundary(text, end)
            if sentence_end > start + self.config['min_chunk_size']:
                end = sentence_end

            chunk = text[start:end]
            if len(chunk) >= self.config['min_chunk_size']:
                chunks.append(chunk)

            # 更新起始位置，考虑重叠
            start = end - self.config['chunk_overlap']

        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """查找句子边界"""
        sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';', '\n\n']

        # 向前查找句子结束符
        for i in range(position, min(position + 100, len(text))):
            if text[i] in sentence_endings:
                return i + 1  # 包含结束符

        # 向后查找
        for i in range(position, max(position - 100, 0), -1):
            if i < len(text) and text[i] in sentence_endings:
                return i + 1

        # 在空白处分割
        for i in range(position, min(position + 50, len(text))):
            if text[i].isspace():
                return i + 1

        return position  # 未找到合适边界，返回原位置

    def validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证文档格式"""
        validated = []

        for i, doc in enumerate(documents):
            # 检查必需字段
            if 'id' not in doc or 'content' not in doc:
                logger.warning(f"文档 {i} 缺少必需字段，跳过")
                continue

            # 检查内容长度
            content = str(doc['content'])
            if len(content) < 10:
                logger.debug(f"文档 {i} 内容过短，跳过: {doc['id']}")
                continue

            validated.append(doc)

        logger.info(f"文档验证完成: {len(documents)} -> {len(validated)} 个有效文档")
        return validated