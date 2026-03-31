# RAG/utils/embedding_utils.py
"""
嵌入工具函数
"""
import numpy as np
from typing import List, Union
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingUtils:
    """嵌入工具类"""

    def __init__(self, model_name: str = "moka-ai/m3e-base"):
        """
        初始化嵌入工具

        Args:
            model_name: 嵌入模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载嵌入模型"""
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"嵌入模型加载成功，维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 文本或文本列表
            **kwargs: 传递给模型的参数

        Returns:
            向量数组
        """
        if not self.model:
            raise ValueError("嵌入模型未加载")

        if isinstance(texts, str):
            texts = [texts]

        # 默认参数
        encode_kwargs = {
            'normalize_embeddings': True,
            'show_progress_bar': len(texts) > 10,
            **kwargs
        }

        try:
            embeddings = self.model.encode(texts, **encode_kwargs)
            return embeddings
        except Exception as e:
            logger.error(f"编码文本失败: {e}")
            # 返回零向量作为降级方案
            if isinstance(texts, list):
                dimension = self.model.get_sentence_embedding_dimension()
                return np.zeros((len(texts), dimension))
            else:
                dimension = self.model.get_sentence_embedding_dimension()
                return np.zeros((1, dimension))

    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算向量相似度（余弦相似度）

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            相似度分数 (-1 到 1)
        """
        if vec1.ndim == 1:
            vec1 = vec1.reshape(1, -1)
        if vec2.ndim == 1:
            vec2 = vec2.reshape(1, -1)

        # 余弦相似度
        dot_product = np.dot(vec1, vec2.T)
        norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)

        similarity = dot_product / (norm1 * norm2.T + 1e-8)

        return float(similarity[0, 0])

    def find_most_similar(self, query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int = 5) -> tuple:
        """
        查找最相似的文档

        Args:
            query_vec: 查询向量
            doc_vecs: 文档向量矩阵
            top_k: 返回数量

        Returns:
            (索引列表, 分数列表)
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # 计算相似度
        similarities = np.dot(doc_vecs, query_vec.T).flatten()

        # 获取top_k
        if top_k > len(similarities):
            top_k = len(similarities)

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        return top_indices.tolist(), top_scores.tolist()