# RAG/utils/model_cache.py
"""
模型缓存管理器 - 单例模式避免重复加载模型
"""
import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from typing import Optional

logger = logging.getLogger(__name__)


class ModelCache:
    _instance = None
    _embedding_model = None
    _reranker_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    def get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """获取或创建嵌入模型实例"""
        if self._embedding_model is None:
            logger.info(f"加载嵌入模型: {model_name}")
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    def get_reranker_model(self, model_name: str) -> CrossEncoder:
        """获取或创建重排序模型实例"""
        if self._reranker_model is None:
            logger.info(f"加载重排序模型: {model_name}")
            self._reranker_model = CrossEncoder(model_name)
        return self._reranker_model

    def clear_cache(self):
        """清理模型缓存"""
        self._embedding_model = None
        self._reranker_model = None
        import gc
        gc.collect()


# 使用示例
model_cache = ModelCache()
embedding_model = model_cache.get_embedding_model("moka-ai/m3e-base")