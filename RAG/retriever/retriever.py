# RAG/retriever/retriever.py
"""
检索器实现 - 优化版
"""
from typing import List, Dict, Any, Tuple
import logging
from .vector_store import VectorStore
from .reranker import Reranker
import numpy as np

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG检索器 - 优化版"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化向量存储
        self.vector_store = VectorStore(
            embedding_model=config['embedding_model'],
            storage_path=config['vector_db_path']
        )

        # 初始化重排序器
        self.reranker = None
        if config.get('enable_reranking', True):
            self.reranker = Reranker(config['reranker_model'])

        # 加载索引
        if not self.vector_store.load():
            logger.warning("向量索引不存在，需要重新构建")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        检索相关文档 - 优化版

        Args:
            query: 查询文本
            top_k: 返回文档数量

        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = self.config.get('top_k_final', 5)

        # 初步检索更多文档
        initial_k = self.config.get('top_k_initial', 50)
        documents, scores = self.vector_store.search(query, initial_k)

        if not documents:
            logger.info(f"未检索到任何文档: {query}")
            return []

        logger.info(f"初步检索到 {len(documents)} 个文档，最高分: {max(scores) if scores else 0:.4f}")

        # 将分数附加到文档
        for doc, score in zip(documents, scores):
            doc['vector_score'] = score

        # 过滤结果 - 使用动态阈值
        filtered_docs = self._filter_with_dynamic_threshold(documents, scores, query)

        if not filtered_docs:
            # 如果没有文档通过过滤，返回相似度最高的几个
            logger.warning("无文档通过过滤，返回相似度最高的文档")
            for i in range(min(3, len(documents))):
                documents[i]['vector_score'] = scores[i]
                filtered_docs.append(documents[i])

        # 重排序
        if self.reranker and filtered_docs and len(filtered_docs) > 1:
            logger.info(f"对 {len(filtered_docs)} 个文档进行重排序")
            reranked_docs = self.reranker.rerank(query, filtered_docs)
        else:
            reranked_docs = filtered_docs

        # 记录最终结果
        if reranked_docs:
            final_scores = [doc.get('final_score', doc.get('vector_score', 0)) for doc in reranked_docs]
            logger.info(
                f"最终返回 {len(reranked_docs[:top_k])} 个文档，分数范围: {min(final_scores):.4f}-{max(final_scores):.4f}")

        return reranked_docs[:top_k]

    def _filter_with_dynamic_threshold(self, documents: List[Dict], scores: List[float], query: str) -> List[Dict]:
        """使用动态阈值过滤结果，并限制重排序文档数"""
        if not documents or not scores:
            return []

        # 按分数从高到低排序
        import numpy as np
        sorted_indices = np.argsort(scores)[::-1]

        # 关键优化：限制进入重排序的文档数量
        max_for_rerank = 8  # 建议值：10-20，在质量和速度间平衡

        filtered = []
        for i in range(min(max_for_rerank, len(sorted_indices))):
            idx = sorted_indices[i]
            documents[idx]['score'] = scores[idx]
            filtered.append(documents[idx])

        logger.info(f"动态过滤: {len(documents)} -> {len(filtered)} 个文档（限制最多{max_for_rerank}个）")
        return filtered

    def _is_relevant(self, document: Dict, query: str) -> bool:
        """判断文档是否相关 - 更宽松的检查"""
        text = document.get('content', '').lower()
        query_lower = query.lower()

        # 检查是否有共同的关键词（长度大于2）
        query_terms = [term for term in query_lower.split() if len(term) > 2]

        for term in query_terms:
            if term in text:
                return True

        # 如果查询包含特定医学术语，进行子串匹配
        medical_terms = ['糖尿病', '高血压', '感冒', '症状', '治疗', '预防']
        for term in medical_terms:
            if term in query and term in text:
                return True

        return True  # 暂时放宽，让重排序模型决定

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加新文档到向量存储"""
        self.vector_store.create_index(documents)
        self.vector_store.save()

# 在VectorRetriever类中添加缓存

from functools import lru_cache
from datetime import datetime, timedelta

class VectorRetriever:
    def __init__(self, config: Dict[str, Any]):
        # ... 原有初始化代码 ...
        self._cache = {}  # 添加缓存字典
        self._cache_time = {}  # 缓存时间
        self._cache_ttl = 300  # 缓存有效时间（秒）

    def retrieve(self, query: str, top_k: int = 2, use_cache: bool = True) -> List[Dict]:
        """检索文档，带缓存机制"""
        cache_key = f"{query}_{top_k}"

        # 检查缓存
        if use_cache and cache_key in self._cache:
            cache_time = self._cache_time.get(cache_key)
            if cache_time and datetime.now() - cache_time < timedelta(seconds=self._cache_ttl):
                logger.info(f"从缓存获取结果: {cache_key}")
                return self._cache[cache_key]

        # 原有检索逻辑...
        results = self._retrieve_impl(query, top_k)

        # 存入缓存
        if use_cache and results:
            self._cache[cache_key] = results
            self._cache_time[cache_key] = datetime.now()

        return results