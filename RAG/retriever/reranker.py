# RAG/retriever/reranker.py
"""
重排序器 - 对检索结果进行重排序
"""
from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器

        Args:
            model_name: 重排序模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载重排序模型"""
        try:
            logger.info(f"正在加载重排序模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("重排序模型加载成功")
        except Exception as e:
            logger.error(f"加载重排序模型失败: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """改进的重排序方法"""
        if not documents or not self.model:
            return documents

        if top_k is None:
            top_k = len(documents)

        try:
            # 准备查询-文档对
            pairs = [(query, doc.get('content', '')) for doc in documents]

            # 计算相关性分数
            scores = self.model.predict(pairs, convert_to_numpy=True)

            # 将分数附加到文档
            for doc, score in zip(documents, scores):
                # 重排序分数通常在0-1之间
                relevance_score = float(score)
                doc['relevance_score'] = relevance_score

                # 计算综合分数（结合向量分数和重排序分数）
                vector_score = doc.get('score', 0)

                # 加权平均：重排序权重更高
                if vector_score > 0:
                    # 归一化处理
                    final_score = 0.7 * relevance_score + 0.3 * vector_score
                else:
                    final_score = relevance_score

                doc['final_score'] = final_score

            # 按综合分数降序排序
            sorted_docs = sorted(documents, key=lambda x: x.get('final_score', 0), reverse=True)

            # 记录重排序结果
            if sorted_docs:
                logger.debug(f"重排序完成: 最高分={sorted_docs[0].get('final_score', 0):.4f}")

            return sorted_docs[:top_k]

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return documents

    def batch_rerank(self, queries: List[str], documents_list: List[List[Dict]]) -> List[List[Dict]]:
        """批量重排序"""
        results = []
        for query, docs in zip(queries, documents_list):
            reranked = self.rerank(query, docs)
            results.append(reranked)
        return results