# RAG/retriever/vector_store.py
"""
向量存储实现 - 优化版
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import pickle
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储管理器 - 优化版"""

    def __init__(self, embedding_model: Union[str, SentenceTransformer], storage_path: str):
        """
        初始化向量存储

        Args:
            embedding_model: 嵌入模型，可以是模型名称字符串或已加载的SentenceTransformer对象
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.index = None
        self.documents = []
        self.metadatas = []

        # 处理embedding_model参数
        if isinstance(embedding_model, SentenceTransformer):
            # 如果是已加载的模型对象
            self.embedding_model = embedding_model
            self.model_name = None
            logger.info("使用已加载的SentenceTransformer对象")
        elif isinstance(embedding_model, str):
            # 如果是模型名称字符串
            self.model_name = embedding_model
            self.embedding_model = None
            logger.info(f"模型名称: {self.model_name}")
        else:
            raise TypeError(f"embedding_model 必须是字符串或SentenceTransformer对象，但传入的是 {type(embedding_model)}")

    def _get_embedding_model(self) -> SentenceTransformer:
        """获取嵌入模型实例（惰性加载）"""
        if self.embedding_model is None:
            if self.model_name is not None:
                logger.info(f"正在加载嵌入模型: {self.model_name}")
                try:
                    self.embedding_model = SentenceTransformer(self.model_name)
                    logger.info(f"嵌入模型加载成功: {self.model_name}")
                except Exception as e:
                    logger.error(f"加载嵌入模型失败: {e}")
                    raise
            else:
                raise ValueError("未指定模型名称，无法加载嵌入模型")
        return self.embedding_model

    def create_index(self, documents: List[Dict[str, Any]]) -> None:
        """创建向量索引"""
        logger.info(f"开始创建向量索引，文档数: {len(documents)}")

        # 提取文本内容
        texts = [doc['content'] for doc in documents]
        self.documents = documents
        self.metadatas = [doc.get('metadata', {}) for doc in documents]

        # 生成嵌入向量
        embedding_model = self._get_embedding_model()
        embeddings = embedding_model.encode(texts, show_progress_bar=True)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        logger.info(f"向量索引创建完成，维度: {dimension}")

    def save(self) -> None:
        """保存向量索引"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(self.storage_path / "index.faiss"))

        # 保存文档、元数据和模型信息
        save_data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'model_name': self.model_name
        }

        with open(self.storage_path / "documents.pkl", 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"向量索引已保存到: {self.storage_path}")
        logger.info(f"保存的模型名称: {self.model_name}")

    def load(self) -> bool:
        """加载向量索引"""
        try:
            index_path = self.storage_path / "index.faiss"
            docs_path = self.storage_path / "documents.pkl"

            if not index_path.exists():
                logger.warning(f"FAISS索引文件不存在: {index_path}")
                return False
            if not docs_path.exists():
                logger.warning(f"文档文件不存在: {docs_path}")
                return False

            # 加载FAISS索引
            logger.info(f"加载FAISS索引: {index_path}")
            self.index = faiss.read_index(str(index_path))

            # 加载文档、元数据和模型信息
            logger.info(f"加载文档数据: {docs_path}")
            with open(docs_path, 'rb') as f:
                save_data = pickle.load(f)

            self.documents = save_data.get('documents', [])
            self.metadatas = save_data.get('metadatas', [])

            # 如果保存的模型名称不为空，使用保存的名称
            saved_model_name = save_data.get('model_name')
            if saved_model_name and self.model_name is None:
                self.model_name = saved_model_name
                logger.info(f"从保存数据中恢复模型名称: {self.model_name}")

            logger.info(f"向量索引加载成功，文档数: {len(self.documents)}")

            # 验证索引和文档数量匹配
            if self.index.ntotal != len(self.documents):
                logger.warning(f"索引中的向量数({self.index.ntotal})与文档数({len(self.documents)})不匹配")

            return True

        except Exception as e:
            logger.error(f"加载向量索引失败: {e}", exc_info=True)
            return False

    def search(self, query: str, k: int = 10) -> Tuple[List[Dict], List[float]]:
        """搜索相似文档 - 优化相似度计算"""
        if self.index is None or len(self.documents) == 0:
            return [], []

        # 编码查询
        embedding_model = self._get_embedding_model()
        query_embedding = embedding_model.encode([query])

        # 搜索
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        # 处理结果
        results = []
        scores = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents) and idx >= 0:  # 确保索引有效
                # 方法1: 改进的相似度计算
                # 对于归一化后的向量，L2距离在[0,2]之间
                # 将距离转换为相似度：similarity = 1 - distance/2
                similarity = 1.0 - (distance / 2.0)

                # 确保相似度在合理范围内
                similarity = max(0.001, min(1.0, similarity))

                # 方法2: 使用指数转换（对近距离更敏感）
                # similarity = np.exp(-distance)

                results.append(self.documents[idx])
                scores.append(similarity)

        return results, scores

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if self.index is None:
            return {"status": "未初始化"}

        return {
            "status": "已加载",
            "document_count": len(self.documents),
            "vector_count": self.index.ntotal,
            "dimension": self.index.d if hasattr(self.index, 'd') else "未知",
            "model_name": self.model_name
        }