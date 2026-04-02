"""
知识融合器
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class KnowledgeFuser:
    """知识融合器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fuse(self, kg_result: Dict[str, Any], rag_results: List[Dict],
             user_query: str) -> Dict[str, Any]:
        """
        融合知识图谱结果和RAG检索结果

        Args:
            kg_result: 知识图谱查询结果
            rag_results: RAG检索结果
            user_query: 用户查询

        Returns:
            融合后的结果
        """
        # 提取知识图谱信息
        kg_context = self._extract_kg_context(kg_result)

        # 提取RAG信息
        rag_context = self._extract_rag_context(rag_results)

        # 判断主次
        if self._is_kg_primary(kg_result, rag_results):
            # 以知识图谱为主，RAG补充
            return self._fuse_kg_primary(kg_context, rag_context, user_query)
        else:
            # 以RAG为主，知识图谱补充
            return self._fuse_rag_primary(kg_context, rag_context, user_query)

    def _extract_kg_context(self, kg_result: Dict[str, Any]) -> str:
        """提取知识图谱上下文"""
        if not kg_result or not kg_result.get('success'):
            return "知识图谱中未找到相关信息。"

        data = kg_result.get('data', {})
        if not data:
            return "知识图谱中未找到相关信息。"

        # 根据意图组织信息
        intent = data.get('intent', '')
        result = data.get('result', '')

        if isinstance(result, list):
            result_str = "、".join(result)
        else:
            result_str = str(result)

        return f"【知识图谱】关于{data.get('disease_name', '')}的{intent}信息：{result_str}"

    def _extract_rag_context(self, rag_results: List[Dict]) -> str:
        """提取RAG上下文"""
        if not rag_results:
            return "未检索到相关医学文献。"

        contexts = []
        for i, result in enumerate(rag_results[:2], 1):  
            content = result.get('content', '')
            # 截断过长的内容
            if len(content) > 500:
                content = content[:500] + "..."
            contexts.append(f"文献{i}：{content}")

        return "【医学文献】\n" + "\n\n".join(contexts)

    def _is_kg_primary(self, kg_result: Dict, rag_results: List[Dict]) -> bool:
        """判断是否以知识图谱为主"""
        # 如果知识图谱有完整结果，且置信度高，则以知识图谱为主
        if kg_result and kg_result.get('success'):
            kg_data = kg_result.get('data', {})
            if kg_data and kg_data.get('result'):
                return True

        # 如果RAG结果质量高，数量多，则以RAG为主
        if rag_results and len(rag_results) >= 2:
            return False

        return True  # 默认以知识图谱为主

    def _fuse_kg_primary(self, kg_context: str, rag_context: str, query: str) -> Dict[str, Any]:
        """以知识图谱为主的融合策略"""
        return {
            'primary_source': 'kg',
            'context': f"{kg_context}\n\n{rag_context}",
            'has_kg_info': True,
            'has_rag_info': bool(rag_context and "未检索" not in rag_context)
        }

    def _fuse_rag_primary(self, kg_context: str, rag_context: str, query: str) -> Dict[str, Any]:
        """以RAG为主的融合策略"""
        return {
            'primary_source': 'rag',
            'context': f"{rag_context}\n\n{kg_context}",
            'has_kg_info': bool(kg_context and "未找到" not in kg_context),
            'has_rag_info': True
        }
