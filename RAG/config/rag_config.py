# RAG/config/rag_config.py
"""
RAG模块配置文件 - 优化版
"""
import os
from pathlib import Path

# 基础路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAG_ROOT = PROJECT_ROOT / "RAG"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
CMEDQA2_DATA_PATH = DATA_DIR / "cMedQA2"
VECTOR_DB_PATH = RAG_ROOT / "vector_db"

# 模型配置
EMBEDDING_MODEL = "moka-ai/m3e-base"
RERANKER_MODEL = "BAAI/bge-reranker-base"
DEEPSEEK_MODEL = "deepseek-r1:1.5b"  # 修改为1.5b版本

# 检索配置 - 关键优化
RETRIEVE_CONFIG = {
    "top_k_initial": 20,      # 优化：减少初步检索数量
    "top_k_final": 2,         # 最终返回数量
    "similarity_threshold": 0.001,
    "max_doc_length": 1000,
    "enable_reranking": True,
    "min_score_for_rerank": 0.0001,
}

# DeepSeek配置 - 针对1.5b模型优化
DEEPSEEK_CONFIG = {
    "api_base": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "deepseek-r1:1.5b",  # 修改为1.5b
    "max_tokens": 512,         # 减少生成长度，加快速度
    "temperature": 0.1,        # 降低随机性，加快速度
    "top_p": 0.7,
    "timeout": 60,            # 缩短超时时间
    "frequency_penalty": 0.1,  # 减少重复
    "presence_penalty": 0.1,   # 提高多样性
}

# 提示模板配置 - 优化为快速回答模式
PROMPT_TEMPLATES = {
    "medical_qa": """你是一个专业的医疗AI助手。请根据以下医学知识和用户问题，提供准确、简洁、直接的回答。

相关医学知识：
{context}

用户问题：{question}

请直接、简洁地回答，不需要过多解释，控制在150字以内。
请快速思考，不要思考太长时间。

你的回答：""",

    "knowledge_fusion": """请将以下知识图谱信息和检索到的医学知识融合，简洁回答用户问题：

【知识图谱信息】
{kg_context}

【相关医学文献】
{rag_context}

用户问题：{question}

请快速整合以上信息，给出直接、简洁的回答：
1. 先给出核心答案
2. 如果有补充信息，简要说明
3. 控制在100字以内
不要思考太长时间，请快速回答。""",

    "rag_augment": """请基于以下核心答案和相关医学信息，提供一个更全面的简洁回答：

【核心答案】
{kg_answer}

【补充医学信息】
{rag_info}

用户问题：{question}

请快速整合信息，提供一个简洁、完整的回答（100字以内）。
如果补充信息与核心答案一致，可以直接使用核心答案。
请快速思考，不要花费太长时间。""",

    "quick_answer": """请根据以下信息，快速、直接地回答用户问题：

相关信息：
{context}

用户问题：{question}

请直接给出答案，不需要解释，控制在80字以内。
请快速思考，立即回答。"""
}