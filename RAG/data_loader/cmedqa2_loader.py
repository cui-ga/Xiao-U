# RAG/data_loader/cmedqa2_loader.py
"""
cMedQA2 数据加载器
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CMEDQA2Loader:
    """cMedQA2 数据集加载器"""
    
    def __init__(self, data_path: str):
        """
        初始化加载器
        Args:
            data_path: cMedQA2数据目录路径
        """
        self.data_path = Path(data_path)
        
    def load_data(self) -> List[Dict[str, Any]]:
        """
        加载cMedQA2数据集，合并同一问题的多个答案
        
        Returns:
            文档列表，每个文档包含合并后的问题和所有答案
        """
        try:
            # 1. 定位文件
            question_file = self.data_path / "question.csv"
            answer_file = self.data_path / "answer.csv"
            
            if not question_file.exists():
                raise FileNotFoundError(f"问题文件不存在: {question_file}")
            if not answer_file.exists():
                raise FileNotFoundError(f"答案文件不存在: {answer_file}")
            
            logger.info(f"正在加载问题文件: {question_file}")
            questions_df = pd.read_csv(question_file, encoding='utf-8')
            
            logger.info(f"正在加载答案文件: {answer_file}")
            answers_df = pd.read_csv(answer_file, encoding='utf-8')
            
            # 2. 确保列名正确（处理可能的列名变体）
            questions_df = self._standardize_columns(questions_df, is_question=True)
            answers_df = self._standardize_columns(answers_df, is_question=False)
            
            # 3. 合并数据（一个问题对应多个答案）
            merged_df = pd.merge(questions_df, answers_df, on='question_id', how='inner', suffixes=('_q', '_a'))
            
            if merged_df.empty:
                logger.warning("合并后的数据为空，请检查数据文件")
                return []
            
            logger.info(f"数据合并完成，共 {len(merged_df)} 行")
            
            # 4. 按问题ID分组，合并多个答案
            documents = []
            for q_id, group in merged_df.groupby('question_id'):
                # 获取问题内容（所有行的问题内容相同）
                question_text = group.iloc[0]['content_q']
                
                # 获取该问题的所有答案
                all_answers = []
                for i, (_, row) in enumerate(group.iterrows()):
                    answer_text = str(row['content_a']).strip()
                    if answer_text:  # 过滤空答案
                        all_answers.append(f"【答案{i+1}】{answer_text}")
                
                if not all_answers:
                    continue  # 跳过没有有效答案的问题
                
                # 构建完整文档内容
                combined_content = f"问题：{question_text}\n\n" + "\n\n".join(all_answers)
                
                # 构建文档字典
                doc = {
                    'id': str(q_id),
                    'content': combined_content,
                    'question': question_text,
                    'answers': all_answers,
                    'answer_count': len(all_answers),
                    'metadata': {
                        'source': 'cMedQA2',
                        'question_id': int(q_id),
                        'question_length': len(question_text),
                        'avg_answer_length': sum(len(ans) for ans in all_answers) / len(all_answers)
                    }
                }
                documents.append(doc)
            
            logger.info(f"成功处理 {len(documents)} 个问题文档（合并了多个答案）")
            return documents
            
        except Exception as e:
            logger.error(f"加载cMedQA2数据失败: {e}", exc_info=True)
            return []
    
    def _standardize_columns(self, df: pd.DataFrame, is_question: bool) -> pd.DataFrame:
        """标准化DataFrame列名"""
        df = df.copy()
        
        # 处理列名变体
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            
            if is_question:
                # 问题文件列名标准化
                if 'question_id' in col_lower or 'qid' in col_lower:
                    column_mapping[col] = 'question_id'
                elif 'content' in col_lower or 'question' in col_lower or 'title' in col_lower:
                    column_mapping[col] = 'content_q'
            else:
                # 答案文件列名标准化
                if 'question_id' in col_lower or 'qid' in col_lower:
                    column_mapping[col] = 'question_id'
                elif 'ans_id' in col_lower or 'aid' in col_lower or 'answer_id' in col_lower:
                    column_mapping[col] = 'ans_id'
                elif 'content' in col_lower or 'answer' in col_lower:
                    column_mapping[col] = 'content_a'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    def get_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not documents:
            return {}
        
        total_questions = len(documents)
        total_answers = sum(doc['answer_count'] for doc in documents)
        avg_answers_per_question = total_answers / total_questions if total_questions > 0 else 0
        
        # 文本长度统计
        question_lengths = [len(doc['question']) for doc in documents]
        content_lengths = [len(doc['content']) for doc in documents]
        
        stats = {
            'total_questions': total_questions,
            'total_answers': total_answers,
            'avg_answers_per_question': round(avg_answers_per_question, 2),
            'avg_question_length': round(sum(question_lengths) / total_questions, 2) if total_questions > 0 else 0,
            'avg_content_length': round(sum(content_lengths) / total_questions, 2) if total_questions > 0 else 0,
            'min_question_length': min(question_lengths) if question_lengths else 0,
            'max_question_length': max(question_lengths) if question_lengths else 0,
        }
        
        logger.info(f"数据统计: {stats}")
        return stats