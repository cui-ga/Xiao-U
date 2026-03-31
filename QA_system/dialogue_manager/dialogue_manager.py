"""
对话管理器主模块
协调所有对话管理组件
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .dialogue_state import DialogueState, DialogueTurn
from .history_manager import DialogueHistoryManager
from .context_builder import DialogueContextBuilder

logger = logging.getLogger(__name__)


class DialogueManager:
    """对话管理器（主类）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话管理器

        Args:
            config: 系统配置
        """
        self.config = config.get('dialogue', {})
        self.enabled = self.config.get('enabled', True)

        if not self.enabled:
            logger.info("对话管理功能已禁用")
            return

        # 初始化组件
        self.history_manager = DialogueHistoryManager(self.config)
        self.context_builder = DialogueContextBuilder(self.config)

        logger.info("对话管理器初始化完成")

    def is_enabled(self) -> bool:
        """检查对话管理是否启用"""
        return self.enabled

    def process_user_query(self, session_id: str, user_query: str,
                           system_intent: Optional[str] = None) -> Tuple[str, DialogueState]:
        """
        处理用户查询（对话管理入口点）

        Args:
            session_id: 会话ID
            user_query: 用户查询
            system_intent: 系统意图（如果有）

        Returns:
            (处理后的查询, 对话状态对象)
        """
        if not self.enabled:
            return user_query, DialogueState(session_id=session_id)

        # 获取或创建会话
        dialogue_state = self.history_manager.get_session(session_id, create_if_missing=True)
        if not dialogue_state:
            return user_query, DialogueState(session_id=session_id)

        original_query = user_query

        # 步骤1: 检查是否需要重置对话
        if self._should_reset_dialogue(user_query):
            dialogue_state.reset()
            return user_query, dialogue_state

        # 步骤2: 指代消解
        if dialogue_state.core_entity:
            user_query, coref_entities = self.context_builder.resolve_coreference(user_query, dialogue_state)
        else:
            coref_entities = []

        # 保存原始查询（指代消解前）
        processed_query = user_query

        return processed_query, dialogue_state

    def update_dialogue_state(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """
        更新对话状态（在系统生成回答后调用）

        Args:
            session_id: 会话ID
            turn_data: 本轮对话数据，应包含：
                - user_query: 用户查询
                - intent: 识别出的意图
                - entities: 识别出的实体
                - normalized_entities: 规范化后的实体
                - system_response: 系统回答
                - answer_source: 答案来源
                - response_time: 响应时间
                - success: 是否成功

        Returns:
            是否更新成功
        """
        if not self.enabled:
            return False

        dialogue_state = self.history_manager.get_session(session_id, create_if_missing=False)
        if not dialogue_state:
            return False

        # 创建对话轮次对象
        turn = DialogueTurn(
            turn_id=dialogue_state.get_turn_count() + 1,
            user_query=turn_data.get('user_query', ''),
            intent=turn_data.get('intent'),
            entities=turn_data.get('entities', []),
            normalized_entities=turn_data.get('normalized_entities', []),
            system_response=turn_data.get('system_response', ''),
            answer_source=turn_data.get('answer_source'),
            response_time=turn_data.get('response_time', 0.0),
            success=turn_data.get('success', False),
            errors=turn_data.get('errors', [])
        )

        # 添加到对话历史
        dialogue_state.add_turn(turn)
        self.history_manager.update_session(dialogue_state)

        logger.debug(
            f"更新对话状态: session={session_id}, turn_id={turn.turn_id}, intent={turn.intent}, entities={len(turn.entities)}")
        return True

    def enrich_query_for_modules(self, query: str, dialogue_state: DialogueState,
                                 current_intent: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        为后续模块（KG、RAG）丰富查询

        Args:
            query: 当前查询
            dialogue_state: 对话状态
            current_intent: 当前意图

        Returns:
            (丰富后的查询, 补充的实体列表)
        """
        if not self.enabled:
            return query, []

        # 用上下文信息丰富查询
        enriched_query, supplementary_entities = self.context_builder.enrich_query_with_context(
            query, dialogue_state, current_intent
        )

        return enriched_query, supplementary_entities

    def build_retrieval_context(self, query: str, dialogue_state: DialogueState) -> str:
        """
        为RAG检索构建上下文

        Args:
            query: 原始查询
            dialogue_state: 对话状态

        Returns:
            增强后的检索查询
        """
        if not self.enabled:
            return query

        return self.context_builder.build_context_for_retrieval(query, dialogue_state)

    def get_dialogue_context(self, session_id: str, max_turns: int = 3) -> Dict[str, Any]:
        """
        获取对话上下文信息

        Args:
            session_id: 会话ID
            max_turns: 最大历史轮次

        Returns:
            上下文信息字典
        """
        if not self.enabled:
            return {'history': [], 'has_context': False}

        dialogue_state = self.history_manager.get_session(session_id, create_if_missing=False)
        if not dialogue_state:
            return {'history': [], 'has_context': False}

        recent_turns = dialogue_state.get_recent_turns(max_turns)
        history = []

        for turn in recent_turns:
            history.append({
                'user': turn.user_query,
                'system': turn.system_response,
                'intent': turn.intent,
                'timestamp': turn.timestamp.isoformat() if turn.timestamp else None
            })

        return {
            'has_context': len(history) > 0,
            'history': history,
            'core_entity': dialogue_state.core_entity,
            'current_intent': dialogue_state.current_intent,
            'turn_count': dialogue_state.get_turn_count(),
            'session_id': session_id
        }

    def reset_dialogue(self, session_id: str) -> bool:
        """重置指定会话的对话"""
        if not self.enabled:
            return False

        return self.history_manager.reset_session(session_id)

    def end_dialogue(self, session_id: str) -> bool:
        """结束指定会话"""
        if not self.enabled:
            return False

        return self.history_manager.end_session(session_id)

    def _should_reset_dialogue(self, query: str) -> bool:
        """检查是否应该重置对话（用户明确表示要开始新话题）"""
        reset_keywords = [
            '新话题', '新问题', '重新开始', '换个话题', '问别的',
            'reset', 'new topic', 'start over'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in reset_keywords)

    def save_dialogue_history(self, filepath: str) -> bool:
        """保存对话历史到文件"""
        if not self.enabled:
            return False

        return self.history_manager.save_to_file(filepath)

    def load_dialogue_history(self, filepath: str) -> bool:
        """从文件加载对话历史"""
        if not self.enabled:
            return False

        return self.history_manager.load_from_file(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """获取对话管理器统计信息"""
        if not self.enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'active_sessions': self.history_manager.get_active_session_count(),
            'max_history_turns': self.config.get('max_history_turns', 5),
            'session_timeout': self.config.get('session_timeout_seconds', 1800)
        }