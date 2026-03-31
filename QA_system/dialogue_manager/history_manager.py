"""
历史管理器模块
负责对话状态的存储、检索和清理
"""
import json
import time
import threading
from typing import Dict, Optional, List,Any
from pathlib import Path
from datetime import datetime
import logging

from .dialogue_state import DialogueState

logger = logging.getLogger(__name__)


class DialogueHistoryManager:
    """对话历史管理器（基于内存，可扩展为基于数据库）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化历史管理器

        Args:
            config: 对话配置
        """
        self.config = config
        self.sessions: Dict[str, DialogueState] = {}
        self.max_history_turns = config.get('max_history_turns', 5)
        self.session_timeout = config.get('session_timeout_seconds', 1800)

        # 清理过期会话的线程
        self.cleanup_interval = 300  # 每5分钟清理一次
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()

        logger.info(f"对话历史管理器初始化完成，最大历史轮次: {self.max_history_turns}, 超时: {self.session_timeout}秒")

    def get_session(self, session_id: str, create_if_missing: bool = True) -> Optional[DialogueState]:
        """
        获取对话会话

        Args:
            session_id: 会话ID
            create_if_missing: 如果不存在是否创建

        Returns:
            对话状态对象
        """
        # 清理过期会话
        self._cleanup_session_if_expired(session_id)

        if session_id in self.sessions:
            return self.sessions[session_id]

        if create_if_missing:
            new_session = DialogueState(session_id=session_id)
            self.sessions[session_id] = new_session
            logger.debug(f"创建新对话会话: {session_id}")
            return new_session

        return None

    def update_session(self, session: DialogueState) -> None:
        """更新会话"""
        self.sessions[session.session_id] = session

        # 限制历史记录长度
        if len(session.history) > self.max_history_turns * 2:  # 保留一些缓冲
            session.history = session.history[-self.max_history_turns:]
            logger.debug(f"修剪会话历史: {session.session_id}，保留最近{self.max_history_turns}轮")

    def reset_session(self, session_id: str) -> bool:
        """重置指定会话"""
        if session_id in self.sessions:
            self.sessions[session_id].reset()
            logger.info(f"重置对话会话: {session_id}")
            return True
        return False

    def end_session(self, session_id: str) -> bool:
        """结束指定会话"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            logger.info(f"结束对话会话: {session_id}")
            return True
        return False

    def _cleanup_session_if_expired(self, session_id: str) -> None:
        """清理过期会话（如果需要）"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.is_expired(self.session_timeout):
                logger.info(f"清理过期会话: {session_id}")
                del self.sessions[session_id]

    def _cleanup_expired_sessions(self) -> None:
        """定期清理所有过期会话"""
        while True:
            time.sleep(self.cleanup_interval)
            try:
                expired_sessions = []
                for session_id, session in self.sessions.items():
                    if session.is_expired(self.session_timeout):
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    logger.debug(f"定期清理过期会话: {session_id}")
                    del self.sessions[session_id]

                if expired_sessions:
                    logger.info(f"定期清理完成，清理了 {len(expired_sessions)} 个过期会话")
            except Exception as e:
                logger.error(f"清理过期会话时出错: {e}")

    def save_to_file(self, filepath: str) -> bool:
        """保存所有会话到文件（用于持久化）"""
        try:
            data = {
                'sessions': {sid: session.to_dict() for sid, session in self.sessions.items()},
                'saved_at': datetime.now().isoformat()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"对话历史已保存到文件: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存对话历史到文件失败: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """从文件加载会话"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.sessions.clear()
                for session_id, session_data in data.get('sessions', {}).items():
                    self.sessions[session_id] = DialogueState.from_dict(session_data)

                logger.info(f"从文件加载对话历史: {filepath}，加载了 {len(self.sessions)} 个会话")
                return True
        except Exception as e:
            logger.error(f"从文件加载对话历史失败: {e}")

        return False

    def get_active_session_count(self) -> int:
        """获取活跃会话数量"""
        return len([s for s in self.sessions.values() if s.is_active])