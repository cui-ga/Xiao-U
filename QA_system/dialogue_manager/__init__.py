"""
对话管理模块
提供多轮对话支持
"""

from .dialogue_manager import DialogueManager
from .dialogue_state import DialogueState, DialogueTurn
from .history_manager import DialogueHistoryManager
from .context_builder import DialogueContextBuilder

__all__ = [
    'DialogueManager',
    'DialogueState',
    'DialogueTurn',
    'DialogueHistoryManager',
    'DialogueContextBuilder'
]