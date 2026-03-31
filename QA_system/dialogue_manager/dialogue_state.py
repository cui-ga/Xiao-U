"""
对话状态定义模块
定义对话状态、对话轮次的数据结构
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """单轮对话记录"""
    turn_id: int
    user_query: str
    timestamp: datetime = field(default_factory=datetime.now)

    # 识别结果
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    normalized_entities: List[Dict[str, Any]] = field(default_factory=list)

    # 系统响应
    system_response: Optional[str] = None
    answer_source: Optional[str] = None
    response_time: float = 0.0

    # 处理元数据
    success: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于JSON序列化）"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTurn':
        """从字典创建实例"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DialogueState:
    """对话状态（一个完整的会话）"""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # 对话历史
    history: List[DialogueTurn] = field(default_factory=list)

    # 当前对话的核心信息（用于快速访问和指代消解）
    current_intent: Optional[str] = None
    current_entities: List[Dict[str, Any]] = field(default_factory=list)
    core_entity: Optional[Dict[str, Any]] = None  # 当前对话的核心实体（如主要讨论的疾病）

    # 会话属性
    is_active: bool = True

    def add_turn(self, turn: DialogueTurn) -> None:
        """添加一轮对话"""
        self.history.append(turn)
        self.last_activity = datetime.now()

        # 更新当前状态
        self.current_intent = turn.intent
        self.current_entities = turn.normalized_entities or turn.entities

        # 尝试确定核心实体（优先级：疾病 > 症状 > 药品）
        if self.current_entities:
            for entity_type in ['DISEASE', '疾病']:
                disease_entities = [e for e in self.current_entities
                                    if e.get('type') == entity_type]
                if disease_entities:
                    self.core_entity = disease_entities[0]
                    break

            if not self.core_entity:
                for entity_type in ['SYMPTOM', '症状']:
                    symptom_entities = [e for e in self.current_entities
                                        if e.get('type') == entity_type]
                    if symptom_entities:
                        self.core_entity = symptom_entities[0]
                        break

            if not self.core_entity and self.current_entities:
                self.core_entity = self.current_entities[0]

    def get_last_turn(self) -> Optional[DialogueTurn]:
        """获取上一轮对话"""
        if len(self.history) > 0:
            return self.history[-1]
        return None

    def get_turn_count(self) -> int:
        """获取对话轮次"""
        return len(self.history)

    def get_recent_turns(self, n: int = 3) -> List[DialogueTurn]:
        """获取最近n轮对话"""
        return self.history[-n:] if self.history else []

    def is_expired(self, timeout_seconds: int) -> bool:
        """检查会话是否已过期"""
        if not self.is_active:
            return True
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds

    def reset(self) -> None:
        """重置对话状态（开始新话题）"""
        self.history.clear()
        self.current_intent = None
        self.current_entities.clear()
        self.core_entity = None
        self.last_activity = datetime.now()
        logger.info(f"对话状态已重置: session_id={self.session_id}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        data['history'] = [turn.to_dict() for turn in self.history]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueState':
        """从字典创建实例"""
        # 处理时间字段
        for time_field in ['created_at', 'last_activity']:
            if time_field in data and isinstance(data[time_field], str):
                data[time_field] = datetime.fromisoformat(data[time_field])

        # 处理历史记录
        if 'history' in data and isinstance(data['history'], list):
            history_data = data.pop('history')
            instance = cls(**data)
            instance.history = [DialogueTurn.from_dict(turn_data) for turn_data in history_data]
            return instance

        return cls(**data)