"""
上下文构建器模块
负责指代消解和查询重写
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class DialogueContextBuilder:
    """对话上下文构建器（处理指代消解和查询扩展）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化上下文构建器

        Args:
            config: 对话配置
        """
        self.config = config
        self.enable_coreference = config.get('enable_coreference_resolution', True)
        self.enable_rewrite = config.get('enable_query_rewrite', True)
        self.core_entity_types = config.get('core_entity_types', ['DISEASE', '疾病'])

        # 指代词映射
        self.coreference_patterns = [
            (
            r'^(它|他|她|这个|那种|该)(的)?(症状|表现|治疗|治疗方法|病因|原因|预防|科室|检查|禁忌|并发症|相关病症|吃什么|不能吃什么|推荐什么药)?[？?]?$',
            '它'),
            (r'^(这|那)(种)?病(的)?(症状|治疗|病因|预防)[？?]?$', '这病'),
            (r'^(其|他的|她的)(症状|治疗|病因)[？?]?$', '其'),
            (r'^(怎么|如何)(治疗|预防|检查|诊断)[？?]?$', '怎么'),
            (r'^(有(什么|哪些)|是什么)(症状|表现|治疗方法|病因|并发症)[？?]?$', '有什么'),
            (r'^(应该|需要)(吃(什么)?|用(什么)?|做(什么)?|看(什么)?)[？?]?$', '应该'),
            (r'^(能不能|是否可以|可不可以)(吃|用|做)[？?]?$', '能不能'),
        ]

    def resolve_coreference(self, query: str, dialogue_state: 'DialogueState') -> Tuple[str, List[Dict[str, Any]]]:
        """
        指代消解：识别查询中的指代词，并用上下文实体替换

        Args:
            query: 用户查询
            dialogue_state: 当前对话状态

        Returns:
            (重写后的查询, 更新的实体列表)
        """
        if not self.enable_coreference or not dialogue_state.core_entity:
            return query, []

        original_query = query
        updated_entities = []

        # 检查是否包含指代词
        for pattern, placeholder in self.coreference_patterns:
            if re.match(pattern, query.strip()):
                core_entity = dialogue_state.core_entity
                entity_name = core_entity.get('entity_name') or core_entity.get('text') or core_entity.get(
                    'normalized_text', '')

                if entity_name:
                    # 根据指代词类型构建完整查询
                    if placeholder in ['它', '这病', '其']:
                        # 提取意图关键词
                        intent_keyword = ''
                        for intent_word in ['症状', '治疗', '病因', '预防', '科室', '检查', '禁忌', '并发症', '吃什么',
                                            '不能吃什么', '推荐什么药']:
                            if intent_word in query:
                                intent_keyword = intent_word
                                break

                        if intent_keyword:
                            rewritten = f"{entity_name}的{intent_keyword}？"
                        else:
                            # 如果没有明确意图关键词，尝试从上一轮继承
                            last_turn = dialogue_state.get_last_turn()
                            if last_turn and last_turn.intent:
                                # 将意图标签转换为中文问题
                                intent_map = {
                                    '临床表现(病症表现)': '症状',
                                    '治疗方法': '治疗方法',
                                    '病因': '病因',
                                    '预防': '预防措施',
                                    '所属科室': '应该看什么科',
                                    '化验/体检方案': '需要做什么检查',
                                    '相关病症': '并发症',
                                    '建议食物': '可以吃什么',
                                    '食物禁忌': '不能吃什么',
                                    '推荐药品': '推荐什么药'
                                }
                                question_part = intent_map.get(last_turn.intent, '相关信息')
                                rewritten = f"{entity_name}的{question_part}？"
                            else:
                                rewritten = f"{entity_name}{query.replace(placeholder, '').strip()}"
                    else:
                        rewritten = f"{entity_name}{query}"

                    logger.info(f"指代消解: '{original_query}' -> '{rewritten}' (核心实体: {entity_name})")

                    # 将核心实体添加到当前查询的实体列表中
                    updated_entities.append(core_entity.copy())

                    return rewritten, updated_entities

        return query, []

    def enrich_query_with_context(self, query: str, dialogue_state: 'DialogueState',
                                  current_intent: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        用上下文信息丰富查询

        Args:
            query: 用户查询
            dialogue_state: 当前对话状态
            current_intent: 当前识别出的意图

        Returns:
            (丰富后的查询, 补充的实体列表)
        """
        if not self.enable_rewrite:
            return query, []

        original_query = query
        enriched_query = query
        supplementary_entities = []

        # 情况1: 查询非常简短，且存在核心实体
        if len(query.strip()) < 6 and dialogue_state.core_entity:
            core_entity = dialogue_state.core_entity
            entity_name = core_entity.get('entity_name') or core_entity.get('text', '')

            if entity_name and '是什么' not in query and '什么是' not in query:
                # 尝试从意图推断完整问题
                if current_intent:
                    intent_question_map = {
                        '临床表现(病症表现)': f"{entity_name}有什么症状？",
                        '治疗方法': f"{entity_name}怎么治疗？",
                        '病因': f"{entity_name}是什么原因引起的？",
                        '预防': f"{entity_name}怎么预防？",
                        '所属科室': f"{entity_name}应该看什么科？",
                        '化验/体检方案': f"{entity_name}需要做什么检查？",
                        '相关病症': f"{entity_name}有哪些并发症？",
                        '建议食物': f"{entity_name}可以吃什么？",
                        '食物禁忌': f"{entity_name}不能吃什么？",
                        '推荐药品': f"{entity_name}推荐什么药？"
                    }

                    if current_intent in intent_question_map:
                        enriched_query = intent_question_map[current_intent]
                        supplementary_entities.append(core_entity.copy())
                        logger.info(
                            f"基于意图丰富查询: '{original_query}' -> '{enriched_query}' (意图: {current_intent})")

        # 情况2: 查询缺少明确实体，但上下文中有实体
        if dialogue_state.core_entity and not self._has_entity_mention(query, dialogue_state.current_entities):
            # 检查查询是否包含常见的医疗问题模式
            medical_keywords = ['症状', '治疗', '病因', '预防', '科室', '检查', '禁忌', '并发症', '吃什么',
                                '不能吃什么', '推荐什么药']
            if any(keyword in query for keyword in medical_keywords):
                core_entity = dialogue_state.core_entity
                entity_name = core_entity.get('entity_name') or core_entity.get('text', '')

                if entity_name and not entity_name in query:
                    enriched_query = f"{entity_name}{query}"
                    supplementary_entities.append(core_entity.copy())
                    logger.info(f"补充实体到查询: '{original_query}' -> '{enriched_query}'")

        return enriched_query, supplementary_entities

    def _has_entity_mention(self, query: str, entities: List[Dict[str, Any]]) -> bool:
        """检查查询中是否提到了实体"""
        if not entities:
            return False

        for entity in entities:
            entity_text = entity.get('text', '')
            entity_name = entity.get('entity_name', '')
            normalized_text = entity.get('normalized_text', '')

            # 检查实体文本是否出现在查询中
            for text in [entity_text, entity_name, normalized_text]:
                if text and text in query:
                    return True

        return False

    def build_context_for_retrieval(self, query: str, dialogue_state: 'DialogueState') -> str:
        """
        为RAG检索构建上下文增强的查询

        Args:
            query: 原始查询
            dialogue_state: 对话状态

        Returns:
            用于检索的增强查询
        """
        context_parts = []

        # 添加上下文信息
        if dialogue_state.core_entity:
            entity_name = dialogue_state.core_entity.get('entity_name') or dialogue_state.core_entity.get('text', '')
            if entity_name:
                context_parts.append(f"疾病：{entity_name}")

        if dialogue_state.current_intent:
            context_parts.append(f"查询类型：{dialogue_state.current_intent}")

        # 添加最近的历史（最多2轮）
        recent_turns = dialogue_state.get_recent_turns(2)
        if recent_turns:
            history_text = []
            for turn in recent_turns:
                if turn.user_query and turn.system_response:
                    history_text.append(f"用户：{turn.user_query}")
                    history_text.append(f"助手：{turn.system_response[:100]}...")

            if history_text:
                context_parts.append("最近对话：" + " | ".join(history_text))

        if context_parts:
            context_str = " ".join(context_parts)
            enhanced_query = f"{query} [上下文：{context_str}]"
            logger.debug(f"RAG检索增强查询: {enhanced_query}")
            return enhanced_query

        return query