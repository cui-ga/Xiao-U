import os
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

sys.path.insert(0, os.path.join(project_root, 'Entity_linking'))
from entity_normalizer import EntityNormalizer

from KG_query.kg_querier import KnowledgeGraphQuerier
from KG_query.query_formatter import QueryResultFormatter


def simple_test():
    """简化测试"""
    print("简化知识图谱查询测试")
    print("=" * 60)

    config = {
        'neo4j_uri': 'bolt://127.0.0.1:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': '12345678',
        'neo4j_database': 'neo4j',
        'enable_debug': True
    }

    # 1. 创建实体规范化器
    print("1. 初始化实体规范化器...")
    entity_normalizer = EntityNormalizer(config)

    # 2. 创建知识图谱查询器
    print("2. 初始化知识图谱查询器...")
    kg_querier = KnowledgeGraphQuerier(config)

    # 3. 只测试属性查询（不测试关系查询）
    test_cases = [
        {
            "query": "糖尿病是什么？",
            "intent": "定义",
            "entity_text": "糖尿病"
        },
        {
            "query": "糖尿病的治愈率是多少？",
            "intent": "治愈率",
            "entity_text": "糖尿病"
        },
        {
            "query": "糖尿病的病因是什么？",
            "intent": "病因",
            "entity_text": "糖尿病"
        },
        {
            "query": "糖尿病怎么预防？",
            "intent": "预防",
            "entity_text": "糖尿病"
        }
    ]

    print("\n3. 执行简化测试...")
    print("-" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['query']}")
        print(f"  意图: {test_case['intent']}")
        print(f"  实体: {test_case['entity_text']}")

        try:
            # 实体规范化
            mock_entity = {
                'text': test_case['entity_text'],
                'type': 'DISEASE',
                'start': 0,
                'end': len(test_case['entity_text']),
                'confidence': 0.95,
                'source': 'test'
            }

            normalized_entities = entity_normalizer.normalize_entities([mock_entity], test_case['query'])

            if not normalized_entities or not normalized_entities[0].get('normalized'):
                print(f"  ❌ 实体规范化失败")
                continue

            entity = normalized_entities[0]
            print(f"  ✅ 实体规范化成功: {entity.get('kg_name')}")

            # 知识图谱查询
            result = kg_querier.query_by_intent(
                intent=test_case['intent'],
                entities=[entity],
                query_text=test_case['query']
            )

            if result.get('success'):
                print(f"  ✅ 知识图谱查询成功")
                print(f"     执行时间: {result.get('execution_time', 0):.3f}秒")

                # 格式化结果
                formatted = QueryResultFormatter.format_by_intent(test_case['intent'], result.get('data', {}))
                answer = formatted.get('answer', '无结果')
                print(f"  📋 回答: {answer[:100]}...")
            else:
                print(f"  ❌ 知识图谱查询失败")
                print(f"     错误: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"  ❌ 测试异常: {type(e).__name__}: {str(e)}")

    # 4. 显示统计信息
    print("\n" + "=" * 60)
    print("4. 查询统计信息:")
    stats = kg_querier.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 5. 清理资源
    print("\n5. 清理资源...")
    entity_normalizer.close()
    kg_querier.close()

    print("\n测试完成!")


if __name__ == "__main__":
    simple_test()