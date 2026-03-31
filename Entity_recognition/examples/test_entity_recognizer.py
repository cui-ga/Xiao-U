import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
entity_recognition_root = os.path.dirname(current_script_dir)
project_root = os.path.dirname(entity_recognition_root)

sys.path.insert(0, project_root)

from Entity_recognition.config.entity_config import EntityConfig
from Entity_recognition.entity_recognition.entity_recognizer import EntityRecognizer

import json


def main():
    # 1. 创建配置
    config = EntityConfig(
        neo4j_uri="neo4j://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password="12345678"
    )

    # 2. 创建实体识别器
    recognizer = EntityRecognizer(config)

    # 3. 测试文本
    test_cases = [
        "糖尿病有什么症状？",
        "高血压怎么治疗？需要吃什么药？",
        "心脏病应该看什么科？需要做什么检查？",
        "我最近头痛、发烧，是不是感冒了？",
        "阿司匹林是治什么的？可以和阿莫西林一起吃吗？",
        "胃痛应该吃什么食物？忌吃什么？",
        "糖尿病的治疗方法有哪些？",
        "CT检查能查出肺癌吗？",
    ]

    # 4. 测试
    print("实体识别测试开始...")
    print("=" * 60)

    all_results = {}

    for text in test_cases:
        print(f"\n查询: {text}")
        entities = recognizer.recognize(text)

        if entities:
            print(f"识别到 {len(entities)} 个实体:")
            for entity in entities:
                entity_info = f"  - {entity['text']} ({entity['type']})"
                if entity.get('linked'):
                    entity_info += f" → 链接到: {entity.get('normalized_text', entity['text'])}"
                entity_info += f" [置信度: {entity.get('confidence', 0):.2f}]"
                print(entity_info)
        else:
            print("  未识别到实体")

        all_results[text] = entities

    # 5. 保存结果
    output_path = "entity_recognition_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        # 转换不可序列化的对象
        serializable_results = {}
        for text, entities in all_results.items():
            serializable_results[text] = []
            for entity in entities:
                serializable_entity = {}
                for key, value in entity.items():
                    if key not in ['processing_time', 'text_length']:  # 保留基本类型
                        serializable_entity[key] = value
                serializable_results[text].append(serializable_entity)

        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")

    # 6. 关闭识别器
    recognizer.close()

    print("\n测试完成！")


if __name__ == "__main__":
    main()