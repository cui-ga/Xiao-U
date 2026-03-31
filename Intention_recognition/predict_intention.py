"""
医疗意图识别预测脚本
使用训练好的模型进行预测
"""
import os
import sys
from pathlib import Path
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalIntentPredictor:
    """医疗意图预测器"""

    def __init__(self, model_path=None):
        """
        初始化预测器

        Args:
            model_path: 模型路径，默认为 saved_models/best_model
        """
        if model_path is None:
            model_path = Path(__file__).parent / "saved_models" / "best_model"

        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.load_model()

        logger.info(f"医疗意图预测器初始化完成，使用设备: {self.device}")

    # 修改 load_model 方法
    def load_model(self):
        """加载模型和配置"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")

            # 加载配置
            config_path = self.model_path / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

            # 修复：处理 label_mapping 键不存在的情况
            if "label_mapping" in self.config:
                self.label_mapping = self.config["label_mapping"]
            else:
                # 从 id2label 和 label2id 构建 label_mapping
                id2label = self.config.get("id2label", {})
                label2id = self.config.get("label2id", {})

                if not id2label or not label2id:
                    # 尝试从 model_config 中获取
                    model_config = self.config.get("model_config", {})
                    id2label = model_config.get("id2label", {})
                    label2id = model_config.get("label2id", {})

                if id2label and label2id:
                    self.label_mapping = {
                        "id2label": id2label,
                        "label2id": label2id
                    }
                else:
                    # 如果都没有，创建默认映射
                    logger.warning("配置文件中没有找到 label_mapping，创建默认映射")
                    self.label_mapping = {
                        "label2id": {
                            "定义": 0, "病因": 1, "预防": 2,
                            "临床表现(病症表现)": 3, "相关病症": 4,
                            "治疗方法": 5, "所属科室": 6, "传染性": 7,
                            "治愈率": 8, "禁忌": 9, "化验/体检方案": 10,
                            "治疗时间": 11, "其他": 12
                        },
                        "id2label": {
                            "0": "定义", "1": "病因", "2": "预防",
                            "3": "临床表现(病症表现)", "4": "相关病症",
                            "5": "治疗方法", "6": "所属科室", "7": "传染性",
                            "8": "治愈率", "9": "禁忌", "10": "化验/体检方案",
                            "11": "治疗时间", "12": "其他"
                        }
                    }

            # 加载模型和分词器
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

            # 获取模型配置
            self.model_config = self.config.get("model_config", {})
            self.max_length = self.model_config.get("max_length", 128)

            logger.info(f"模型加载成功，共有 {len(self.label_mapping['id2label'])} 个意图类别")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def predict(self, text, max_length=128, return_probs=False, top_k=3):
        """
        预测单个文本的意图

        Args:
            text: 输入文本
            max_length: 最大序列长度
            return_probs: 是否返回概率分布
            top_k: 返回top_k个预测结果

        Returns:
            预测结果字典
        """
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # 移到设备
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            predicted_id = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0, predicted_id].item()

        # 获取标签名称
        predicted_label = self.label_mapping["id2label"].get(str(predicted_id), "未知")

        result = {
            'text': text,
            'predicted_intent': predicted_label,
            'predicted_id': predicted_id,
            'confidence': confidence
        }

        if return_probs or top_k > 1:
            # 获取所有类别的概率
            all_probs = probabilities[0].cpu().numpy()
            all_results = []

            for i, prob in enumerate(all_probs):
                label = self.label_mapping["id2label"].get(str(i), f"类别{i}")
                all_results.append({
                    'label': label,
                    'id': i,
                    'probability': float(prob)
                })

            # 按概率排序
            all_results.sort(key=lambda x: x['probability'], reverse=True)

            result['top_predictions'] = all_results[:top_k]

            if return_probs:
                result['all_probabilities'] = {item['label']: item['probability'] for item in all_results}

        return result

    def predict_batch(self, texts, max_length=128):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text, max_length)
            results.append(result)
        return results

    def interactive_test(self):
        """交互式测试"""
        print("\n" + "=" * 60)
        print("医疗意图识别交互测试")
        print("输入 'quit' 或 '退出' 结束")
        print("=" * 60)

        while True:
            user_input = input("\n请输入医疗问题: ").strip()

            if user_input.lower() in ['quit', '退出', 'exit']:
                print("测试结束")
                break

            if not user_input:
                print("输入不能为空")
                continue

            try:
                result = self.predict(user_input, return_probs=True, top_k=3)

                print(f"\n问题: {result['text']}")
                print(f"预测意图: {result['predicted_intent']}")
                print(f"置信度: {result['confidence']:.4f}")

                print("Top-3 预测:")
                for i, pred in enumerate(result.get('top_predictions', []), 1):
                    print(f"  {i}. {pred['label']:20s} - {pred['probability']:.4f}")

            except Exception as e:
                print(f"预测错误: {e}")


def quick_test():
    """快速测试"""
    predictor = MedicalIntentPredictor()

    # 测试样本
    test_samples = [
        "孩子牙疼虫牙，带去医院看病了。能不能吃这个药呢？",
        "老年人心肌梗死的症状有哪些特点？",
        "心绞痛可以运动吗？",
        "这段时间一直鼻子不通气，嗓子里有痰，咳不出来",
        "糖尿病应该怎么预防并发症？",
        "高血压患者可以喝酒吗？",
        "感冒了应该挂什么科？",
        "高血压是什么原因引起的？",
        "糖尿病有哪些症状？",
        "拔智齿后一般要疼几天？",
    ]

    print("\n" + "=" * 60)
    print("医疗意图识别快速测试")
    print("=" * 60)

    results = predictor.predict_batch(test_samples)

    for i, result in enumerate(results, 1):
        print(f"\n{i:2d}. 问题: {result['text']}")
        print(f"    预测意图: {result['predicted_intent']}")
        print(f"    置信度: {result['confidence']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗意图识别预测")
    parser.add_argument("--mode", type=str, default="quick",
                        choices=["quick", "interactive"],
                        help="测试模式: quick(快速测试), interactive(交互式)")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_test()
    elif args.mode == "interactive":
        predictor = MedicalIntentPredictor()
        predictor.interactive_test()