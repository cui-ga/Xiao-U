"""
医疗意图识别模型训练脚本
使用 hfl/chinese-macbert-base 模型
"""
import os
import sys
import logging
import time
from pathlib import Path
import json


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

from config import (
    MODEL_CONFIG, LABEL_MAPPING, TRAIN_DATA_PATH,
    BEST_MODEL_PATH, LOG_FILE, MODEL_SAVE_DIR
)

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MedicalIntentDataset(Dataset):
    """医疗意图识别数据集"""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_and_preprocess_data(data_path):
    """加载和预处理数据"""
    logger.info(f"正在加载数据: {data_path}")

    try:
        df = pd.read_csv(data_path)

        required_cols = ['text', 'label_class', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV文件缺少必要列: {col}")

        def standardize_label(label):
            if not isinstance(label, str):
                return "其他"
            clean = label.strip().replace("（", "(").replace("）", ")").replace(" ", "")
            if clean in LABEL_MAPPING["label2id"]:
                return clean
            if "临床表现" in clean or "病症表现" in clean:
                return "临床表现(病症表现)"
            return clean

        df['label_class_clean'] = df['label_class'].apply(standardize_label)

        df['label_id'] = df['label_class_clean'].apply(
            lambda x: LABEL_MAPPING["label2id"].get(x, LABEL_MAPPING["label2id"]["其他"])
        )

        mismatch_count = (df['label'] != df['label_id']).sum()
        if mismatch_count > 0:
            logger.warning(f"发现 {mismatch_count} 条数据标签不一致，使用标准映射修正")
            df['label'] = df['label_id']

        logger.info("数据加载完成，类别分布:")
        dist = df['label_class_clean'].value_counts()
        for label, count in dist.items():
            percent = count / len(df) * 100
            logger.info(f"  {label:20s}: {count:4d} ({percent:.1f}%)")

        return df['text'].values, df['label'].values

    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise


class IntentClassifierTrainer:
    """意图分类训练器"""

    def __init__(self):
        self.config = MODEL_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=self.config["num_labels"]
        )
        self.model.to(self.device)

        self.history = {
            'epoch': [],
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        self.best_f1 = 0
        self.best_epoch = 0

    def create_dataloaders(self, texts, labels, batch_size=None):
        """创建数据加载器"""
        if batch_size is None:
            batch_size = self.config["batch_size"]
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=self.config["validation_split"],
            random_state=self.config["random_seed"],
            stratify=labels
        )

        train_dataset = MedicalIntentDataset(
            train_texts, train_labels, self.tokenizer, self.config["max_length"]
        )
        val_dataset = MedicalIntentDataset(
            val_texts, val_labels, self.tokenizer, self.config["max_length"]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        logger.info(f"训练集: {len(train_dataset)} 条样本")
        logger.info(f"验证集: {len(val_dataset)} 条样本")

        return train_loader, val_loader

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, val_loader):
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="评估"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        report = classification_report(
            all_labels, all_preds,
            target_names=[LABEL_MAPPING["id2label"][str(i)] for i in range(self.config["num_labels"])],
            digits=4
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'report': report,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )

        total_steps = len(train_loader) * self.config["epochs"]
        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        logger.info("=" * 60)
        logger.info("开始训练医疗意图识别模型")
        logger.info(f"模型: {self.config['model_name']}")
        logger.info(f"总训练步数: {total_steps}")
        logger.info(f"批次大小: {self.config['batch_size']}")
        logger.info(f"学习率: {self.config['learning_rate']}")
        logger.info("=" * 60)

        for epoch in range(self.config["epochs"]):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler, epoch)

            eval_results = self.evaluate(val_loader)

            epoch_time = time.time() - epoch_start

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(eval_results['loss'])
            self.history['val_acc'].append(eval_results['accuracy'])
            self.history['val_f1'].append(eval_results['f1'])

            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']} (耗时: {epoch_time:.1f}s)")
            logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            logger.info(f"验证损失: {eval_results['loss']:.4f}, 验证准确率: {eval_results['accuracy']:.4f}")
            logger.info(f"验证F1分数: {eval_results['f1']:.4f}")
            
            if eval_results['f1'] > self.best_f1:
                self.best_f1 = eval_results['f1']
                self.best_epoch = epoch + 1
                self.save_model()
                logger.info(f"✅ 保存最佳模型 (F1: {self.best_f1:.4f})")

        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info(f"最佳模型在 Epoch {self.best_epoch}, F1分数: {self.best_f1:.4f}")
        logger.info(f"模型保存在: {BEST_MODEL_PATH}")
        logger.info("=" * 60)

        return self.best_f1

    def save_model(self):
        """保存模型和配置"""
        BEST_MODEL_PATH.mkdir(exist_ok=True, parents=True)

        self.model.save_pretrained(BEST_MODEL_PATH)
        self.tokenizer.save_pretrained(BEST_MODEL_PATH)

        full_config = {
            "model_config": self.config,
            "label_mapping": LABEL_MAPPING,
            "train_history": self.history,
            "best_epoch": self.best_epoch,
            "best_f1": self.best_f1
        }

        config_path = BEST_MODEL_PATH / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config, f, ensure_ascii=False, indent=2)

        logger.info(f"模型配置已保存: {config_path}")


def main():
    """主训练流程"""
    logger.info("=" * 60)
    logger.info("医疗意图识别模型训练开始")
    logger.info("=" * 60)

    try:
        texts, labels = load_and_preprocess_data(TRAIN_DATA_PATH)

        trainer = IntentClassifierTrainer()

        train_loader, val_loader = trainer.create_dataloaders(texts, labels)

        best_f1 = trainer.train(train_loader, val_loader)

        logger.info("\n最终模型在验证集上的表现:")
        final_eval = trainer.evaluate(val_loader)
        logger.info(f"准确率: {final_eval['accuracy']:.4f}")
        logger.info(f"F1分数: {final_eval['f1']:.4f}")

        logger.info("\n各类别性能 (从验证集分类报告):")
        logger.info(final_eval['report'])

        logger.info("=" * 60)
        logger.info("训练流程完成！模型已准备好用于医疗智能问答系统。")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
