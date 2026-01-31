import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    set_seed
)

# 1. 基础配置
set_seed(42) # 固定随机种子，让结果可复现
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 使用国内镜像

# 2. 预处理函数 (定义在 main 外面)
def create_labels(row):
    if row['winner_model_a'] == 1: return 0
    if row['winner_model_b'] == 1: return 1
    return 2

def preprocess_function(examples, tokenizer):
    # 修复之前的拼接错误：使用列表推导式处理批量数据
    text_pairs = [
        str(a) + " [SEP] " + str(b) 
        for a, b in zip(examples['response_a'], examples['response_b'])
    ]
    
    inputs = tokenizer(
        examples['prompt'],
        text_pair=text_pairs,
        truncation=True,
        max_length=512,  # 显存如果够大可以设为 1024
        padding="max_length",
    )
    inputs['labels'] = examples['label']
    return inputs

# 3. 执行主体
if __name__ == '__main__':
    # --- A. 数据加载与清洗 ---
    print("正在加载数据...")
    train_path = '你的文件目录/train.csv'
    train_df = pd.read_csv(train_path, low_memory=False)

    # 仅保留需要的列，彻底剔除 Unnamed 等幽灵列
    useful_cols = ['id', 'prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
    train_df = train_df[useful_cols].copy()

    # 强制类型转换，防止 PyArrow 转换失败
    train_df['id'] = train_df['id'].astype(str)
    for col in ['winner_model_a', 'winner_model_b', 'winner_tie']:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(int)
    for col in ['prompt', 'response_a', 'response_b']:
        train_df[col] = train_df[col].astype(str)

    # 创建标签列
    train_df['label'] = train_df.apply(create_labels, axis=1).astype(int)

    # --- B. 准备 Dataset ---
    raw_dataset = Dataset.from_pandas(train_df)
    split_dataset = raw_dataset.train_test_split(test_size=0.2)

    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("正在分词处理（Map）...")
    tokenized_datasets = split_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=useful_cols # 清理掉原始文本列，节省显存
    )

    # --- C. 模型与训练配置 ---
    print("初始化模型...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir="./chatbot_arena_results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,   # 显存小就设为 4，大就设为 8 或 16
        gradient_accumulation_steps=4,  # 累计更新，相当于有效 batch_size = 4*4=16
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        fp16=True,                      # 开启半精度，显卡必备
        dataloader_num_workers=0,       # 修复 Windows 多进程报错的关键：设为 0
        load_best_model_at_end=True,
        logging_steps=50
    )

    # 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    # --- D. 开始训练 ---
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("开始点火训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model("./final_chatbot_model")
    print("训练完成！模型已保存。")
