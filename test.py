import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 基础配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_path = "./final_chatbot_model"  # 你刚刚保存模型的路径

def preprocess_test_function(examples, tokenizer):
    # 与训练时保持完全一致的拼接逻辑
    text_pairs = [
        str(a) + " [SEP] " + str(b) 
        for a, b in zip(examples['response_a'], examples['response_b'])
    ]
    
    return tokenizer(
        examples['prompt'],
        text_pair=text_pairs,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

if __name__ == '__main__':
    # 2. 加载测试数据
    print("正在加载测试集...")
    test_path = '你的文件目录/test.csv'
    test_df = pd.read_csv(test_path)
    
    # 记录 ID 用于最后提交
    test_ids = test_df['id'].astype(str).tolist()

    # 3. 加载模型和分词器
    print("正在加载微调后的模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 4. 转换 Dataset
    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test = test_dataset.map(
        lambda x: preprocess_test_function(x, tokenizer), 
        batched=True
    )

    # 5. 使用 Trainer 进行预测（这样处理 Batching 和 GPU 搬运最快）
    # 我们不需要训练，所以参数可以很简单
    predict_args = TrainingArguments(
        output_dir="./temp_preds",
        per_device_eval_batch_size=8, 
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=0
    )

    trainer = Trainer(model=model, args=predict_args)

    print("正在进行预测（Inference）...")
    raw_preds = trainer.predict(tokenized_test)

    # 6. 将预测结果（Logits）转化为概率（Softmax）
    # 模型输出的是三列数字，我们需要把它们变成加起来等于 1 的概率
    logits = torch.from_numpy(raw_preds.predictions)
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()

    # 7. 生成提交文件
    # 假设：0 -> model_a, 1 -> model_b, 2 -> tie
    submission = pd.DataFrame({
        'id': test_ids,
        'winner_model_a': probs[:, 0],
        'winner_model_b': probs[:, 1],
        'winner_tie': probs[:, 2]
    })

    submission.to_csv('submission.csv', index=False)
    print("🎉 预测完成！提交文件 'submission.csv' 已生成。")
