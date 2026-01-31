项目简介：基于Chatbot Arena 数据集，通过构建深度学习模型预测人类对两个不同 LLM 生成回答的偏好。

数据预处理：首先标签化处理训练集的结果，将相关数据强制类型转换确保数据的一致性，使用 Hugging Face Tokenizer 对多轮对话文本进行分词处理，采用 Prompt + Response A + Response B 的三元组拼接策略，利用特殊分隔符 [SEP] 构建上下文感知序列，使模型能够理解对话的逻辑连贯性。
模型训练：通过采用 DeBERTa-v3 预训练架构，并针对 Windows 显存受限环境实施了梯度累积（Gradient Accumulation）与 混合精度训练（FP16）策略，在保证数值稳定性的同时大幅提升了算力利用率。

index.py 为训练过程
test.py 为验证过程
