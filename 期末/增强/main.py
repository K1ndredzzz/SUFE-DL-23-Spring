import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import nlpaug.augmenter.word as naw

# 超参
BATCH_SIZE = 32
EPOCHS = 120
MAX_LENGTH = 40  # query句子最大长度不超过40
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01  # 权重衰减系数
MODEL_NAME = 'bert-base-chinese' # 预训练模型选用bert-base-chinese

set_seed(42)

# 加载csv文件作为训练集
raw_dataset = load_dataset('csv', data_files='train_dataset.csv', split='train')

# 将label列编码为数字
raw_dataset = raw_dataset.class_encode_column('label')

# 加载bert中文预训练模型的自动分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 定义分词器函数 对query列文本进行编码
def encode(examples):
    return tokenizer(examples['query'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# 应用分词函数编码应用分词函数编码到整个数据集 分batch 去掉#列即序号
raw_dataset = raw_dataset.map(encode, batched=True, remove_columns='#')

# 加载nlpaug包中的ContextualWordEmbsAug上下文数据增强方法 模型仍使用bert-base-chinese
augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-chinese', action='substitute')

# 定义数据增强函数 增强query列
def augment_text(example):
    example['query'] = augmenter.augment(example['query'])
    return example

# 应用数据增强函数
augmented_dataset = raw_dataset.map(augment_text)

# 不划分训练集于验证集
train_dataset = raw_dataset
valid_dataset = raw_dataset

# 设置该三列数据格式为pytorch张量
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 创建dataloader 训练集shuffle
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型
# 使用huggingface中的序列分类器
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=train_dataset.features['label'].num_classes)
device = 'cuda'
model.to(device)
# 优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# lr scheduler
scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))

# 训练
best_acc = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播输入query的张量、掩码、label的张量 计算loss 反向传播更新梯度
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # 更新参数 更新lr 清空梯度
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # loss加总
        total_loss += loss.detach().item()

    # evaluation 计算验证集精度
    model.eval()
    valid_predictions = []
    for batch in valid_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # 禁用梯度
        with torch.no_grad():
            # 由input_ids和attention_mask计算结果
            outputs = model(input_ids, attention_mask=attention_mask)
        # 收集预测结果 收集类别维度最大概率的结果
        valid_predictions.extend(outputs.logits.argmax(-1).cpu().numpy())
    # 计算accuracy
    valid_acc = np.mean(valid_predictions == valid_dataset['label'].numpy())
    print(f"Epoch: {epoch + 1}, Train Loss: {total_loss:.3f}, Validation Accuracy: {valid_acc:.4f}")

    # 保存在验证集上最好的结果
    if valid_acc >= best_acc:
        model.save_pretrained('output')

# 预测结果
# 加载测试集 保证编码与训练时一致
test_dataset = load_dataset('csv', data_files='test_dataset.csv', split='train')
test_id = test_dataset['id']
test_dataset = test_dataset.map(encode, batched=True, remove_columns='id')
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 调用模型训练好的参数
model = AutoModelForSequenceClassification.from_pretrained('output')
model.to(device)

# 评估模式 收集预测结果
model.eval()
predictions = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    # 收集结果
    predictions.extend(outputs.logits.argmax(-1).cpu().numpy())

# 转字符串
predicted_labels = train_dataset.features['label'].int2str(predictions)

# 保存预测结果 以csv格式存储
submission = pd.DataFrame({'id': test_id, 'label': predicted_labels})
submission.to_csv('nlpaug.csv', index=False)