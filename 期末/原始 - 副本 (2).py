# 引入所需的库和模块
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# 定义超参数
BATCH_SIZE = 32  # 批处理大小
EPOCHS = 70  # 迭代次数
MAX_LENGTH = 40  # 序列最大长度
LEARNING_RATE = 5e-5  # 学习率
WEIGHT_DECAY = 0.01  # 权重衰减系数
NUM_WARMUP_STEPS = 1000  # 热身步骤数
MODEL_NAME = 'bert-base-chinese'  # 预训练模型名
TRAIN_SIZE = 0.9  # 训练集比例，剩余为验证集

# 设置随机种子，保证结果可复现
set_seed(42)

# 加载csv数据集，数据路径为'train_dataset.csv'
raw_dataset = load_dataset('csv', data_files='train_dataset.csv', split='train')

# 对标签进行编码，将类别标签转化为数字
raw_dataset = raw_dataset.class_encode_column('label')

# 从预训练模型名加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 定义编码函数，对输入的文本进行编码
def encode(examples):
    return tokenizer(examples['query'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# 对数据集进行编码
raw_dataset = raw_dataset.map(encode, batched=True, remove_columns='#')


# # 不划分训练集于验证集（需要注释下面几行代码）
train_dataset = raw_dataset
valid_dataset = raw_dataset

# 设置数据格式，用于训练模型
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 准备模型，优化器和学习率调度器
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=train_dataset.features['label'].num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))

# 进行模型训练
best_acc = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 模型前向传播，计算loss
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # 反向传播，更新梯度
        loss.backward()

        # 优化器步进，更新模型参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 清空梯度
        optimizer.zero_grad()

        total_loss += loss.detach().item()

    # 验证模型，计算验证集上的精度
    model.eval()
    valid_predictions = []
    for batch in valid_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        valid_predictions.extend(outputs.logits.argmax(-1).cpu().numpy())

    valid_acc = np.mean(valid_predictions == valid_dataset['label'].numpy())
    print(f"Epoch: {epoch + 1}, Train Loss: {total_loss:.3f}, Validation Accuracy: {valid_acc:.4f}")

    # 保存在验证集上最好的结果
    if valid_acc >= best_acc:
        model.save_pretrained('output')


# 加载测试数据集，对测试集进行预测
test_dataset = load_dataset('csv', data_files='test_dataset.csv', split='train')
test_id = test_dataset['id']
test_dataset = test_dataset.map(encode, batched=True, remove_columns='id')
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained('output')
model.to(device)

model.eval()
predictions = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions.extend(outputs.logits.argmax(-1).cpu().numpy())

# 将预测的标签从数字转换回原始的类别
predicted_labels = train_dataset.features['label'].int2str(predictions)

# 保存预测结果，以csv格式存储
submission = pd.DataFrame({'id': test_id, 'label': predicted_labels})
submission.to_csv('submit_sample.csv', index=False)