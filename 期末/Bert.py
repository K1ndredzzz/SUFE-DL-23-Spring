import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# 超参数
BATCH_SIZE = 32
EPOCHS = 50
MAX_LENGTH = 35  # 最大长度 query句子长度大多不超过35
LR = 5e-5
TRAIN_SIZE = 0.9  # 训练集比例



# 加载csv文件作为训练集
raw_dataset = load_dataset('csv', data_files='train_dataset.csv', split='train')
# 将label列编码为数组
raw_dataset = raw_dataset.class_encode_column('label')
print(raw_dataset['label'])
# 加载bert中文预训练模型的自动分词器
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
# 定义分词器函数 对query列文本进行编码
def tokenize_function(examples): # 对query列 超过max_length截断 少于的补零
    return tokenizer(examples['query'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# 应用分词函数编码应用分词函数编码到整个数据集 分batch 去掉#列即序号
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['#'])
print(tokenized_dataset)
print(tokenized_dataset['query'])
print(tokenized_dataset['input_ids'])
print(tokenized_dataset['token_type_ids'])
print(tokenized_dataset['attention_mask'])

# 训练集和验证集划分 以观测训练效果
tokenized_dataset = tokenized_dataset.train_test_split(train_size=TRAIN_SIZE, seed=42)
train_dataset = tokenized_dataset['train']
valid_dataset = tokenized_dataset['test']

# 设置该三列数据格式为pytorch张量
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 创建dataloader 训练集shuffle
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)



# 模型
# huggingface中调用序列分类器模型的类 根据训练集label的数量设置输出维度
model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=train_dataset.features['label'].num_classes)
model.to('cuda')
# 优化器
optimizer = AdamW(model.parameters(), lr=LR)
# lr scheduler 单个周期内的学习率策略 自动调整LR 每轮迭代的步数
scheduler = OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))



# 训练
best_acc = 0
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['label'].to('cuda')

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
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        # 禁用梯度
        with torch.no_grad():
            # 由input_ids和attention_mask计算结果
            outputs = model(input_ids, attention_mask=attention_mask)
        #收集预测结果 用cpu收集类别维度最大概率的数组
        valid_predictions.extend(outputs.logits.argmax(-1).cpu().numpy())
    
    valid_acc = np.mean(valid_predictions == valid_dataset['label'].numpy())
    print(f"Epoch: {epoch+1}, Train Loss: {total_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")
    
    # 保存在验证集上最好的结果
    if valid_acc >= best_acc:
        best_acc = valid_acc
        model.save_pretrained('output')



# 预测结果
# 加载测试集 保证编码与训练时一致
test_dataset = load_dataset('csv', data_files='test_dataset.csv', split='train')
test_id = test_dataset['id']
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns='id')
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 调用模型训练好的参数
model = AutoModelForSequenceClassification.from_pretrained('output')
model.to('cuda')


# 评估模式 收集预测结果
model.eval()
predictions = []
for batch in test_loader:
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    # 禁用梯度
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions.extend(outputs.logits.argmax(-1).cpu().numpy())

# 数组转字符串
predicted_labels = train_dataset.features['label'].int2str(predictions)

# 下载结果csv文件
submission = pd.DataFrame({'id': test_id, 'label': predicted_labels})
submission.to_csv('submit_sample.csv', index=False)