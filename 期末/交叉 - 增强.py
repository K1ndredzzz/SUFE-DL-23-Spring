import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import nlpaug.augmenter.word as naw

# 定义超参数
BATCH_SIZE = 32  # 批处理大小
EPOCHS = 70  # 迭代次数
MAX_LENGTH = 40  # 序列最大长度
LEARNING_RATE = 5e-5  # 学习率
WEIGHT_DECAY = 0.01  # 权重衰减系数
MODEL_NAME = 'bert-base-chinese'  # 预训练模型名
N_FOLDS = 7  # 交叉验证折数
RANDOM_STATE = 42  # 随机种子

# 设置随机种子，保证结果可复现
set_seed(RANDOM_STATE)

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

# 数据增强
augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-chinese', action='substitute')

# 编写数据增强函数
def augment_text(example):
    example['query'] = augmenter.augment(example['query'])
    return example

# 对数据集进行增强
augmented_dataset = raw_dataset.map(augment_text)

# 创建交叉验证对象
cross_validator = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# 进行交叉验证
for fold, (train_indices, valid_indices) in enumerate(cross_validator.split(raw_dataset)):
    # 划分训练集和验证集
    train_subset = raw_dataset.select(train_indices)
    valid_subset = raw_dataset.select(valid_indices)

    # 设置数据格式，用于训练模型
    train_subset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    valid_subset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # 创建数据加载器
    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 准备模型，优化器和学习率调度器
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=train_subset.features['label'].num_classes)
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

        valid_acc = np.mean(valid_predictions == valid_subset['label'].numpy())
        print(f"Fold: {fold+1}, Epoch: {epoch+1}, Train Loss: {total_loss:.3f}, Validation Accuracy: {valid_acc:.4f}")

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
    predicted_labels = train_subset.features['label'].int2str(predictions)

    # 保存预测结果，以csv格式存储
    submission = pd.DataFrame({'id': test_id, 'label': predicted_labels})
    submission.to_csv(f'submit_sample_fold_{fold+1}.csv', index=False)
