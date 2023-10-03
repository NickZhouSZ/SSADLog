import torch
import json
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split  # Add the train_test_split function
import re
import string
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# 从JSON文件中读取日志数据
json_file_path = 'train_dataset_b.json'
best_model_path =        'train_BGL_with_clean_best_model'
final_model_path =        'train_BGL_with_clean_final_model'

# 定义训练参数
batch_size = 32
num_epochs = 20
Learning_rate = 5e-6       # 學習率

best_val_loss = float('inf')
early_stop_count = 0
early_stop_patience = 3

def save_model(model, path):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(path)

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

# 定义自定义数据集类
class LogDataset(Dataset):
    def __init__(self, log_data, tokenizer):
        self.log_data = log_data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.log_data)
    
    def __getitem__(self, idx):
        log_entry = self.log_data[idx]
        dec_t = ' '.join(log_entry['Dec'])
        inc_t = ' '.join(log_entry['Inc'])
        context = clean(dec_t)
        incident = clean(inc_t)

        label = log_entry['labels']
        
        tokenized_inputs = self.tokenizer.encode_plus(
            text=context,
            text_pair=incident,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,  # Add [CLS], [SEP], and [PAD] tokens
        )
        
        inputs = {
            'input_ids': tokenized_inputs['input_ids'].squeeze(),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor([label]),
        }
   
        return inputs

# 下载预训练的BERT模型和tokenizer
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 将模型移动到CUDA设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with open(json_file_path, 'r') as file:
    log_data = json.load(file)

import random
for epoch in range(num_epochs):
    random.shuffle(log_data)
    # Ensure log_data is a list of dictionaries
    if not isinstance(log_data, list):
        raise ValueError("The JSON data should be in the format of a list of dictionaries.")

    # Split the dataset into train and validation sets
    train_data, test_data = train_test_split(log_data, test_size=0.1, random_state=42)

    # 创建训练数据集
    train_dataset = LogDataset(train_data, tokenizer)
    test_dataset = LogDataset(test_data, tokenizer)


    # Calculate class weights
    num_samples = len(train_dataset)
    num_positive_samples = sum(1 for data in train_dataset if data['labels'] == 1)
    num_negative_samples = num_samples - num_positive_samples

    # Check if num_positive_samples is zero to avoid division by zero
    if num_positive_samples == 0:
        weight_positive = 1.0
    else:
        weight_positive = num_negative_samples / num_positive_samples
    weight_negative = 1.0

    # Scale the class weights to sum up to the number of classes (2)
    sum_weights = weight_negative + weight_positive
    weight_negative /= sum_weights
    weight_positive /= sum_weights
    print(f"weight_positive: {weight_positive}, weight_negative: {weight_negative}")

    # Convert class weights to a tensor on the appropriate device
    class_weights = torch.tensor([weight_negative, weight_positive], device=device)

    # Define loss function with the class_weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Learning_rate, eps=1e-8)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 訓練模式
    model.train()
    AllTrainLoss = 0.0
    count = 0
    val_true_labels = []
    val_pred_labels = []
    progress_bar = tqdm(range(len(train_dataloader)), leave=False)

    for batch_index, batch_dict in enumerate(train_dataloader):
        # Move tensors to the device
        input_ids = batch_dict['input_ids'].to(device)
        attention_mask = batch_dict['attention_mask'].to(device)
        labels = batch_dict['labels'].to(device)
        progress_bar.update(1)

        # Perform forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        logits = outputs.logits
        loss = loss_fn(logits, labels.squeeze())
        #loss = outputs.loss

        AllTrainLoss += loss.item()
        count += 1

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    Average_train_loss = round(AllTrainLoss/count, 3)

    # Validation mode
    model.eval()
    val_correct = 0.0
    total_samples = 0
    val_loss = 0.0
    val_count = 0
    # Lists to store the true labels and predicted labels
    all_labels = []
    all_predictions = []    

    with torch.no_grad():
        for batch_index, batch_dict in enumerate(test_dataloader):
            input_ids = batch_dict['input_ids'].to(device)
            attention_mask = batch_dict['attention_mask'].to(device)
            labels = batch_dict['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = loss_fn(outputs.logits, labels.squeeze())
            #loss = outputs.loss

            _, predicted_labels = torch.max(outputs.logits, 1)

            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

            total_samples += labels.size(0)

            val_loss += loss.item()
            val_count += 1
            
    Average_val_loss = round(val_loss / val_count, 3)

    # Flatten the array of arrays
    flat_all_labels = np.concatenate(all_labels).tolist()

    precision = precision_score(flat_all_labels, all_predictions)
    recall = recall_score(flat_all_labels, all_predictions)
    f1 = f1_score(flat_all_labels, all_predictions)

    print("mismatching:")
    mismatch_indices = [i for i, (label, pred) in enumerate(zip(flat_all_labels, all_predictions)) if label != pred]

    print("total dataset items:", len(log_data))
    print("total mismatch:", len(mismatch_indices))

    print('\n Val Loss: {}, Precision: {}, Recall: {}, F1: {}'.format(
        Average_val_loss, precision, recall, f1))

    # Early stopping
    if Average_val_loss < best_val_loss:
        best_val_loss = Average_val_loss
        early_stop_count = 0
        # Save the best model
        save_model(model, best_model_path)
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_patience:
            print("Early stopping triggered.")
            break

# 模型存檔
save_model(model, final_model_path)