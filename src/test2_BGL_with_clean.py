import torch
import json
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split  # Add the train_test_split function
import re
import string
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# 从JSON文件中读取日志数据

test_file_path = 'test2_dataset_b_60.json'
config_path =       're_train_BGL_with_clean_best_model/config.json'
model_path =        're_train_BGL_with_clean_best_model/pytorch_model.bin'


# 定义训练参数
batch_size = 32
num_epochs = 8
Learning_rate = 5e-6       # 學習率

best_val_loss = float('inf')
early_stop_count = 0
early_stop_patience = 3

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

# load and init
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(config_path)
model = BertForSequenceClassification.from_pretrained(model_path, config=config)
print("Model loaded\n")

# 将模型移动到CUDA设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with open(test_file_path, 'r') as file:
    log_data = json.load(file)

# Ensure log_data is a list of dictionaries
if not isinstance(log_data, list):
    raise ValueError("The JSON data should be in the format of a list of dictionaries.")

# create test dataset
test_dataset = LogDataset(log_data, tokenizer)

# 创建数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("dataloader finished\n")
# test mode
model.eval()
total_samples = 0
val_loss = 0.0
val_count = 0

# Lists to store the true labels and predicted labels
all_labels = []
all_predictions = []

progress_bar = tqdm(range(len(test_dataloader)), leave=False)

with torch.no_grad():
    for batch_index, batch_dict in enumerate(test_dataloader):

        input_ids = batch_dict['input_ids'].to(device)
        attention_mask = batch_dict['attention_mask'].to(device)
        labels = batch_dict['labels'].to(device)
        progress_bar.update(1)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

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

print(flat_all_labels)
print("\n=======================================\n")
print(all_predictions)
print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")

print("mismatching:")
mismatch_indices = [i for i, (label, pred) in enumerate(zip(flat_all_labels, all_predictions)) if label != pred]

print("total items:", len(log_data))
print("total mismatch:", len(mismatch_indices))
for index in mismatch_indices:
    print("Index:", index)
    print("Label:", flat_all_labels[index])
    print("Prediction:", all_predictions[index])
    print("JSON Content:", log_data[index])  # Replace this with the actual variable containing JSON data
    print("-----")

print('\n Val Loss: {}, Precision: {}, Recall: {}, F1: {}'.format(
    Average_val_loss, precision, recall, f1))
