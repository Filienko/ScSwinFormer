import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scanpy as sc
from sklearn.metrics import precision_recall_fscore_support


data_train = pd.read_csv('data/Adipose/human_Adipose1372_train.csv', index_col=0)
data_test = pd.read_csv('data/Adipose/human_Adipose1372_test.csv', index_col=0)

data_train = data_train.T
data_test = data_test.T

labels_train = pd.read_csv('data/Adipose/trainlabel.csv',header=None)
labels_test = pd.read_csv('data/Adipose/testlabel.csv',header=None)


labels_train_df = labels_train.drop(labels_train.columns[0], axis=1)
labels_test_df = labels_test.drop(labels_test.columns[0], axis=1)


labels_train_df = labels_train_df.applymap(lambda x: x - 1).T
labels_test_df = labels_test_df.applymap(lambda x: x - 1).T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 重置DataFrame索引，确保它们与X的行顺序匹配
labels_train_df = labels_train_df.reset_index(drop=True)
labels_test_df = labels_test_df.reset_index(drop=True)

# 并确保data_train和data_test没有设置特殊的行索引，如果是DataFrame的话也重置它们
if isinstance(data_train, pd.DataFrame):
    data_train = data_train.reset_index(drop=True)
if isinstance(data_test, pd.DataFrame):
    data_test = data_test.reset_index(drop=True)

adata = sc.AnnData(pd.concat([data_train, data_test], ignore_index=True),
                   obs=pd.concat([labels_train_df, labels_test_df], ignore_index=True))


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)


n_top_genes = 2000
sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
highly_variable_genes = adata.var.highly_variable
adata = adata[:, highly_variable_genes]

n_train = data_train.shape[0]
adata_train = adata[:n_train]
adata_test = adata[n_train:]


class AnnDataset(Dataset):
  def __init__(self, adata):
  self.data = torch.tensor(adata.X, dtype=torch.float32)
  self.labels = torch.tensor(adata.obs[0].values, dtype=torch.long)

 def __len__(self):
  return self.data.shape[0]

 def __getitem__(self, idx):
  return self.data[idx], self.labels[idx]

train_dataset = AnnDataset(adata_train)
test_dataset = AnnDataset(adata_test)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


def sliding_window_embedding(sequence, window_size):
 # sequence 形状是 (batch_size, seq_length)
 batch_size, seq_length = sequence.shape

 # 新的序列长度
 new_seq_length = seq_length - window_size + 1

 # 初始化一个零张量来保存嵌入序列
 embedded_sequence = torch.zeros((batch_size, new_seq_length, window_size), device=sequence.device)

 # 对于序列中的每个位置，取出滑动窗口的向量
 for i in range(new_seq_length):
  window = sequence[:, i:i + window_size]
  embedded_sequence[:, i, :] = window

 return embedded_sequence


class TransformerClassifier(nn.Module):
 def __init__(self, input_size, output_size, window_size=1999, d_model=256, nhead=64,
              dim_feedforward=1024, num_layers=1, dropout=0.2):
  super(TransformerClassifier, self).__init__()

  self.window_size = window_size
  self.embedding = nn.Linear(window_size, d_model)

  # Correctly calculate sequence length after sliding window embedding
  new_seq_length = input_size - window_size + 1

  # Initialize cls_token and pos_embedding
  self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
  self.pos_embedding = nn.Parameter(torch.randn(1, new_seq_length + 1, d_model))  # Plus 1 for cls_token

  self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
  self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

  self.layer_norm = nn.LayerNorm(d_model)
  self.fc = nn.Linear(d_model, output_size)

 def forward(self, x):
  # Create local sliding window tokens
  windows = x.unfold(1, self.window_size, 1).contiguous()
  windows = windows.view(x.size(0), windows.size(1), -1)

  # Linear embedding
  x = self.embedding(windows)

  # Add cls_token to the sequence
  cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
  x = torch.cat((cls_tokens, x), dim=1)

  # Add position embeddings
  x += self.pos_embedding

  # Apply layer normalization
  x = self.layer_norm(x)

  # Encoder
  x = self.transformer_encoder(x.permute(1, 0, 2))  # Transformer requires [seq_len, batch, features]

  # Take the output of cls_token for classification
  x = x[0]
  x = self.fc(x)

  return x

input_size = n_top_genes
output_size = len(np.unique(labels_test_df))
model = TransformerClassifier(input_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)


# 训练函数
def train(model, device, train_loader, criterion, optimizer):
 model.train()  # 将模型设置为训练模式
 running_loss = 0.0
 correct = 0
 total = 0

 for batch_idx, (data, target) in enumerate(train_loader):

  data, target = data.to(device), target.to(device)

  optimizer.zero_grad()

  output = model(data)

  loss = criterion(output, target)

  loss.backward()

  optimizer.step()

  # for accuracy
  running_loss += loss.item() * data.size(0)
  _, predicted = torch.max(output.data, 1)
  total += target.size(0)
  correct += (predicted == target).sum().item()

 train_loss = running_loss / len(train_loader.dataset)
 accuracy = 100. * correct / total
 print(f'Train Loss: {train_loss:.6f}, Train Accuracy: {accuracy:.2f}%')


def test(model, device, test_loader, criterion):
 model.eval()
 test_loss = 0
 correct = 0
 all_targets = []
 all_predicted = []

 with torch.no_grad():
  for data, target in test_loader:
   data, target = data.to(device), target.to(device)
   output = model(data)
   test_loss += criterion(output, target).item()
   _, predicted = torch.max(output.data, 1)
   correct += (predicted == target).sum().item()
   all_targets.extend(target.view_as(predicted).cpu().numpy())
   all_predicted.extend(predicted.cpu().numpy())

 test_loss /= len(test_loader.dataset)
 accuracy = 100. * correct / len(test_loader.dataset)

 # Calculate Macro Precision, Recall, and F1 Score and convert to percentages
 precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predicted, average='macro')
 precision *= 100
 recall *= 100
 f1 *= 100

 print(f'Test Loss: {test_loss:.6f}, '
       f'Test Accuracy: {accuracy:.2f}%, '
       f'Macro Precision: {precision:.2f}%, '
       f'Macro Recall: {recall:.2f}%, '
       f'Macro F1: {f1:.2f}%')

num_epochs = 100  # SGD 20，SGD3 03（2000，1024），Adam 130(2000,1024)

# 
for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}:')
    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader, criterion)