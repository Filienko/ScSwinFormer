import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from sklearn.metrics import precision_recall_fscore_support
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Read data
data_train = pd.read_csv('data/AML/train.csv', index_col=0)
data_test = pd.read_csv('data/AML/test.csv', index_col=0)

# Transpose data
data_train = data_train.T
data_test = data_test.T

# Read labels
labels_train = pd.read_csv('data/AML/trainlabel.csv', header=None)
labels_test = pd.read_csv('data/AML/testlabel.csv', header=None)

# Drop the first column and process labels
labels_train_df = labels_train.drop(labels_train.columns[0], axis=1)
labels_test_df = labels_test.drop(labels_test.columns[0], axis=1)

# Update deprecated applymap with map
labels_train_df = labels_train_df.T.apply(lambda x: x - 1)
labels_test_df = labels_test_df.T.apply(lambda x: x - 1)

# Reset indices to ensure alignment
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)
labels_train_df = labels_train_df.reset_index(drop=True)
labels_test_df = labels_test_df.reset_index(drop=True)

# Verify dimensions before creating AnnData
print("Data shapes before AnnData creation:")
print(f"Training data: {data_train.shape}")
print(f"Testing data: {data_test.shape}")
print(f"Training labels: {labels_train_df.shape}")
print(f"Testing labels: {labels_test_df.shape}")

# Ensure dimensions match before concatenation
combined_data = pd.concat([data_train, data_test], ignore_index=True)
combined_labels = pd.concat([labels_train_df, labels_test_df], ignore_index=True)
missing_labels = set(combined_data.index) - set(combined_labels.index)
print("Missing labels for indices:", missing_labels)

print("\nCombined shapes:")
print(f"Combined data: {combined_data.shape}")
print(f"Combined labels: {combined_labels.shape}")

# Create AnnData object with verified dimensions
adata = sc.AnnData(combined_data, obs=combined_labels)

# Continue with preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes
n_top_genes = 2000
sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
highly_variable_genes = adata.var.highly_variable
adata = adata[:, highly_variable_genes]
print("adata.shape after processed", adata.shape)
# Split back into train and test
n_train = data_train.shape[0]
adata_train = adata[:n_train]
adata_test = adata[n_train:]

# Dataset class remains the same
class AnnDataset(Dataset):
    def __init__(self, adata):
        self.data = torch.tensor(adata.X, dtype=torch.float32)
        self.labels = torch.tensor(adata.obs[0].values, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create datasets
train_dataset = AnnDataset(adata_train)
test_dataset = AnnDataset(adata_test)

# Create dataloaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sliding_window_embedding(sequence, window_size):
 # sequence (batch_size, seq_length)
 batch_size, seq_length = sequence.shape

 new_seq_length = seq_length - window_size + 1

 embedded_sequence = torch.zeros((batch_size, new_seq_length, window_size), device=sequence.device)

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
output_size = len(np.unique(labels_test_df)) + 5 # Add 1 for the cls_token
model = TransformerClassifier(input_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)



def train(model, device, train_loader, criterion, optimizer):
 model.train()
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

num_epochs = 50  # SGD 20，SGD3 03（2000，1024），Adam 130(2000,1024)

for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}:')
    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader, criterion)

# Convert data to NumPy arrays
X_train = adata_train.X
X_test = adata_test.X
y_train = labels_train_df.values.ravel()
y_test = labels_test_df.values.ravel()

# LightGBM Model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_train)),
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 31,
    'verbose': -1
}

lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_test])

lgb_preds = np.argmax(lgb_model.predict(X_test), axis=1)
lgb_acc = accuracy_score(y_test, lgb_preds)
lgb_precision, lgb_recall, lgb_f1, _ = precision_recall_fscore_support(y_test, lgb_preds, average='macro')

print(f"LightGBM Accuracy: {lgb_acc * 100:.2f}%, Macro Precision: {lgb_precision * 100:.2f}%, "
      f"Macro Recall: {lgb_recall * 100:.2f}%, Macro F1: {lgb_f1 * 100:.2f}%")

# RandomForest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='macro')

print(f"RandomForest Accuracy: {rf_acc * 100:.2f}%, Macro Precision: {rf_precision * 100:.2f}%, "
      f"Macro Recall: {rf_recall * 100:.2f}%, Macro F1: {rf_f1 * 100:.2f}%")

# Extract embeddings from the trained Transformer model
def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            emb = model.forward(data)  # CLS token output before final classification
            embeddings.append(emb.cpu().numpy())
            labels.append(target.cpu().numpy())

    return np.vstack(embeddings), np.concatenate(labels)


# Get embeddings from train and test sets
X_train_emb, y_train = get_embeddings(model, train_loader, device)
X_test_emb, y_test = get_embeddings(model, test_loader, device)

# LightGBM Model Training
lgb_train = lgb.Dataset(X_train_emb, y_train)
lgb_test = lgb.Dataset(X_test_emb, y_test, reference=lgb_train)

params = {
    'objective': 'multiclass',
    'num_class': output_size,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Train LightGBM
lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_test], num_boost_round=100)

# Make Predictions
y_pred = np.argmax(lgbm_model.predict(X_test_emb), axis=1)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

print(f'LightGBM on Transformer Embeddings - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
