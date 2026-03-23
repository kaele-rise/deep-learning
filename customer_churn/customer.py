import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(pd.read_csv('train.csv'), test_size=0.2)
test_data = pd.read_csv('test.csv')

x_train = train_data.drop(['id', 'Churn'], axis=1)
y_train = train_data['Churn'].replace({'Yes': 1, 'No': 0})
y_train = y_train.astype('int')

x_val = val_data.drop(['id', 'Churn'], axis=1)
y_val = val_data['Churn'].replace({'Yes': 1, 'No': 0})
y_val = y_val.astype('int')

x_test = test_data.drop(['id'], axis=1)


num_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
cat_columns = ['gender', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

preprocessing_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)
])

x_train = preprocessing_pipeline.fit_transform(x_train)
x_val = preprocessing_pipeline.transform(x_val)
x_test = preprocessing_pipeline.transform(x_test)


import torch
import torch.nn as nn
from torch.nn import Module
from random import randint
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

class CustomerModel(Module):
  def __init__(self, input_dim, hidden_dim1, output_dim):
    super().__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim1)
    self.layer2 = nn.Linear(hidden_dim1, output_dim)
    self.bn1 = nn.BatchNorm1d(hidden_dim1)

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = self.bn1(x)
    x = self.layer2(x)

    return x

input_dim = x_train.shape[1]
model = CustomerModel(input_dim, 16, 1)


# параметры обучения
batch_size = 200
loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(params=model.parameters(), lr=0.01, weight_decay=0.001)
epochs = 4

# data loader
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
train_dataset = data.TensorDataset(x_train_t, y_train_t)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

x_val_t = torch.tensor(x_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
val_dataset = data.TensorDataset(x_val_t, y_val_t)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# обучение модели

loss_val = []
loss_tr = []
for e in range(epochs):
  model.train()
  lm_count = 0
  loss_mean = 0

  train_tqdm = tqdm(train_loader, desc=f'Epoch {e+1}/{epochs}', leave=False)
  for X_batch, y_batch in train_tqdm:
    y_pred = model(X_batch)
    loss = loss_func(y_pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    lm_count += 1
    loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
    train_tqdm.set_description(f"loss_mean={loss_mean:.3f}")

  model.eval()
  val_loss_sum = 0.0
  with torch.no_grad():
    for x_batch, y_batch in val_loader:
      y_val_pred = model(x_batch)
      loss = loss_func(y_val_pred, y_batch)
      val_loss_sum += loss.item() * X_batch.size(0)

  val_loss_avg = val_loss_sum / len(val_dataset)

  loss_val.append(val_loss_avg)
  loss_tr.append(loss_mean)

  print(f"epoch: {e}, loss_mean: {loss_mean:.3f}, val_loss: {val_loss_avg:.3f}")

  # сохранение данных обученой модели в model_dnn.tar
  st = model.state_dict()
  torch.save(st, "model_dnn.tar")

  # загрузка данных обученой модели
  state_dict = torch.load("model_dnn.tar", weights_only=True)
  model.load_state_dict(state_dict)

  model.eval()

  X_test = torch.tensor(x_test, dtype=torch.float32)
  test_ids = test_data['id']

  with torch.no_grad():
      logits = model(X_test)
      prediction = torch.sigmoid(logits)

  classes = (prediction >= 0.5).int()
  prediction_np = classes.numpy().flatten()
  y_test = pd.DataFrame({'id': test_ids,
                         'Churn': prediction_np})

  y_test.to_csv('submission.csv', index=False)