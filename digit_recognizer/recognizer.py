import pandas as pd
from torch.utils import data
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import Module, Linear
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# загрузка и нормализация данных

train_data_pd, val_data_pd = train_test_split(pd.read_csv("digit_train.csv"), test_size=0.2)
test_data_pd = pd.read_csv("digit_test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

x_train_np = train_data_pd.drop("label", axis=1).values.astype('float32')
y_train_np = train_data_pd["label"].values.astype('int64')

x_val_np = val_data_pd.drop("label", axis=1).values.astype('float32')
y_val_np = val_data_pd["label"].values.astype('int64')

x_train_t = torch.tensor(x_train_np) / 255.0
y_train_t = torch.tensor(y_train_np)

x_val_t = torch.tensor(x_val_np) / 255.0
y_val_t = torch.tensor(y_val_np)

x_test_t = torch.tensor(test_data_pd.values, dtype=torch.float32) / 255.0

train_dataset = TensorDataset(x_train_t, y_train_t)
val_dataset = TensorDataset(x_val_t, y_val_t)

batch_size = 64
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# класс н.с.

class DigitRecognizer(Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
    super().__init__()
    self.layer1 = Linear(input_dim, hidden_dim1)
    self.layer2 = Linear(hidden_dim1, hidden_dim2)
    self.layer3 = Linear(hidden_dim2, output_dim)

  def forward(self, x):
    x = nn.functional.relu(self.layer1(x))
    x = nn.functional.relu(self.layer2(x))
    x = self.layer3(x)

    return x

model_dr = DigitRecognizer(28 * 28, 128, 32, 10)

optimizer = optim.Adam(params=model_dr.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
epochs = 3

loss_lst_val = []
loss_val = []

for e in range(epochs):
  model_dr.train()

  loss_mean = 0
  lm_count = 0

  train_tqdm = tqdm(train_data, leave=False)
  for x_train, y_train in train_tqdm:
      predict = model_dr(x_train)
      loss = loss_func(predict, y_train)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      lm_count += 1
      loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
      train_tqdm.set_description(f"epoch {e + 1}/{epochs}, loss_mean={loss_mean:.3f}")

  model_dr.eval()
  Q_val = 0
  count_val = 0

  for x_val, y_val in val_data:
      with torch.no_grad():
          p = model_dr(x_val)
          loss = loss_func(p, y_val)
          Q_val += loss.item()
          count_val += 1

  Q_val /= count_val

  loss_lst_val.append(Q_val)
  loss_val.append(loss_mean)

  print(f"loss_mean: {loss_mean:.3f}, Q_val: {Q_val:.3f}")


model_dr.eval()

with torch.no_grad():
  test_output = model_dr(x_test_t)
  test_predictions = torch.argmax(test_output, dim=1)

prediction = pd.DataFrame({
    "ImageId": sample_submission["ImageId"].values,
    "Label": test_predictions
})


print(test_predictions[:10])
prediction.to_csv("prediction.csv", index=False)


