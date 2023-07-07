import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import datetime
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
#from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import math
from tqdm import tqdm
import os
import shutil
from os import listdir
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModel

# Specifying GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))
pan = parent + '/PAN18'

image1 = pd.read_csv ('Swin_train_final_output.csv')
image2 = pd.read_csv ('Swin_validation_final_output.csv')
image3 = pd.read_csv ('Swin_test_final_output.csv')
text1 = pd.read_csv (path+'/electra_train_text_combined.csv')
text2 = pd.read_csv (path+'/electra_test_text_combined.csv')
text3 = pd.read_csv (path+'/electra_test_text_combined.csv')

train = pd.DataFrame (data= {'img0':list(image1['data0']), 'img1':list(image1['data1']), 'txt0':list(text1['data0']), 'txt1':list(text1['data1']), 'label':list(image1['Label'])})
val = pd.DataFrame (data= {'img0':list(image2['data0']), 'img1':list(image2['data1']), 'txt0':list(text2['data0']), 'txt1':list(text2['data1']), 'label':list(image2['Label'])})
test = pd.DataFrame (data= {'img0':list(image3['data0']), 'img1':list(image3['data1']), 'txt0':list(text3['data0']), 'txt1':list(text3['data1']), 'label':list(image3['Label'])})

train_data = torch.tensor (train[['img0','img1','txt0','txt1']].to_numpy())
train_label = torch.tensor(train[['label']].to_numpy())
val_data = torch.tensor (val[['img0','img1','txt0','txt1']].to_numpy())
val_label = torch.tensor(val[['label']].to_numpy())
test_data = torch.tensor (test[['img0','img1','txt0','txt1']].to_numpy())
test_label = torch.tensor(test[['label']].to_numpy())

train_ds = TensorDataset (train_data, train_label)
val_ds = TensorDataset (val_data, val_label)
test_ds = TensorDataset (test_data, test_label)

# Model for Combining the 10 images
class Combination (nn.Module):
  def __init__ (self, hidden, dropout, input_size=4, classes=2):
    super (Combination, self).__init__()
    self.dropout = nn.Dropout (p=dropout)
    self.lin1 = nn.Linear (input_size, hidden)
    self.lin2 = nn.Linear (hidden, classes)
  def forward (self, input):
    output = self.dropout (input)
    output = self.lin1(input)
    output = self.lin2(output)
    return output

def train_epochs (model, dataloader, ce_loss, optimizer, device, scheduler, entry_size):
  model = model.train ()
  losses = []
  correct_predictions_count = 0
  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0
  for data in dataloader:
      input_ids = data[0].to (device)
      targets = data[1].squeeze(1).to (device)
      outputs = model (input_ids.float())
      _, preds = torch.max (outputs, dim=1)
      loss = ce_loss (outputs, targets)
      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())
      F_correct += torch.sum ((preds == 0) & (preds == targets))
      F_incorrect += torch.sum ((preds == 0) & (preds != targets))
      M_correct += torch.sum ((preds == 1) & (preds == targets))
      M_incorrect += torch.sum ((preds == 1) & (preds != targets))
      loss.backward ()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
  return correct_predictions_count.double() / entry_size, np.mean(losses), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), F_correct / (F_correct + M_incorrect), M_correct / (M_correct + F_incorrect)

def eval_model (model, dataloader, ce_loss, device, entry_size, best_acc):
  model = model.eval()
  losses = []
  correct_predictions_count = 0
  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0
  with torch.no_grad():
    for data in dataloader:
      input_ids = data[0].to (device)
      targets = data[1].squeeze(1).to (device)
      outputs = model (input_ids.float())
      _, preds = torch.max (outputs, dim=1)
      loss = ce_loss (outputs, targets)
      l = len (targets[:])
      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())
      F_correct += torch.sum ((preds == 0) & (preds == targets))
      F_incorrect += torch.sum ((preds == 0) & (preds != targets))
      M_correct += torch.sum ((preds == 1) & (preds == targets))
      M_incorrect += torch.sum ((preds == 1) & (preds != targets))
  return correct_predictions_count.double() / entry_size, np.mean(losses), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), F_correct / (F_correct + M_incorrect), M_correct / (M_correct + F_incorrect)

train_loader = DataLoader (train_ds, batch_size=8, num_workers=2, shuffle=True)
val_loader = DataLoader (val_ds, batch_size=8, num_workers=2, shuffle=False)
test_loader = DataLoader (test_ds, batch_size=8, num_workers=2, shuffle=False)

model = Combination (dropout= 0.5, hidden= 5)
model = model.to (device)

epochs = 1000
lr = 0.00001
optimizer = optim.AdamW (model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup (optimizer, num_warmup_steps=0, num_training_steps=total_steps)
ce_loss = nn.CrossEntropyLoss().to (device)
best_accuracy = 0
for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_F_Percision, train_M_Percision, train_F_Recall, train_M_Recall = train_epochs (model, train_loader, ce_loss, optimizer, device, scheduler, len(train_ds) )
  print(f'Train loss {train_loss} accuracy {train_acc} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_F_Percision, val_M_Percision, val_F_Recall, val_M_Recall = eval_model(model, test_loader, ce_loss, device, len(test_ds), best_acc2 )
  print(f'Val loss {val_loss} accuracy {val_acc} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()
  if val_acc > best_accuracy:
    torch.save(imgModel.state_dict(), 'Swin-ELECTRA')
    best_accuracy = val_acc

#loading the model
imgModel = ImageClassifier (swin)
imgModel = imgModel.to (device)
imgModel.load_state_dict(torch.load('Swin-ELECTRA'))

def TheCombinationModel (model, dataloader, device):
  model = model.eval()
  out = []
  targets = []
  for data in dataloader:
      input_ids = data[0].to (device)
      targets.append(data[1].item())
      outputs = model (input_ids.float())
      out.append (outputs.tolist())
  return out, targets

P, A = TheCombinationModel (model, test_loader, device)

CM = confusion_matrix (A, P)
CM = CM / len (P)
CM = pd.DataFrame (CM, index=['Female','Male'], columns=['Female','Male'])

plt.figure(figsize = (3,3))
sns.heatmap(CM, annot=True)
plt.xlabel("Predicted Values", fontsize = 11)
plt.ylabel("True Values", fontsize = 11)
plt.show()

target_names = ['Female', 'Male']
print(classification_report(A, P, target_names=target_names))
