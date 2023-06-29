import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import LevitFeatureExtractor, LevitForImageClassification
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
from sklearn.metrics import classification_report
from collections import defaultdict
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))

ds = ImageFolder (parent + '/train')
test = ImageFolder (parent + '/test')
feature_extractor = LevitFeatureExtractor.from_pretrained('facebook/levit-128S')
levit = LevitForImageClassification.from_pretrained('facebook/levit-128S', num_labels=len(label2id), label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)

class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings
train_loader = DataLoader(train, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=True)
val_loader = DataLoader(validation, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)
test_loader = DataLoader(test, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)

class ImageClassifier (nn.Module):
  def __init__ (self, levit, classes=2):
    super (ImageClassifier, self).__init__()
    self.levit = levit
  def forward (self, input_ids):
    return self.levit (input_ids)

imgModel = ImageClassifier (levit)
imgModel = imgModel.to (device)

epochs = 10
optimizer = AdamW (imgModel.parameters(), lr=2e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup (optimizer, num_warmup_steps=0, num_training_steps=total_steps)
ce_loss = nn.CrossEntropyLoss().to (device)

def train_epochs (model, dataloader, ce_loss, optimizer, device, scheduler, entry_size):
  model = model.train ()
  losses = []
  correct_predictions_count = 0
  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0
  for data in dataloader:
    feature = data['pixel_values'].to (device)
    targets = data['labels'].to (device)
    outputs = model (feature)
    _, preds = torch.max (outputs.logits, dim=1)
    loss = ce_loss (outputs.logits, targets)
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

def eval_model (model, dataloader, ce_loss, device, entry_size):
  model = model.eval()
  losses = []
  correct_predictions_count = 0
  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0
  with torch.no_grad():
    for data in dataloader:
      feature = data['pixel_values'].to (device)
      targets = data['labels'].to (device)
      outputs = model (feature)
      _, preds = torch.max (outputs.logits, dim=1)
      loss = ce_loss (outputs.logits, targets)
      l = len (targets[:])
      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())
      F_correct += torch.sum ((preds == 0) & (preds == targets))
      F_incorrect += torch.sum ((preds == 0) & (preds != targets))
      M_correct += torch.sum ((preds == 1) & (preds == targets))
      M_incorrect += torch.sum ((preds == 1) & (preds != targets))
  return correct_predictions_count.double() / entry_size, np.mean(losses), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), F_correct / (F_correct + M_incorrect), M_correct / (M_correct + F_incorrect)

torch.cuda.empty_cache()
# For saving the history
history = defaultdict(list)
best_accuracy = 0
for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_F_Percision, train_M_Percision, train_F_Recall, train_M_Recall = train_epochs (imgModel, train_loader, ce_loss, optimizer, device, scheduler, len(train) )
  print(f'Train loss {train_loss} accuracy {train_acc} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_F_Percision, val_M_Percision, val_F_Recall, val_M_Recall = eval_model(imgModel, test_loader, ce_loss, device, len(test) )
  print(f'Val loss {val_loss} accuracy {val_acc} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['train_F_Percision'].append (train_F_Percision)
  history['train_M_Percision'].append (train_M_Percision)
  history['train_F_Recall'].append (train_F_Recall)
  history['train_M_Recall'].append (train_M_Recall)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  history['val_F_Percision'].append (val_F_Percision)
  history['val_M_Percision'].append (val_M_Percision)
  history['val_F_Recall'].append (val_F_Recall)
  history['val_M_Recall'].append (val_M_Recall)
  if val_acc > best_accuracy:
    torch.save(imgModel.state_dict(), 'LeViTModel')
    best_accuracy = val_acc

#loading the model
imgModel = ImageClassifier (levit)
imgModel = imgModel.to (device)
imgModel.load_state_dict(torch.load('LeViTModel1'))

# We want images to be in order in train, here. We don't want shuffle
train_loader = DataLoader(train, batch_size=19, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)
val_loader = DataLoader(validation, batch_size=19, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)
test_loader = DataLoader(test, batch_size=19, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)

def TransformerModel (model, dataloader, device):
  model = model.eval()
  out = []
  targets = []
  A = []
  P = []
  for data in dataloader:
      feature = data['pixel_values'].to (device)
      targets.append(data['labels'][0].item())
      outputs = model (feature)
      out.append (outputs.logits.tolist())
      target = data['labels'].to (device)
      _, preds = torch.max (outputs.logits, dim=1)
      l = len (target[:])
      for i in range(0, l):
        a = target[:][i].item()
        p = preds[:][i].item()
        A.append (a)
        P.append (p)
  return out, targets, A, P

#loading the model
imgModel = ImageClassifier (levit)
imgModel = imgModel.to (device)
imgModel.load_state_dict(torch.load('LeViTModel'))

train_concat, train_labels, A, P = TransformerModel (imgModel, train_loader, device)
val_concat, test_labels, A, P = TransformerModel (imgModel, val_loader, device)
test_concat, test_labels, A, P = TransformerModel (imgModel, test_loader, device)

d0 = [x[0] for sub in train_concat for x in sub]
d1 = [x[1] for sub in train_concat for x in sub]
d2 = [x for x in train_labels for q in range(0, 10)] # Batch Size was 10
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'labels':d2})
df.to_csv ('LeViT_train_combine.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in val_concat for x in sub]
d1 = [x[1] for sub in val_concat for x in sub]
d2 = [x for x in val_labels for q in range(0, 10)] # Batch Size was 10
df1 = pd.DataFrame(data={'data0':d0, 'data1':d1, 'labels':d2})
df1.to_csv ('LeViT_val_combine.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in test_concat for x in sub]
d1 = [x[1] for sub in test_concat for x in sub]
d2 = [x for x in test_labels for q in range(0, 10)] # Batch Size was 10
df2 = pd.DataFrame(data={'data0':d0, 'data1':d1, 'labels':d2})
df2.to_csv ('LeViT_test_combine.csv', index=False, encoding='utf-8')

# Combining images starts from here.
train_ds = []
train_lbl = []
dummy = []
q = 0
for i in df.index:
  dummy.append (df['data0'][i])
  dummy.append (df['data1'][i])
  q += 1
  if q == 10:
    train_ds.append (torch.tensor(dummy))
    train_lbl.append (df['labels'][i-1])
    q = 0
    dummy = []
val_ds = []
val_lbl = []
dummy = []
q = 0
for i in df1.index:
  dummy.append (df1['data0'][i])
  dummy.append (df1['data1'][i])
  q += 1
  if q == 10:
    val_ds.append (torch.tensor(dummy))
    val_lbl.append (df1['labels'][i-1])
    q = 0
    dummy = []
test_ds = []
test_lbl = []
dummy = []
q = 0
for i in df1.index:
  dummy.append (df1['data0'][i])
  dummy.append (df1['data1'][i])
  q += 1
  if q == 10:
    test_ds.append (torch.tensor(dummy))
    test_lbl.append (df1['labels'][i-1])
    q = 0
    dummy = []

train_ds = TensorDataset (torch.stack(train_ds), torch.tensor (train_lbl))
val_ds = TensorDataset (torch.stack(val_ds), torch.tensor (val_lbl))
test_ds = TensorDataset (torch.stack(test_ds), torch.tensor (test_lbl))

train_loader = DataLoader (train_ds, batch_size=16, num_workers=2, shuffle=True)
test_loader = DataLoader (test_ds, batch_size=16, num_workers=2, shuffle=False)
val_loader = DataLoader (val_ds, batch_size=16, num_workers=2, shuffle=False)

# Model for Combining the 10 images
class Combine10ImageClassifier (nn.Module):
  def __init__ (self, dropout, hidden, input_size=20, classes=2):
    super (Combine10ImageClassifier, self).__init__()
    self.dropout = nn.Dropout (p= dropout)
    self.regressor1 = nn.Linear (input_size, hidden)
    self.regressor2 = nn.Linear (hidden, classes)
  def forward (self, input):
    output = self.dropout (input)
    output = self.regressor1 (output)
    output = self.regressor2 (output)
    return output

model = Combine10ImageClassifier (dropout=0.5, hidden = 8)
model = model.to (device)

epochs = 2000
optimizer = AdamW (model.parameters(), lr=0.001)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup (optimizer, num_warmup_steps=0, num_training_steps=total_steps)
ce_loss = nn.CrossEntropyLoss().to (device)

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
      targets = data[1].to (device)
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

def eval_model (model, dataloader, ce_loss, device, entry_size):
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
      targets = data[1].to (device)
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

torch.cuda.empty_cache()
# For saving the history
history = defaultdict(list)
best_accuracy = 0
for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_F_Percision, train_M_Percision, train_F_Recall, train_M_Recall = train_epochs (model, train_loader, ce_loss, optimizer, device, scheduler, len(train_ds) )
  print(f'Train loss {train_loss} accuracy {train_acc} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_F_Percision, val_M_Percision, val_F_Recall, val_M_Recall = eval_model(model, test_loader, ce_loss, device, len(test_ds) )
  print(f'Val loss {val_loss} accuracy {val_acc} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['train_F_Percision'].append (train_F_Percision)
  history['train_M_Percision'].append (train_M_Percision)
  history['train_F_Recall'].append (train_F_Recall)
  history['train_M_Recall'].append (train_M_Recall)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  history['val_F_Percision'].append (val_F_Percision)
  history['val_M_Percision'].append (val_M_Percision)
  history['val_F_Recall'].append (val_F_Recall)
  history['val_M_Recall'].append (val_M_Recall)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'SwinModelCombiningImages')
    best_accuracy = val_acc

model = Combine10ImageClassifier (dropout=0.1, hidden = 8)
model = model.to (device)
model.load_state_dict(torch.load('LeViTModelCombinigImages'))

train_loader = DataLoader (train_ds, batch_size=1, num_workers=2, shuffle=False)
val_loader = DataLoader (val_ds, batch_size=1, num_workers=2, shuffle=False)
test_loader = DataLoader (test_ds, batch_size=1, num_workers=2, shuffle=False)

def TheImageCombinedModel (model, dataloader, device):
  model = model.eval()
  out = []
  targets = []
  for data in dataloader:
      input_ids = data[0].to (device)
      target = data[1].to (device)
      outputs = model (input_ids.float())
      l = len (target)
      for i in range (l):
        dummy = outputs[i].tolist()
        out.append (np.argmax(dummy))
        targets.append (target[i].item())
  return out, targets

P, A = TheImageCombinedModel (model, test_loader, device)

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

train_loader = DataLoader (train_ds, batch_size=16, num_workers=2, shuffle=False) #no shuffles
val_loader = DataLoader (val_ds, batch_size=16, num_workers=2, shuffle=False) #no shuffles
test_loader = DataLoader (test_ds, batch_size=16, num_workers=2, shuffle=False) #no shuffles

def TheImageCombinedModel (model, dataloader, device):
  model = model.eval()
  out0 = []
  out1 = []
  targets = []
  for data in dataloader:
      input_ids = data[0].to (device)
      target = data[1].to (device)
      outputs = model (input_ids.float())
      l = len (target)
      for i in range (l):
        dummy = outputs[i].tolist()
        out0.append (dummy[0])
        out1.append (dummy[1])
        targets.append (target[i].item())
  return out0, out1, targets

P0, P1, A = TheImageCombinedModel (model, train_loader, device)
df = pd.DataFrame (data= {'data0':P0, 'data1':P1, 'Label':A})
df.to_csv ('LeViT_train_final_output.csv', index=False)
P0, P1, A = TheImageCombinedModel (model, val_loader, device)
df = pd.DataFrame (data= {'data0':P0, 'data1':P1, 'Label':A})
df.to_csv ('LeViT_validation_final_output.csv', index=False)
P0, P1, A = TheImageCombinedModel (model, test_loader, device)
df = pd.DataFrame (data= {'data0':P0, 'data1':P1, 'Label':A})
df.to_csv ('LeViT_test_final_output.csv', index=False)

os.remove ('LeViT_train_combine.csv')
os.remove ('LeViT_val_combine.csv')
os.remove ('LeViT_test_combine.csv')
