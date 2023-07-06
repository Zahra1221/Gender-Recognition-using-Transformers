import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, AutoFeatureExtractor, SwinForImageClassification
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

train = ImageFolder (parent + '/train')
validation = ImageFolder (parent + '/validation')
test = ImageFolder (parent + '/test')

label2id = {'B':'0', 'F':'1', 'M':'2'}
id2label = {'0':'B', '1':'F', '2':'M'}
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swin-base-patch4-window7-224')
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
swin = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224', num_labels=len(label2id), label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)

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
  def __init__ (self, swin):
    super (ImageClassifier, self).__init__()
    self.swin = swin
  def forward (self, input_ids):
    output = self.swin (input_ids)
    return output

imgModel = ImageClassifier (swin)
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
  B_correct = 0
  B_incorrect = 0
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
    B_correct += torch.sum ((preds == 0) & (preds == targets))
    B_incorrect += torch.sum ((preds == 0) & (preds != targets))
    F_correct += torch.sum ((preds == 1) & (preds == targets))
    F_incorrect += torch.sum ((preds == 1) & (preds != targets))
    M_correct += torch.sum ((preds == 2) & (preds == targets))
    M_incorrect += torch.sum ((preds == 2) & (preds != targets))
    loss.backward ()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions_count.double() / entry_size, np.mean(losses), B_correct / (B_correct + B_incorrect), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), B_correct / (B_correct + F_incorrect + M_incorrect), F_correct / (F_correct + B_incorrect + M_incorrect), M_correct / (M_correct + B_incorrect + F_incorrect)

def eval_model (model, dataloader, ce_loss, device, entry_size):
  model = model.eval()
  losses = []
  correct_predictions_count = 0
  B_correct = 0
  B_incorrect = 0
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
      B_correct += torch.sum ((preds == 0) & (preds == targets))
      B_incorrect += torch.sum ((preds == 0) & (preds != targets))
      F_correct += torch.sum ((preds == 1) & (preds == targets))
      F_incorrect += torch.sum ((preds == 1) & (preds != targets))
      M_correct += torch.sum ((preds == 2) & (preds == targets))
      M_incorrect += torch.sum ((preds == 2) & (preds != targets))
  return correct_predictions_count.double() / entry_size, np.mean(losses), B_correct / (B_correct + B_incorrect), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), B_correct / (B_correct + F_incorrect+ M_incorrect), F_correct / (F_correct + B_incorrect + M_incorrect), M_correct / (M_correct + B_incorrect + F_incorrect)

torch.cuda.empty_cache()

# For saving the history
history = defaultdict(list)
best_accuracy = 0

for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_B_Percision, train_F_Percision, train_M_Percision, train_B_Recall, train_F_Recall, train_M_Recall = train_epochs (imgModel, train_loader, ce_loss, optimizer, device, scheduler, len(train) )
  print(f'Train loss {train_loss} accuracy {train_acc} Brand Percision {train_B_Percision} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Brand Recall {train_B_Recall} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_B_Percision, val_F_Percision, val_M_Percision, val_B_Recall, val_F_Recall, val_M_Recall = eval_model(imgModel, val_loader, ce_loss, device, len(val) )
  print(f'Val loss {val_loss} accuracy {val_acc} Brand Percision {val_B_Percision} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Brand Recall {val_B_Recall} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
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
    torch.save(imgModel.state_dict(), 'SwinModel1')
    best_accuracy = val_acc

#loading the model
imgModel = ImageClassifier (swin)
imgModel = imgModel.to (device)
imgModel.load_state_dict(torch.load('SwinModel1'))

def test_model (model, dataloader, ce_loss, device, entry_size):
  model = model.eval()
  A = []
  P = []
  with torch.no_grad():
    for data in dataloader:
      feature = data['pixel_values'].to (device)
      targets = data['labels'].to (device)
      outputs = model (feature)
      _, preds = torch.max (outputs.logits, dim=1)
      l = len (targets[:])
      for i in range(0, l):
        a = targets[:][i].item()
        p = preds[:][i].item()
        A.append (a)
        P.append (p)
  return A, P

A, P = test_model (imgModel, test_loader, ce_loss, device, len(test))

CM = confusion_matrix (A, P)
CM = CM / len (P)
CM = pd.DataFrame (CM, index=['Brand','Female','Male'], columns=['Brand','Female','Male'])

plt.figure(figsize = (3,3))
sns.heatmap(CM, annot=True)
plt.xlabel("Predicted Values", fontsize = 11)
plt.ylabel("True Values", fontsize = 11)

target_names = ['Brand','Female', 'Male']
print(classification_report(A, P, target_names=target_names))

train_loader = DataLoader(train, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)
val_loader = DataLoader(validation, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)
test_loader = DataLoader(test, batch_size=8, collate_fn=ImageClassificationCollator(feature_extractor), num_workers=2, shuffle=False)

#loading the model
imgModel = ImageClassifier (swin)
imgModel = imgModel.to (device)
imgModel.load_state_dict(torch.load('SwinModel1'))

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

train_concat, train_labels, A1, P1 = TransformerModel (imgModel, train_loader, device)
val_concat, val_labels, A2, P2 = TransformerModel (imgModel, val_loader, device)
test_concat, test_labels, A3, P3 = TransformerModel (imgModel, test_loader, device)

d0 = [x[0] for sub in train_concat for x in sub]
d1 = [x[1] for sub in train_concat for x in sub]
d2 = [x[2] for sub in train_concat for x in sub]
d3 = [x for x in train_labels] # Batch Size was 8
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2, 'labels':A1})
df.to_csv ('swin_Kaggle_train_final.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in train_concat for x in sub]
d1 = [x[1] for sub in train_concat for x in sub]
d2 = [x[2] for sub in train_concat for x in sub]
d3 = [x for x in train_labels] # Batch Size was 8
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2, 'labels':A2})
df.to_csv ('swin_Kaggle_val_final.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in train_concat for x in sub]
d1 = [x[1] for sub in train_concat for x in sub]
d2 = [x[2] for sub in train_concat for x in sub]
d3 = [x for x in train_labels] # Batch Size was 8
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2, 'labels':A3})
df.to_csv ('swin_Kaggle_test_final.csv', index=False, encoding='utf-8')

