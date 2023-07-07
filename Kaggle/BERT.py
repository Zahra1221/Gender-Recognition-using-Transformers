import torch
import pytorch_lightning as pl
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import classification_report
from collections import defaultdict
import transformers
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from collections import defaultdict
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModel, BertTokenizerFast, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import emoji
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))
data = parent + '/Dataset'

train = pd.read_csv (data+'/train.csv')
val = pd.read_csv (data+'/validation.csv')
test = pd.read_csv (data+'/test.csv')

ids_train = pd.unique(train['id']).tolist
ids = []
texts = []
for i in ids_train:
  dummy = train[train['id'] == i]
  for t in list(dummy['text']):
    # Cleaning
    t = emoji.demojize(t)
    t = re.sub ('\S+@\S+', '', t)
    t = re.sub ('@\S+', '', t)
    t = re.sub ('http\S+', '', t)
    t = t.replace ('....', '.')
    t = t.replace ('...', '.')
    t = t.replace ('..', '.')
    t = t.replace ('   ', ' ')
    t = t.replace ('  ', ' ')
    t = t.replace (',,,,', ',')
    t = t.replace (',,,', ',')
    t = t.replace (',,', ',')
    if t[0] == ' ':
      t = t[1:]
    s = len (t)
    if s != 0 and t[s-1] == ' ':
      t = t[:s-1]
    p1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,:.!?' -$%()[]/}{_"
    clean = ''
    for letter in t:
        if letter in p1:# or letter == '"':
            clean += letter
    clean = clean.replace ('....', '.')
    clean = clean.replace ('...', '.')
    clean = clean.replace ('..', '.')
    clean = clean.replace ('   ', ' ')
    clean = clean.replace ('  ', ' ')
    t = emoji.emojize(clean)
    t = t.replace ('amp', 'and')
    ids.append (i)
    texts.append (t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv (path+'/kaggle_train_texts.csv', index=False)

ids_val = pd.unique(val['id']).tolist
ids = []
texts = []
for i in ids_val:
  dummy = val[val['id'] == i]
  for t in list(dummy['text']):
    # Cleaning
    t = emoji.demojize(t)
    t = re.sub ('\S+@\S+', '', t)
    t = re.sub ('@\S+', '', t)
    t = re.sub ('http\S+', '', t)
    t = t.replace ('....', '.')
    t = t.replace ('...', '.')
    t = t.replace ('..', '.')
    t = t.replace ('   ', ' ')
    t = t.replace ('  ', ' ')
    t = t.replace (',,,,', ',')
    t = t.replace (',,,', ',')
    t = t.replace (',,', ',')
    if t[0] == ' ':
      t = t[1:]
    s = len (t)
    if s != 0 and t[s-1] == ' ':
      t = t[:s-1]
    p1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,:.!?' -$%()[]/}{_"
    clean = ''
    for letter in t:
        if letter in p1:# or letter == '"':
            clean += letter
    clean = clean.replace ('....', '.')
    clean = clean.replace ('...', '.')
    clean = clean.replace ('..', '.')
    clean = clean.replace ('   ', ' ')
    clean = clean.replace ('  ', ' ')
    t = emoji.emojize(clean)
    t = t.replace ('amp', 'and')
    ids.append (i)
    texts.append (t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv (path+'/kaggle_val_texts.csv', index=False)

ids_test = pd.unique(test['id']).tolist
ids = []
texts = []
for i in ids_test:
  dummy = test[test['id'] == i]
  for t in list(dummy['text']):
    # Cleaning
    t = emoji.demojize(t)
    t = re.sub ('\S+@\S+', '', t)
    t = re.sub ('@\S+', '', t)
    t = re.sub ('http\S+', '', t)
    t = t.replace ('....', '.')
    t = t.replace ('...', '.')
    t = t.replace ('..', '.')
    t = t.replace ('   ', ' ')
    t = t.replace ('  ', ' ')
    t = t.replace (',,,,', ',')
    t = t.replace (',,,', ',')
    t = t.replace (',,', ',')
    if t[0] == ' ':
      t = t[1:]
    s = len (t)
    if s != 0 and t[s-1] == ' ':
      t = t[:s-1]
    p1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,:.!?' -$%()[]/}{_"
    clean = ''
    for letter in t:
        if letter in p1:# or letter == '"':
            clean += letter
    clean = clean.replace ('....', '.')
    clean = clean.replace ('...', '.')
    clean = clean.replace ('..', '.')
    clean = clean.replace ('   ', ' ')
    clean = clean.replace ('  ', ' ')
    t = emoji.emojize(clean)
    t = t.replace ('amp', 'and')
    ids.append (i)
    texts.append (t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv (path+'/kaggle_test_texts.csv', index=False)

train_texts = pd.read_csv (path+'/kaggle_train_texts.csv')
val_texts = pd.read_csv (path+'/kaggle_val_texts.csv')
test_texts = pd.read_csv (path+'/kaggle_test_texts.csv')

bert = BertModel.from_pretrained ('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained ('bert-base-cased')

ids = pd.unique(train_texts['ids']).tolist()
id2texts_train = {}
for i in ids:
  id2texts_train[i] = []
  df1 = train_texts[train_texts['ids'] == i]
  for j in range (0, 9):
    id2texts_test[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
  id2texts_test[i].append (' '.join (list(df1['texts'].astype(str))[90:]))

ids = pd.unique(val_texts['ids']).tolist()
id2texts_val = {}
for i in ids:
  id2texts_val[i] = []
  df1 = val_texts[val_texts['ids'] == i]
  for j in range (0, 9):
    id2texts_val[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
  id2texts_val[i].append (' '.join (list(df1['texts'].astype(str))[90:]))

ids = pd.unique(test_texts['ids']).tolist()
id2texts_test = {}
for i in ids:
  id2texts_test[i] = []
  df1 = test_texts[test_texts['ids'] == i]
  for j in range (0, 9):
    id2texts_test[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
  id2texts_test[i].append (' '.join (list(df1['texts'].astype(str))[90:]))

seq_len = [len (x.split()) for ls in list(id2texts_train.values()) for x in ls]
max_len = max(seq_len)
pd.Series (seq_len).hist(bins = 30)
print(max_len)
seq_len = [len (x.split()) for ls in list(id2texts_val.values()) for x in ls]
max_len = max(seq_len)
pd.Series (seq_len).hist(bins = 30)
print(max_len)
seq_len = [len (x.split()) for ls in list(id2texts_test.values()) for x in ls]
max_len = max(seq_len)
pd.Series (seq_len).hist(bins = 30)
print(max_len)

max_len = 256
id2tokens_train = {}
for k in list(id2texts_train.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_train[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_train[k] = tokens
id2tokens_val = {}
for k in list(id2texts_val.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_val[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_val[k] = tokens
id2tokens_test = {}
for k in list(id2texts_test.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_test[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_test[k] = tokens

# Convert lists to tensors
ids = [x for x in range(0, len(id2tokens_train)) for i in range(0, 10)]
ids = torch.tensor (ids)
train_seq = [x for k in list(id2tokens_train.keys()) for x in id2tokens_train[k]['input_ids']]
train_seq = torch.tensor (train_seq)
train_mask = [x for k in list(id2tokens_train.keys()) for x in id2tokens_train[k]['attention_mask']]
train_mask = torch.tensor (train_mask)
train_y = []
for k in list(id2tokens_train.keys()):
  for i in range(10):
    if id2gender_train[k] == 'B':
      train_y.append (0)
    elif id2gender_train[k] == 'F': 
      train_y.append (1)
    else:
      train_y.append (2)
train_y = torch.tensor (train_y)

ids = [x for x in range(0, len(id2tokens_val)) for i in range(0, 10)]
ids = torch.tensor (ids)
val_seq = [x for k in list(id2tokens_val.keys()) for x in id2tokens_val[k]['input_ids']]
val_seq = torch.tensor (val_seq)
val_mask = [x for k in list(id2tokens_val.keys()) for x in id2tokens_val[k]['attention_mask']]
val_mask = torch.tensor (val_mask)
val_y = []
for k in list(id2tokens_val.keys()):
  for i in range(10):
    if id2gender_val[k] == 'B':
      train_y.append (0)
    elif id2gender_val[k] == 'F': 
      val_y.append (1)
    else:
      val_y.append (2)
val_y = torch.tensor (val_y)

ids2 = [x for x in range(0, len(id2tokens_test)) for i in range(0, 10)]
ids2 = torch.tensor (ids2)
test_seq = [x for k in list(id2tokens_test.keys()) for x in id2tokens_test[k]['input_ids']]
test_seq = torch.tensor (test_seq)
test_mask = [x for k in list(id2tokens_test.keys()) for x in id2tokens_test[k]['attention_mask']]
test_mask = torch.tensor (test_mask)
test_y = []
for k in list(id2tokens_test.keys()):
  for i in range(10):
    if id2gender_test[k] == 'B':
      test_y.append (0)
    elif id2gender_test[k] == 'F': 
      test_y.append (1)
    else:
      test_y.append (2)
test_y = torch.tensor (test_y)

batch_size = 32

train_data = TensorDataset (train_seq, train_mask, train_y, ids)
train_sampler = RandomSampler (train_data)
train_dataloader = DataLoader (train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset (val_seq, val_mask, val_y, ids)
val_sampler = RandomSampler (val_data)
val_dataloader = DataLoader (val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset (test_seq, test_mask, test_y, ids2)
test_sampler = SequentialSampler (test_data)
test_dataloader = DataLoader (test_data, sampler=test_sampler, batch_size=batch_size)

classes = 3
class BertGenderClassifier (nn.Module):
  def __init__ (self, bert, classes):
    super (BertGenderClassifier, self).__init__()
    self.bert = bert
    self.dropout = nn.Dropout (p=0.1)
    self.out = nn.Linear (self.bert.config.hidden_size, classes)
  def forward (self, input_ids, attention_mask):
    _, bert_output = self.bert (input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    #print (bert_output)
    output = self.dropout (bert_output)
    return self.out (output)

# Create an instance of our model and push it to GPU
model = BertGenderClassifier (bert, classes)
model = model.to (device)

epochs = 8 #20
optimizer = AdamW (model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup (optimizer, num_warmup_steps=0, num_training_steps=total_steps)
ce_loss = nn.CrossEntropyLoss().to (device)

# A function for training 
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
    input_ids = data[0].to (device)
    attention_mask = data[1].to (device)
    targets = data[2].to (device)
    outputs = model (input_ids=input_ids, attention_mask=attention_mask)
    _, preds = torch.max (outputs, dim=1)
    loss = ce_loss (outputs, targets)
    correct_predictions_count += torch.sum (preds == targets)
    losses.append (loss.item())
    loss.backward ()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    B_correct += torch.sum ((preds == 0) & (preds == targets))
    B_incorrect += torch.sum ((preds == 0) & (preds != targets))
    F_correct += torch.sum ((preds == 1) & (preds == targets))
    F_incorrect += torch.sum ((preds == 1) & (preds != targets))
    M_correct += torch.sum ((preds == 2) & (preds == targets))
    M_incorrect += torch.sum ((preds == 2) & (preds != targets))
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
      input_ids = data[0].to (device)
      attention_mask = data[1].to (device)
      targets = data[2].to (device)
      outputs = model (input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max (outputs, dim=1)
      loss = ce_loss (outputs, targets)
      l = len (targets[:])
      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())
      B_correct += torch.sum ((preds == 0) & (preds == targets))
      B_incorrect += torch.sum ((preds == 0) & (preds != targets))
      F_correct += torch.sum ((preds == 1) & (preds == targets))
      F_incorrect += torch.sum ((preds == 1) & (preds != targets))
      M_correct += torch.sum ((preds == 2) & (preds == targets))
      M_incorrect += torch.sum ((preds == 2) & (preds != targets))
  return correct_predictions_count.double() / entry_size, np.mean(losses), B_correct / (B_correct + B_incorrect), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), B_correct / (B_correct + F_incorrect + M_incorrect), F_correct / (F_correct + B_incorrect + M_incorrect), M_correct / (M_correct + B_incorrect + F_incorrect)

torch.cuda.empty_cache()
# For saving the history
history = defaultdict(list)
best_accuracy = 0
for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_B_Percision, train_F_Percision, train_M_Percision, train_B_Recall, train_F_Recall, train_M_Recall = train_epochs (model, train_dataloader, ce_loss, optimizer, device, scheduler, len(train_data) )
  print(f'Train loss {train_loss} accuracy {train_acc} Brand Percision {train_B_Percision} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Brand Recall {train_B_Recall} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_B_Percision, val_F_Percision, val_M_Percision, val_B_Recall, val_F_Recall, val_M_Recall = eval_model(model, val_dataloader, ce_loss, device, len(val_data) )
  print(f'Val loss {val_loss} accuracy {val_acc} Brand Percision {val_B_Percision} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Brand Recall {val_B_Recall} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'BERTModel')
    best_accuracy = val_acc

#loading the model
model = BertGenderClassifier (bert, classes)
model = model.to (device)
model.load_state_dict(torch.load('BERTModel'))

l = len (id2texts_train) / 2

ids = pd.unique(train_texts['ids']).tolist()
id2texts_train1 = {}
id2texts_train2 = {}
q = 0
for i in ids:
  if q < l:
    id2texts_train1[i] = []
    df1 = train_texts[train_texts['ids'] == i]
    for j in range (0, 10):
      id2texts_train1[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
  else:
    id2texts_train2[i] = []
    df1 = train_texts[train_texts['ids'] == i]
    for j in range (0, 10):
      id2texts_train2[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))   
  q += 1

id2tokens_train1 = {}
id2tokens_train2 = {}
for k in list(id2texts_train1.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_train1[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_train1[k] = tokens
for k in list(id2texts_train2.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_train2[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_train2[k] = tokens

ids1 = [x for x in range(0, len(id2tokens_train1)) for i in range(0, 10)]
ids1 = torch.tensor (ids1)
train_seq1 = [x for k in list(id2tokens_train1.keys()) for x in id2tokens_train1[k]['input_ids']]
train_seq1 = torch.tensor (train_seq1)
train_mask1 = [x for k in list(id2tokens_train1.keys()) for x in id2tokens_train1[k]['attention_mask']]
train_mask1 = torch.tensor (train_mask1)
train_y1 = []
for k in list(id2tokens_train1.keys()):
  for i in range(0, 10):
    if id2gender_train1[k] == 'B':
      train_y1.append(0)
    elif id2gender_train1[k] == 'F':
      train_y1.append(1)
    else:
      train_y1.append(2)
train_y1 = torch.tensor (train_y1)

ids2 = [x for x in range(0, len(id2tokens_train2)) for i in range(0, 10)]
ids2 = torch.tensor (ids2)
train_seq2 = [x for k in list(id2tokens_train2.keys()) for x in id2tokens_train2[k]['input_ids']]
train_seq2 = torch.tensor (train_seq2)
train_mask2 = [x for k in list(id2tokens_train2.keys()) for x in id2tokens_train2[k]['attention_mask']]
train_mask2 = torch.tensor (train_mask2)
train_y2 = []
for k in list(id2tokens_train2.keys()):
  for i in range(0, 10):
    if id2gender_train2[k] == 'B':
      train_y2.append(0)
    elif id2gender_train2[k] == 'F':
      train_y2.append(1)
    else:
      train_y2.append(2)
train_y2 = torch.tensor (train_y2)

batch_size = 8
train_data1 = TensorDataset (train_seq1, train_mask1, train_y1, ids1)
#train_sampler1 = RandomSampler (train_data1) # I don't want the order of the data to change
train_dataloader1 = DataLoader (train_data1, batch_size=batch_size)
train_data2 = TensorDataset (train_seq2, train_mask2, train_y2, ids2)
#train_sampler2 = RandomSampler (train_data2) # I don't want the order of the data to change
train_dataloader2 = DataLoader (train_data2, batch_size=batch_size)

def TheBERTModel (model, dataloader, device):
  model = model.eval()
  out = []
  targets = []
  for data in dataloader:
      input_ids = data[0].to (device)
      attention_mask = data[1].to (device)
      targets.append(data[2][0].item())    
      outputs = model (input_ids, attention_mask)
      out.append (outputs.tolist())   
  return out, targets

torch.cuda.empty_cache()
bert_train_out1, bert_train_labels1 = TheBERTModel (model, train_dataloader1, device)

b = []
f = []
m = []
l = len (bert_train_out1)
for i in range (0, l):
  for j in range (0, 8):
    b.append(bert_train_out1[i][j][0])
    f.append(bert_train_out1[i][j][1])
    m.append(bert_train_out1[i][j][2])

labels = []
for item in train_data1:
  labels.append (item[2].item())

df = pd.DataFrame (data={'B':b, 'F':f, 'M':m, 'labels':labels})
df.to_csv (path+'/bert_train_output1.csv')

bert_train_out2, bert_train_labels2 = TheBERTModel (model, train_dataloader2, device)

b = []
f = []
m = []
l = len (bert_train_out2)
for i in range (0, l):
  for j in range (0, 8):
    b.append(bert_train_out2[i][j][0])
    f.append(bert_train_out2[i][j][1])
    m.append(bert_train_out2[i][j][2])

labels = []
for item in train_data2:
  labels.append (item[2].item())

df = pd.DataFrame (data={'B':b, 'F':f, 'M':m, 'labels':labels})
df.to_csv (path+'/bert_train_output2.csv')

batch_size = 8
val_data = TensorDataset (val_seq, val_mask, val_y)#, ids2)
#val_sampler = SequentialSampler (val_data)
val_dataloader = DataLoader (val_data, batch_size=batch_size)

bert_val_out, bert_val_labels = TheBERTModel (model, val_dataloader, device)

b = []
f = []
m = []
l = len (bert_val_out)
for i in range (0, l):
  for j in range (0, 8):
    b.append(bert_val_out[i][j][0])
    f.append(bert_val_out[i][j][1])
    m.append(bert_val_out[i][j][2])

labels = []
for item in val_data:
  labels.append (item[2].item())

df = pd.DataFrame (data={'B':b, 'F':f, 'M':m, 'labels':labels})
df.to_csv (path+'/bert_val_output.csv')

batch_size = 8
test_data = TensorDataset (test_seq, test_mask, test_y)#, ids2)
#test_sampler = SequentialSampler (test_data)
test_dataloader = DataLoader (test_data, batch_size=batch_size)

bert_test_out, bert_test_labels = TheBERTModel (model, test_dataloader, device)

b = []
f = []
m = []
l = len (bert_test_out)
for i in range (0, l):
  for j in range (0, 8):
    b.append(bert_test_out[i][j][0])
    f.append(bert_test_out[i][j][1])
    m.append(bert_test_out[i][j][2])

labels = []
for item in test_data:
  labels.append (item[2].item())

df = pd.DataFrame (data={'B':b, 'F':f, 'M':m, 'labels':labels})
df.to_csv (path+'/bert_test_output.csv')

df0 = pd.read_csv (path+'/bert_train_output1.csv')
df1 = pd.read_csv (path+'/bert_train_output2.csv')
df2 = pd.read_csv (path+'/bert_val_output.csv')
df3 = pd.read_csv (path+'/bert_test_output.csv')

df = pd.concat([df0, df1], ignore_index=True, sort=False)
df = df.reset_index (drop=True)

# Combining
train_output = []
train_label = []
dummy = []
for i in df.index:
  if i != 0 and i%10 == 0:
    train_output.append (torch.tensor (dummy))
    train_label.append (df['labels'][i-1])
    dummy = []
  dummy.append (df['B'][i])
  dummy.append (df['F'][i])
  dummy.append (df['M'][i])
train_output.append (torch.tensor (dummy))
train_label.append (df['labels'][i-1])
val_output = []
val_label = []
dummy = []
for i in df2.index:
  if i != 0 and i%10 == 0:
    val_output.append (torch.tensor (dummy))
    val_label.append (df2['labels'][i-1])
    dummy = []
  dummy.append (df2['B'][i])
  dummy.append (df2['F'][i])
  dummy.append (df2['M'][i])
val_output.append (torch.tensor (dummy))
val_label.append (df2['labels'][i-1])
test_output = []
test_label = []
dummy = []
for i in df3.index:
  if i != 0 and i%10 == 0:
    test_output.append (torch.tensor (dummy))
    test_label.append (df3['labels'][i-1])
    dummy = []
  dummy.append (df3['B'][i])
  dummy.append (df3['F'][i])
  dummy.append (df3['M'][i])
test_output.append (torch.tensor (dummy))
test_label.append (df3['labels'][i-1])

train_ds = TensorDataset (torch.stack(train_output), torch.tensor(train_label))
val_ds = TensorDataset (torch.stack(val_output), torch.tensor(val_label))
test_ds = TensorDataset (torch.stack(test_output), torch.tensor(test_label))

train_loader = DataLoader (train_ds, batch_size=16, num_workers=2, shuffle=True)
val_loader = DataLoader (val_ds, batch_size=16, num_workers=2, shuffle=True)
test_loader = DataLoader (test_ds, batch_size=16, num_workers=2, shuffle=True)

# Model for Combining the 10 images
class BertCombine10Classifier (nn.Module):
  def __init__ (self, hidden, dropout, input_size=20, classes=2):
    super (BertCombine10Classifier, self).__init__()
    self.dropout = nn.Dropout (p= dropout)
    self.lin1 = nn.Linear (input_size, hidden)
    self.lin2 = nn.Linear (hidden, classes)
  def forward (self, input):
    #output = self.dropout (input)
    output = self.lin1(input)
    output = self.lin2(output)
    return output

model = BertCombine10Classifier (hidden= 10, dropout= 0.1)
model = model.to (device)

epochs = 2000
optimizer = torch.optim.adam (model.parameters(), lr=0.001)
total_steps = len(test_loader) * epochs
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
      input_ids = data[0].to (device) 
      targets = data[1].to (device)
      outputs = model (input_ids.float())
      _, preds = torch.max (outputs, dim=1)
      loss = ce_loss (outputs, targets)
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
      input_ids = data[0].to (device)    
      targets = data[1].to (device)
      outputs = model (input_ids.float())     
      _, preds = torch.max (outputs, dim=1)
      loss = ce_loss (outputs, targets)     
      l = len (targets[:])
      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())
      B_correct += torch.sum ((preds == 0) & (preds == targets))
      B_incorrect += torch.sum ((preds == 0) & (preds != targets))
      F_correct += torch.sum ((preds == 1) & (preds == targets))
      F_incorrect += torch.sum ((preds == 1) & (preds != targets))
      M_correct += torch.sum ((preds == 2) & (preds == targets))
      M_incorrect += torch.sum ((preds == 2) & (preds != targets))
  return correct_predictions_count.double() / entry_size, np.mean(losses), B_correct / (B_correct + B_incorrect), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), B_correct / (B_correct + F_incorrect + M_incorrect), F_correct / (F_correct + B_incorrect + M_incorrect), M_correct / (M_correct + B_incorrect + F_incorrect)

torch.cuda.empty_cache()
# For saving the history
history = defaultdict(list)
best_accuracy = 0
for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)
  train_acc, train_loss, train_B_Percision, train_F_Percision, train_M_Percision, train_B_Recall, train_F_Recall, train_M_Recall = train_epochs (model, train_loader, ce_loss, optimizer, device, scheduler, len(train_ds) )
  print(f'Train loss {train_loss} accuracy {train_acc} Brand Percision {train_B_Percision} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Brand Recall {train_B_Recall} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')
  val_acc, val_loss, val_B_Percision, val_F_Percision, val_M_Percision, val_B_Recall, val_F_Recall, val_M_Recall = eval_model(model, val_loader, ce_loss, device, len(val_ds) )
  print(f'Val loss {val_loss} accuracy {val_acc} Brand Percision {val_B_Percision} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Brand Recall {val_B_Recall} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'ModelCombiningTexts_BERT')
    best_accuracy = val_acc

#loading the model
model = BertCombine10Classifier ()
model = model.to (device)
model.load_state_dict(torch.load('ModelCombiningTexts_BERT'))

def TheTextCombinedModel (model, dataloader, device):
  model = model.eval()
  out = []
  targets = []
  A = []
  P = []
  for data in dataloader:
      input_ids = data[0].to (device)
      targets.append(data[1])
      outputs = model (input_ids.float())
      out.append (outputs.tolist())
      target = data[1].to (device)
      _, preds = torch.max (outputs, dim=1)
      l = len (target[:])
      for i in range(0, l):
        a = target[:][i].item()
        p = preds[:][i].item()
        A.append (a)
        P.append (p)   
  return out, targets, A, P
  
train_combine, test_combine_labels, A1, P1 = TheTextCombinedModel (model, train_loader, device)
val_combine, val_combine_labels, A2, P2 = TheTextCombinedModel (model, val_loader, device)
test_combine, test_combine_labels, A, P = TheTextCombinedModel (model, test_loader, device)

d0 = [x[0] for sub in train_combine for x in sub]
d1 = [x[1] for sub in train_combine for x in sub]
d2 = [x[2] for sub in train_combine for x in sub]
d3 = [x for x in A1]
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2 'labels':d3})
df.to_csv (path+'/train_text_combined_BERT.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in val_combine for x in sub]
d1 = [x[1] for sub in val_combine for x in sub]
d2 = [x[2] for sub in val_combine for x in sub]
d3 = [x for x in A2]
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2, 'labels':d3})
df.to_csv (path+'/val_text_combined_BERT.csv', index=False, encoding='utf-8')
d0 = [x[0] for sub in test_combine for x in sub]
d1 = [x[1] for sub in test_combine for x in sub]
d2 = [x[2] for sub in test_combine for x in sub]
d3 = [x for x in A]
df = pd.DataFrame(data={'data0':d0, 'data1':d1, 'data2':d2, 'labels':d3})
df.to_csv (path+'/test_text_combined_BERT.csv', index=False, encoding='utf-8')

os.remove(path+'/bert_train_output1.csv')
os.remove(path+'/bert_train_output2.csv')
os.remove(path+'/bert_val_output.csv')
os.remove(path+'/bert_test_output.csv')

CM = confusion_matrix(A, P)
CM = CM / len (P)
CM = pd.DataFrame(CM, index=['Brand', 'Female','Male'], columns=['Brand', 'Female','Male'])
plt.figure(figsize = (3,3))
sns.heatmap(CM, annot=True)
plt.xlabel("Predicted Values", fontsize = 11)
plt.ylabel("True Values", fontsize = 11)
plt.show()
target_names = ['Brand', 'Female', 'Male']
print(classification_report(A, P, target_names=target_names))




