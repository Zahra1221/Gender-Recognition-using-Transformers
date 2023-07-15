import os
import shutil
from os import listdir
import cv2
import urllib.request
import PIL
from PIL import Image
import pandas as pd
import numpy as np
from PIL import Image as img

# In[]

path = os.path.abspath(__file__)
path = os.path.abspath(os.path.join(path, os.pardir))
parent = os.path.abspath(os.path.join(path, os.pardir))

# In[]

train = pd.read_csv (path+'\\Dataset\\train.csv', lineterminator='\n')
val = pd.read_csv (path+'\\Dataset\\validatoin.csv', lineterminator='\n')
test = pd.read_csv (path+'\\Dataset\\test.csv', lineterminator='\n')


# In[]

os.mkdir(parent+'\\kaggle_image_train')
os.mkdir(parent+'\\kaggle_image_train\\B')
os.mkdir(parent+'\\kaggle_image_train\\F')
os.mkdir(parent+'\\kaggle_image_train\\M')

os.mkdir(parent+'\\kaggle_image_validation')
os.mkdir(parent+'\\kaggle_image_validation\\B')
os.mkdir(parent+'\\kaggle_image_validation\\F')
os.mkdir(parent+'\\kaggle_image_validation\\M')

os.mkdir(parent+'\\kaggle_image_test')
os.mkdir(parent+'\\kaggle_image_test\\B')
os.mkdir(parent+'\\kaggle_image_test\\F')
os.mkdir(parent+'\\kaggle_image_test\\M')

# In[]

train_images = {}
for item in pd.unique(train['UserID']).tolist():
    dummy = train[train['UserID'] == item]
    dummy = dummy.reset_index (drop= True)
    train_images[item] = [dummy['ProfileImage'][0], dummy['Gender'][0]]
    
val_images = {}
for item in pd.unique(val['UserID']).tolist():
    dummy = val[val['UserID'] == item]
    dummy = dummy.reset_index (drop= True)
    val_images[item] = [dummy['ProfileImage'][0], dummy['Gender'][0]]
    
test_images = {}
for item in pd.unique(test['UserID']).tolist():
    dummy = test[test['UserID'] == item]
    dummy = dummy.reset_index (drop= True)
    test_images[item] = [dummy['ProfileImage'][0], dummy['Gender'][0]]

# In[]

os.mkdir(parent+'\\images')

# In[]

for item in train_images.keys():
    g = ''
    if train_images[item][1] == 'brand':
      g = 'B'
    elif train_images[item][1] == 'female':
      g = 'F'
    elif train_images[item][1] == 'male':
      g = 'M'
    
    try:
      imurl = train_images[item][0].replace('_normal','')
      prefix =  imurl[-5:] if imurl[-5] == '.' else imurl[-4:]
      img = urllib.request.urlretrieve (imurl, parent + '\\images\\image' + prefix)
      image = PIL.Image.open(parent + '\\images\\image' + prefix)
      image = image.resize((224, 224))
      image.save (parent+'\\kaggle_image_train\\' + g + '\\' + item + prefix)
    except:
      print ('profile image for user with id ' + item + ', image url ' + train_images[item][0] + 'was not extracted.')
      continue

# In[]

for item in val_images.keys():
    g = ''
    if val_images[item][1] == 'brand':
      g = 'B'
    elif val_images[item][1] == 'female':
      g = 'F'
    elif val_images[item][1] == 'male':
      g = 'M'
    
    try:
      imurl = val_images[item][0].replace('_normal','')
      prefix = imurl[-5:] if imurl[-5] == '.' else imurl[-4:]
      img = urllib.request.urlretrieve (imurl, parent + '\\images\\image' + prefix)
      image = PIL.Image.open(parent + '\\images\\image' + prefix)
      image = image.resize((224, 224))
      image.save (parent+'\\kaggle_image_validation\\' + g + '\\' + item + prefix)
    except:
      print ('profile image for user with id ' + item + ', image url ' + train_images[item][0] + 'was not extracted.')
      continue
  
# In[]

for item in test_images.keys():
    g = ''
    if test_images[item][1] == 'brand':
      g = 'B'
    elif test_images[item][1] == 'female':
      g = 'F'
    elif test_images[item][1] == 'male':
      g = 'M'
    
    try:
      imurl = test_images[item][0].replace('_normal','')
      prefix =  imurl[-5:] if imurl[-5] == '.' else imurl[-4:]
      img = urllib.request.urlretrieve (imurl, parent + '\\images\\image' + prefix)
      image = PIL.Image.open(parent + '\\images\\image' + prefix)
      image = image.resize((224, 224))
      image.save (parent+'\\kaggle_image_test\\' + g + '\\' + item + prefix)
    except:
      print ('profile image for user with id ' + item + ', image url ' + train_images[item][0] + 'was not extracted.')
      continue
  
# In[]

files = shutil.rmtree (parent + '\\images')


