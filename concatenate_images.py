import os
from os import listdir
import cv2
import pandas as pd
import numpy as np
from PIL import Image as img

# In[]
#path = os.getcwd()
path = os.path.abspath(__file__)
path = os.path.abspath(os.path.join(path, os.pardir))
parent = os.path.abspath(os.path.join(path, os.pardir))
trainpath = parent + '\\pan18-author-profiling-training-dataset-2018-02-27'
testpath = parent + '\\pan18-author-profiling-test-dataset-2018-03-20'
print(path, parent, trainpath, testpath)

# In[]
with open(trainpath+'\\en\\en.txt') as f:
  lines = f.readlines()

# In[]
id2gender_train = {}
for item in lines:
  n = 0
  v = ''
  if item[-7] == 'f':
    v = 'F'
    n = -10
  else:
    v = 'M'
    n = -8
  k = item[:n]
  id2gender_train[k] = v
  
# In[]
os.mkdir(parent+'\\images_train')
os.mkdir(parent+'\\images_train\\F')
os.mkdir(parent+'\\images_train\\M')

# In[]
for item in list(id2gender_train.keys()):
  try:
    p = trainpath + '\\en\\photo\\' + item
    cpy = [f for f in listdir(p)]
    q = 0
    for img in cpy:
      imgs = cpy.copy ()
      imgs.remove (img)
  #--------------------------------------------------------
      img1 = cv2.imread (p + '\\' + imgs[0])
      img2 = cv2.imread (p + '\\' + imgs[1])
      img3 = cv2.imread (p + '\\' + imgs[2])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg1 = cv2.vconcat ([img1, img2])
      myimg1 = cv2.vconcat ([myimg1, img3])

      img1 = cv2.imread (p + '\\' + imgs[3])
      img2 = cv2.imread (p + '\\' + imgs[4])
      img3 = cv2.imread (p + '\\' + imgs[5])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg2 = cv2.vconcat ([img1, img2])
      myimg2 = cv2.vconcat ([myimg2, img3])

      img1 = cv2.imread (p + '\\' + imgs[6])
      img2 = cv2.imread (p + '\\' + imgs[7])
      img3 = cv2.imread (p + '\\' + imgs[8])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg3 = cv2.vconcat ([img1, img2])
      myimg3 = cv2.vconcat ([myimg3, img3])

      myimg = cv2.hconcat ([myimg1, myimg2])
      myimg = cv2.hconcat ([myimg, myimg3])
      myimg = cv2.resize (myimg, (224,224))

      g = id2gender_train[item]

      cv2.imwrite (parent+'\\images_train\\' + g + '\\' + item + '.' + str(q) + '.jpeg', myimg)
      q += 1
  except:
  #  print (item, id2gender_train[item], img)
    continue

# In[]

exceptions = {'cbc0e7675ce123b7ca31f127dc7aeff5': [6, 7, 8, 9], 'c5845826bdf2537adc420aa9d665f4eb': [8, 9], 
               'a764d5a1a107bc1a76a106624880a553': [0], '8f28a9f29573b37ee9c7797fb26696ff': [3, 4], 
               '722b9e87ec239f5efe544593ad089ee8': [6, 7]}
for item in list(exceptions.keys()):
  p = trainpath + '\\en\\photo\\' + item
  imgs = listdir(p)
  for i in range(len(exceptions[item])):
    imgs.remove (item + '.' + str(exceptions[item][i])+ '.jpeg')
  for i in range(10):
    img1 = cv2.imread (p + '\\' + imgs[0])
    img2 = cv2.imread (p + '\\' + imgs[1])
    img3 = cv2.imread (p + '\\' + imgs[2])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg1 = cv2.vconcat ([img1, img2])
    myimg1 = cv2.vconcat ([myimg1, img3])
    
    img1 = cv2.imread (p + '\\' + imgs[3])
    img2 = cv2.imread (p + '\\' + imgs[4])
    img3 = cv2.imread (p + '\\' + imgs[5])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg2 = cv2.vconcat ([img1, img2])
    myimg2 = cv2.vconcat ([myimg2, img3])
    
    if len(imgs) > 6:
      img1 = cv2.imread (p + '\\' + imgs[6])
      img2 = cv2.imread (p + '\\' + imgs[7])
      img3 = cv2.imread (p + '\\' + imgs[-1])
    else:
      img1 = cv2.imread (p + '\\' + imgs[-1])
      img2 = cv2.imread (p + '\\' + imgs[-2])
      img3 = cv2.imread (p + '\\' + imgs[-3])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg3 = cv2.vconcat ([img1, img2])
    myimg3 = cv2.vconcat ([myimg3, img3])
    
    myimg = cv2.hconcat ([myimg1, myimg2])
    myimg = cv2.hconcat ([myimg, myimg3])
    myimg = cv2.resize (myimg, (224,224))
    g = id2gender_train[item]
    cv2.imwrite (parent+'\\images_train\\' + g + '\\' + item + '.' + str(i) + '.jpeg', myimg)
    if i == 7:
      dummy = imgs.pop (5)
      imgs.append (dummy)
    else:
      dummy = imgs.pop (0)
      imgs.append (dummy)

# In[]
with open(testpath + '\\en.txt') as f:
  lines = f.readlines()

# In[]
test_id2gender = {}
for item in lines:
  n = 0
  l = len (item)
  if item[-7] == 'f':
    v = 'F'
    n = -10
  else:
    v = 'M'
    n = -8
  k = item[:n]
  test_id2gender[k] = v
  
# In[]
os.mkdir(parent+'\\images_test')
os.mkdir(parent+'\\images_test\\F')
os.mkdir(parent+'\\images_test\\M')

# In[]
for item in list(test_id2gender.keys()):
  try:
    p = testpath + '\\en\\photo\\' + item
    cpy = [f for f in listdir(p)]
    q = 0
    for img in cpy:
        imgs = cpy.copy ()
        imgs.remove (img)
    #--------------------------------------------------------
        img1 = cv2.imread (p + '\\' + imgs[0])
        img2 = cv2.imread (p + '\\' + imgs[1])
        img3 = cv2.imread (p + '\\' + imgs[2])
        img1 = cv2.resize (img1, (224,224))
        img2 = cv2.resize (img2, (224,224))
        img3 = cv2.resize (img3, (224,224))
        myimg1 = cv2.vconcat ([img1, img2])
        myimg1 = cv2.vconcat ([myimg1, img3])
      
        img1 = cv2.imread (p + '\\' + imgs[3])
        img2 = cv2.imread (p + '\\' + imgs[4])
        img3 = cv2.imread (p + '\\' + imgs[5])
        img1 = cv2.resize (img1, (224,224))
        img2 = cv2.resize (img2, (224,224))
        img3 = cv2.resize (img3, (224,224))
        myimg2 = cv2.vconcat ([img1, img2])
        myimg2 = cv2.vconcat ([myimg2, img3])
      
        img1 = cv2.imread (p + '\\' + imgs[6])
        img2 = cv2.imread (p + '\\' + imgs[7])
        img3 = cv2.imread (p + '\\' + imgs[8])
        img1 = cv2.resize (img1, (224,224))
        img2 = cv2.resize (img2, (224,224))
        img3 = cv2.resize (img3, (224,224))
        myimg3 = cv2.vconcat ([img1, img2])
        myimg3 = cv2.vconcat ([myimg3, img3])
      
        myimg = cv2.hconcat ([myimg1, myimg2])
        myimg = cv2.hconcat ([myimg, myimg3])
        myimg = cv2.resize (myimg, (224,224))
        
        g = test_id2gender[item]
        
        cv2.imwrite (parent + '\\images_test\\' + g + '\\' + item + '.' + str(q) + '.jpeg', myimg)
        q += 1
  except:
    #print (item, test_id2gender[item], img)
    continue

# In[]

exceptions = {'3f4e19cc5d05d21ea1f6f33ce22cba07': [0], 'd5215f910f43b77372aba74bb7eb5424': [1], 
               '401ab80d3c0577e533904f2e0c174997': [6] }
for item in list(exceptions.keys()):
  p = testpath + '\\en\\photo\\' + item
  imgs = listdir(p)
  for i in range(len(exceptions[item])):
    imgs.remove (item + '.' + str(exceptions[item][i])+ '.jpeg')
  for i in range(10):
    img1 = cv2.imread (p + '\\' + imgs[0])
    img2 = cv2.imread (p + '\\' + imgs[1])
    img3 = cv2.imread (p + '\\' + imgs[2])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg1 = cv2.vconcat ([img1, img2])
    myimg1 = cv2.vconcat ([myimg1, img3])
    
    img1 = cv2.imread (p + '\\' + imgs[3])
    img2 = cv2.imread (p + '\\' + imgs[4])
    img3 = cv2.imread (p + '\\' + imgs[5])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg2 = cv2.vconcat ([img1, img2])
    myimg2 = cv2.vconcat ([myimg2, img3])
    
    img1 = cv2.imread (p + '\\' + imgs[6])
    img2 = cv2.imread (p + '\\' + imgs[7])
    img3 = cv2.imread (p + '\\' + imgs[-1])
    img1 = cv2.resize (img1, (224,224))
    img2 = cv2.resize (img2, (224,224))
    img3 = cv2.resize (img3, (224,224))
    myimg3 = cv2.vconcat ([img1, img2])
    myimg3 = cv2.vconcat ([myimg3, img3])
    
    myimg = cv2.hconcat ([myimg1, myimg2])
    myimg = cv2.hconcat ([myimg, myimg3])
    myimg = cv2.resize (myimg, (224,224))
    g = test_id2gender[item]
    cv2.imwrite (parent+'\\images_test\\' + g + '\\' + item + '.' + str(i) + '.jpeg', myimg)
    if i == 9:
      dummy = imgs.pop (5)
      imgs.append (dummy)
    else:
      dummy = imgs.pop (0)
      imgs.append (dummy)

