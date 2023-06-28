import os
from os import listdir
import cv2
import pandas as pd
import numpy as np
from PIL import Image as img

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))
pan = parent + '/PAN18'

with open(pan + '/train/en_train.txt') as f:
  lines = f.readlines()

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

for item in list(id2gender_train.keys()):
  try:
    p = path + '/train/' + item
    cpy = [f for f in listdir(p)]
    q = 0
    for img in cpy:
      imgs = cpy.copy ()
      imgs.remove (img)
  #--------------------------------------------------------
      img1 = cv2.imread (p + '/' + imgs[0])
      img2 = cv2.imread (p + '/' + imgs[1])
      img3 = cv2.imread (p + '/' + imgs[2])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg1 = cv2.vconcat ([img1, img2])
      myimg1 = cv2.vconcat ([myimg1, img3])

      img1 = cv2.imread (p + '/' + imgs[3])
      img2 = cv2.imread (p + '/' + imgs[4])
      img3 = cv2.imread (p + '/' + imgs[5])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg2 = cv2.vconcat ([img1, img2])
      myimg2 = cv2.vconcat ([myimg2, img3])

      img1 = cv2.imread (p + '/' + imgs[6])
      img2 = cv2.imread (p + '/' + imgs[7])
      img3 = cv2.imread (p + '/' + imgs[8])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg3 = cv2.vconcat ([img1, img2])
      myimg3 = cv2.vconcat ([myimg3, img3])

      myimg = cv2.hconcat ([myimg1, myimg2])
      myimg = cv2.hconcat ([myimg, myimg3])
      myimg = cv2.resize (myimg, (224,224))

      g = id2gender_train[item]

      cv2.imwrite (path + '/train/' + g + '/' + item + '.' + str(q) + '.jpeg', myimg)
      q += 1
    except:
      continue




with open(pan + '/test/en_test.txt') as f:
  lines = f.readlines()

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

for item in list(test_id2gender.keys()):
    if item in listdir(path):
      p = path + '/test/' + item
      cpy = [f for f in listdir(p)]
      q = 0
      for img in cpy:
      imgs = cpy.copy ()
      imgs.remove (img)
  #--------------------------------------------------------
      img1 = cv2.imread (p + '/' + imgs[0])
      img2 = cv2.imread (p + '/' + imgs[1])
      img3 = cv2.imread (p + '/' + imgs[2])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg1 = cv2.vconcat ([img1, img2])
      myimg1 = cv2.vconcat ([myimg1, img3])

      img1 = cv2.imread (p + '/' + imgs[3])
      img2 = cv2.imread (p + '/' + imgs[4])
      img3 = cv2.imread (p + '/' + imgs[5])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg2 = cv2.vconcat ([img1, img2])
      myimg2 = cv2.vconcat ([myimg2, img3])

      img1 = cv2.imread (p + '/' + imgs[6])
      img2 = cv2.imread (p + '/' + imgs[7])
      img3 = cv2.imread (p + '/' + imgs[8])
      img1 = cv2.resize (img1, (224,224))
      img2 = cv2.resize (img2, (224,224))
      img3 = cv2.resize (img3, (224,224))
      myimg3 = cv2.vconcat ([img1, img2])
      myimg3 = cv2.vconcat ([myimg3, img3])

      myimg = cv2.hconcat ([myimg1, myimg2])
      myimg = cv2.hconcat ([myimg, myimg3])
      myimg = cv2.resize (myimg, (224,224))
      
      g = test_id2gender[item]
      
      cv2.imwrite (path + '/test/' + g + '/' + item + '.' + str(q) + '.jpeg', myimg)
      q += 1





