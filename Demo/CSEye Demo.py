# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:37:14 2018

@author: ameer
"""

import os
import cv2 as cv
import numpy as np
import random

import keras
from keras.applications import vgg19
from keras.models import Model

import matplotlib.pyplot as plt

#%%

##Create the sample data input. The suspect image is inserted as the first
##element of the pics array. 

suspect = ['Sarah_Jessica_Parker_0005']


candidates = ['Alastair_Campbell_0001',
              #'Audrey_Lacroix_0001',
              'Benazir_Bhutto_0003',
              'Carrie-Anne_Moss_0002',
              'Lisa_Ling_0001',
              #'Sarah_Jessica_Parker_0006',
              'Rosario_Dawson_0001',
              'Nancy_Pelosi_0003',
              'Phil_Bredesen_0001', 'Sarah_Jessica_Parker_0006','Audrey_Lacroix_0001',
              'Naji_Sabri_0004']

##Rearrange the images to test the model in different arrangements
##random.shuffle(candidates)

pics = np.zeros((1,11,100,100,3), dtype = np.uint8)

for i in range(11):
    if i == 0:
        pic = './' + suspect[i] + '.jpg'
    else:
        pic = './' + candidates[i-1] + '.jpg'
    print(pic)
    old = cv.imread(pic,1)
    new = None
    new = cv.resize(old, new, fx=100/(old.shape[0]), fy=100/(old.shape[1]))
    pics[0,i,:,:,:] = new.reshape((1,100,100,3))
    

#%%

##Step 1: Extract the features of each of the 11 images using a truncated VGG19
##network. 

custom_vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3))
model = Model(inputs=custom_vgg.input, outputs=custom_vgg.get_layer('block5_conv4').output)

##The network produces 512 features, each represented by a 6x6 matrix
temp = model.predict(pics[0,:,:,:,:])
features = temp.reshape((1,11,6,6,512))

#%%

##Step 2: Compare the extracted features of the suspect to each candidate. This
##step implements the angle, dot product, and element-wise distance measures.

##First, define functions that perform the comparisons

# If we think of each feature map encoding (6x6 matrix) as consisting of 36 
# independent values, we can vectorise the matrix and compute the angle between
# the features of the suspect and that of the matrix. A lower angle will imply 
# similarity between the suspect and candidate.
def angle(sample):
  x = np.reshape(sample,(11,36,512))
  suspect = x[0,:,:]
  candidates = x[1:,:,:]
  new = np.zeros((10,512))
  for i in range(0,10):
    c = candidates[i,:,:]
    for j in range(0,512):
      mag_s = np.linalg.norm(suspect[:,j])
      mag_c = np.linalg.norm(c[:,j])
      if (mag_s == 0) and (mag_c == 0):
          new[i,j] = 0
      elif (mag_s == 0) or (mag_c == 0):
          new[i,j] = np.arccos(0)
      else:
          temp = np.dot(suspect[:,j],c[:,j])/(mag_s * mag_c)
          if (temp > 1):
              temp = 1
          if (temp < -1):
              temp = -1
          new[i,j] = np.arccos(temp)
  return new

# As a second distance measure, we can consider the Dot Product between
# the suspect and each candidate. The geometric interpretation is similar to the 
# angle approach except that in this case we consider the dot product instead of 
# the angle between vectors
def dotprod(sample):
  x = np.reshape(sample,(11,36,512))
  suspect = x[0,:,:]
  candidates = x[1:,:,:]
  new = np.zeros((10,512))
  for i in range(0,10):
    c = candidates[i,:,:]
    for j in range(0,512):
      new[i,j] = np.dot(suspect[:,j],c[:,j])
  return new

# If we think about the features maps almost as heatmaps for features, we can consider the 
# sum of the element wise distances as a measure of difference between two sets of feature maps.
def distance(sample):
  suspect = sample[0,:,:,:]
  candidates = sample[1:,:,:,:]
  for i in range(0,10):
    candidates[i,:,:,:] = candidates[i,:,:,:] - suspect[:,:,:]
  new = np.zeros((10,512))
  for i in range(0,10):
    for j in range(0,512):
      new[i,j] = np.sum(candidates[i,:,:,j])
  return new

# Comparison applies the specified comparison method between the suspect images and the
# candidate. op should be restricted to the set of operations specified. At this time,
# these include 'angle', 'dotprod', and 'distance'. All of these are defined in the above chunk.
# The input, sample is of size (n,11,6,6,512) and the output will be of size (n,10,512)
def comparison(sample, op):
  n = sample.shape[0]
  new = np.zeros((n,10,512))
  for i in range(0,n):
    new[i,:,:] = op(sample[i,:,:,:,:])
    if (i % 10) == 0:
        print(i/10)
  return new

##Perform the comparison for our specific test case. Note that the angle
##measure is the most reliable
compared = comparison(features,angle)

#%%

##Step 3: Use the encoded feature comparisons to determine the candidate that 
##best matches the suspect. This model has been pretrained for use in this 
##demo.

##Trained angle model
CSEye = keras.models.load_model('./CSEye.h5')

##Separate the input data into each image. To faciliate the simulataneous
##processing of each image and the weight sharing scheme, the images enter
##the model separately.

#%%

##Note: Make sure the correct compared features set is used to match the model.
results = CSEye.predict(x=[compared[:,0,:],compared[:,1,:],compared[:,2,:],
                            compared[:,3,:],compared[:,4,:],compared[:,5,:],
                            compared[:,5,:],compared[:,7,:],compared[:,8,:],
                            compared[:,8,:]])
    

plt.bar(x=candidates, height=results.reshape((10,)), align='center', alpha=0.5)
plt.title('Results')
plt.ylabel('Probability of Match')
plt.xlabel('Candidate')
plt.show()
    
img1 = cv.imread('./' + suspect[0] + '.jpg',1)
img2 = cv.imread('./' + candidates[np.argmax(results)] + '.jpg',1)
cv.imshow('Suspect',img1)
cv.imshow('Match',img2)
cv.moveWindow('Match', 300, 0) 
cv.waitKey(0)
cv.destroyAllWindows()












