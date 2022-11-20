#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
import PIL.Image as Image
import math
import numpy as np
import keras.backend as K
from keras.utils import plot_model
import json
from keras.losses import mse
import os
import glob
import random
import cv2
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import InceptionV3
import datetime
from keras.layers import Dense, Input, concatenate, Conv2D, MaxPooling2D, Flatten, Lambda
from keras.layers import Dense, Input, concatenate, Conv2D, MaxPooling2D, Flatten, Lambda, BatchNormalization, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.callbacks import TensorBoard
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.maximum(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True), K.epsilon()))


# # Model

# In[10]:


# feature extraction from input image
img = Input(shape = (112,112,2), name="input_image")

x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv1")(img)
x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv2")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    
x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv3")(x)
x = Conv2D(64, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv4")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
   
x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv5")(x)
x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv6")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    
x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv7")(x)
x = Conv2D(128, (3, 3), padding="same", strides=(1, 1), activation="relu", name="conv8")(x)
x = BatchNormalization()(x)
    
x = Flatten()(x)
x = Dropout(0.75, noise_shape=None, seed=None)(x)
x = Dense(1024, name='FC1')(x)
out = Dense(13, name='loss')(x)

# create model
model = Model(inputs=[img], outputs=[out])

# set output types
target = tf.compat.v1.placeholder(dtype='float32', shape=(13,1)) 

# get model summary
model.summary()

# compile model
model.compile(optimizer=Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=euclidean_distance)
#plot_model(model, to_file='model.png')


# # Get data

# xCam = (self.intrinsic.fx * self.extrinsic.baseline) / disparity
#         
# yCam = - (xCam / self.intrinsic.fx) * (u - self.intrinsic.u0)
#         
# zCam = (xCam / self.intrinsic.fy) * (self.intrinsic.v0 - v)
# 
# yWorld = yCam + self.extrinsic.y
# 
# xWorld = xCam * math.cos(self.extrinsic.pitch) + zCam * \
#             math.sin(self.extrinsic.pitch) + self.extrinsic.x
#         
# zWorld = - xCam * math.sin(self.extrinsic.pitch) + zCam * \
#             math.cos(self.extrinsic.pitch) + self.extrinsic.z
# 
# Focal_length = ImageSizeX /(2 * tan(CameraFOV * π / 360))
# 
# Center_X = ImageSizeX / 2
# 
# Center_Y = ImageSizeY / 2

# ## Town 1

# In[14]:


data_path = "/home/talha/Documents/Camera-Calibration-Carla/Experiments/Town2/"

new_size = (112, 112) 

X = []
Y = []

for folder in os.listdir(data_path):
    
    if "episode" in folder:
    
        episode_data = os.listdir(data_path+folder)
        
        file = open(data_path+folder+"/params.txt", "r")

        i = 0

        for line in file:
    
            if ":" in line:
        
                if i == 0:
        
                    fov = int(line.split(":")[1])
            
                elif i == 1:
            
                    x = int(line.split(":")[1])
            
                elif i == 2:
            
                    y = int(line.split(":")[1])
            
                elif i == 3:
            
                    z = int(line.split(":")[1])
            
                elif i == 4:
            
                    p = int(line.split(":")[1])
            
                elif i == 5:
            
                    yaw = int(line.split(":")[1])
            
                elif i == 6:
            
                    roll = int(line.split(":")[1])
            
            i += 1
        
        
        for fname in episode_data:
        
            if "Left" in fname:
                
                l_im = cv2.imread(data_path+folder+"/"+fname, 0)
                r_im = cv2.imread(data_path+folder+"/"+'RightRGB_'+fname.split("_")[1], 0)
                training_image = np.dstack((l_im, r_im))
                stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                disparity_map = stereo.compute(l_im, r_im)
                disparity_value = np.mean(disparity_map)
        
               
        
                X.append(training_image)
                
                
               
                
                focal_length = new_size[0]/(2*np.tan(fov*np.pi/360))
                center_x = new_size[0]/2
                center_y = center_x
                
                
                xCam = (focal_length * x) / disparity_value
                yCam = - (xCam / focal_length) * (5 - center_x)
                zCam = (xCam / focal_length) * (center_y - 5)

                yWorld = yCam + y
                xWorld = xCam * math.cos(p) + zCam*math.sin(p) + x
                zWorld = - xCam * math.sin(p) + zCam*math.cos(p) + z
        
                
                Y.append([focal_length, focal_length, center_x, center_y, x, disparity_value, x, y, z, p, xWorld, yWorld, zWorld])
        



# ## Town 2

# In[15]:


data_path = "/home/talha/Documents/Camera-Calibration-Carla/Experiments/Town1/"

new_size = (112, 112) 


for folder in os.listdir(data_path):
    
    if "episode" in folder:
    
        episode_data = os.listdir(data_path+folder)
        
        file = open(data_path+folder+"/params.txt", "r")

        i = 0

        for line in file:
    
            if ":" in line:
        
                if i == 0:
        
                    fov = int(line.split(":")[1])
            
                elif i == 1:
            
                    x = int(line.split(":")[1])
            
                elif i == 2:
            
                    y = int(line.split(":")[1])
            
                elif i == 3:
            
                    z = int(line.split(":")[1])
            
                elif i == 4:
            
                    p = int(line.split(":")[1])
            
                elif i == 5:
            
                    yaw = int(line.split(":")[1])
            
                elif i == 6:
            
                    roll = int(line.split(":")[1])
            
            i += 1
        
        
        for fname in episode_data:
        
            if "Left" in fname:
                
                l_im = cv2.imread(data_path+folder+"/"+fname, 0)
                r_im = cv2.imread(data_path+folder+"/"+'RightRGB_'+fname.split("_")[1], 0)
                training_image = np.dstack((l_im, r_im))
                stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
                disparity_map = stereo.compute(l_im, r_im)
                disparity_value = np.mean(disparity_map)
        
               
        
                X.append(training_image)
                
                
               
                
                focal_length = new_size[0]/(2*np.tan(fov*np.pi/360))
                center_x = new_size[0]/2
                center_y = center_x
                
                
                xCam = (focal_length * x) / disparity_value
                yCam = - (xCam / focal_length) * (5 - center_x)
                zCam = (xCam / focal_length) * (center_y - 5)

                yWorld = yCam + y
                xWorld = xCam * math.cos(p) + zCam*math.sin(p) + x
                zWorld = - xCam * math.sin(p) + zCam*math.cos(p) + z
        
                
                Y.append([focal_length, focal_length, center_x, center_y, x, disparity_value, x, y, z, p, xWorld, yWorld, zWorld])
        



# In[16]:


print ("dataset: ",np.shape(X))


# In[17]:


print ("Training Dataset: ",len(X)*0.7, "Test Dataset: ", len(Y)*0.3)






# In[ ]:


avg_fov_train = 0


k = 0


for i  in range(len(Y[:int(len(Y)*0.7)])):
    
    avg_fov_train += 2*np.arctan(112/(2*Y[i][0]))
    
    k += 1


avg_fov_train /= len(Y[:int(len(Y)*0.7)])

print ("avg. fov train: ",avg_fov_train)


# In[ ]:


th_0 = 0
th_1 = 0
th_2 = 0
th_3 = 0
th_4 = 0
th_5 = 0

percent_correct = []

for i  in range(len(Y[int(len(Y)*0.7):])):
    
    predicted_fov = avg_fov_train
    actual_fov = 2*np.arctan(112/(2*Y[k][0]))
    
    if abs(predicted_fov - actual_fov) <= 0:
        
        th_0 += 1 
        
    if abs(predicted_fov - actual_fov) <= 1:
        
        th_1 += 1
        
    if abs(predicted_fov - actual_fov) <= 2:
        
        th_2 += 1
        
    if abs(predicted_fov - actual_fov) <= 3:
        
        th_3 += 1
        
    if abs(predicted_fov - actual_fov) <= 4:
        
        th_4 += 1
        
    if abs(predicted_fov - actual_fov) <= 5:
        
        th_5 += 1

    k += 1

percent_correct.append(th_0/len(Y[int(len(Y)*0.7):])*100)
percent_correct.append(th_1/len(Y[int(len(Y)*0.7):])*100)
percent_correct.append(th_2/len(Y[int(len(Y)*0.7):])*100)
percent_correct.append(th_3/len(Y[int(len(Y)*0.7):])*100)
percent_correct.append(th_4/len(Y[int(len(Y)*0.7):])*100)
percent_correct.append(th_5/len(Y[int(len(Y)*0.7):])*100)



# In[ ]:


#th_0, th_1, th_2, th_3, th_4, th_5


# In[ ]:


print (th_0/len(Y[int(len(Y)*0.7):]), th_1/len(Y[int(len(Y)*0.7):]), th_2/len(Y[int(len(Y)*0.7):]), th_3/len(Y[int(len(Y)*0.7):]), th_4/len(Y[int(len(Y)*0.7):]), th_5/len(Y[int(len(Y)*0.7):]))

