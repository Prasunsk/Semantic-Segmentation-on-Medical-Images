# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:43:06 2020

@author: darshu
"""

import cv2
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

def load_images_from_folder(path):
    X_path = os.path.join(path+'\Augmented_images_contour')
    Y_path = os.path.join(path+'\Augmented_images')
    X_images = []
    Y_images = []

    for filename in os.listdir(X_path):
        img = cv2.imread(os.path.join(X_path,filename),0)
        #plt.imshow(img)
        if img is not None:
            X_images.append(img)
        for filename_y in os.listdir(Y_path):
            if filename == filename_y:
                img = cv2.imread(os.path.join(Y_path,filename_y),0)
                Y_images.append(img)
    X_images =np.array(X_images)
    Y_images =np.array(Y_images)
    return X_images,Y_images




def Split_Train_Val(X_images,Y_images):
    val_data_size = int(20*len(X_images)/100)

    valid_X = X_images[:val_data_size]
    valid_Y = Y_images[:val_data_size]

    train_X = X_images[val_data_size:]
    train_Y = Y_images[val_data_size:]
    
    return valid_X,valid_Y,train_X,train_Y


def U_net(input_shape = (256,208,1)):
    input_img = Input(shape=input_shape)
    filters = [64,128,256,512,1024]
    
    
    step1 = Conv2D(filters[0], (3, 3), activation='relu', input_shape= input_shape, strides = (1, 1),padding='same')(input_img)
    step1 = Conv2D(filters[0], (3, 3), activation='relu',strides = (1, 1), padding='same')(step1) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(step1)
    

    step2 = Conv2D(filters[1], (3, 3), activation='relu', strides = (1, 1), padding='same')(pool1)
    step2 = Conv2D(filters[1], (3, 3), activation='relu', strides = (1, 1), padding='same')(step2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(step2)
    
    
    step3 = Conv2D(filters[2], (3, 3), activation='relu', strides = (1, 1), padding='same')(pool2)
    step3 = Conv2D(filters[2], (3, 3), activation='relu', strides = (1, 1), padding='same')(step3) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(step3)
    
    
    step4 = Conv2D(filters[3], (3, 3), activation='relu', strides = (1, 1), padding='same')(pool3)
    step4 = Conv2D(filters[3], (3, 3), activation='relu', strides = (1, 1), padding='same')(step4) 
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(step4)
    
    step5 = Conv2D(filters[4], (3, 3), activation='relu', strides = (1, 1), padding='same')(pool4)
    step5 = Conv2D(filters[4], (3, 3), activation='relu', strides = (1, 1), padding='same')(step5) 
    
    us = keras.layers.UpSampling2D((2, 2))(step5)
    skip = keras.layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same')(step5)
    concat = keras.layers.Concatenate()([us, skip])
   
    step6 = Conv2D(filters[3], (3, 3), activation='relu', strides = (1, 1), padding='same')(concat)
    step6 = Conv2D(filters[3], (3, 3), activation='relu', strides = (1, 1), padding='same')(step6) 

    us = keras.layers.UpSampling2D((2, 2))(step6)
    skip = keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(step6)
    concat = keras.layers.Concatenate()([us, skip])
    
    step7 = Conv2D(filters[2], (3, 3), activation='relu', strides = (1, 1), padding='same')(concat)
    step7 = Conv2D(filters[2], (3, 3), activation='relu', strides = (1, 1), padding='same')(step7) 

    us = keras.layers.UpSampling2D((2, 2))(step7)
    skip = keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(step7)
    concat = keras.layers.Concatenate()([us, skip])
  
    step8 = Conv2D(filters[1], (3, 3), activation='relu', strides = (1, 1), padding='same')(concat)
    step8 = Conv2D(filters[1], (3, 3), activation='relu', strides = (1, 1), padding='same')(step8) 

    us = keras.layers.UpSampling2D((2, 2))(step8)
    skip = keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(step8)
    concat = keras.layers.Concatenate()([us, skip])
    
    step9 = Conv2D(filters[0], (3, 3), activation='relu', strides = (1, 1), padding='same')(concat)
    step9 = Conv2D(filters[0], (3, 3), activation='relu', strides = (1, 1), padding='same')(step9) 

    step10 = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(step9) 
    
    last_step = keras.models.Model(input_img, step10)
    return last_step


path=r"C:\Users\PRATT\Desktop\5th Sem Project & Assignments\EE782_AML\Solution\Files"

X_images,Y_images=load_images_from_folder(path)

X_images = X_images.reshape(X_images.shape[0], 256, 208, 1)
Y_images = Y_images.reshape(Y_images.shape[0], 256, 208, 1)
X_images = X_images.astype('float32')
Y_images = Y_images.astype('float32')
X_images /= 255
Y_images /= 255

valid_X,valid_Y,train_X,train_Y=Split_Train_Val(X_images,Y_images)

epochs=20
batch_size=32
patience = 5 

# train_X = train_X.reshape(X_train.shape[0], 256, 208, 1)
# X_test = test.reshape(X_test.shape[0], 256, 208, 1)


checkpoint_filepath = '\tmp\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_auc',
    mode='max',
    save_best_only=True)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="auc", min_delta=0.001, patience=patience)
# input_img = Input((256, 208, 1), name='img')
# model = U_net(input_img)
model = U_net()
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
history = model.fit(train_X,
                        train_Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(valid_X, valid_Y),
                        callbacks=[model_checkpoint_callback]
                        )