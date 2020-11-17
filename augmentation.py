# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:47:27 2020

@author: darshu

"""
import load_data as ls
from keras.preprocessing.image import ImageDataGenerator


mypath  = r"C:\Users\PRATT\Desktop\5th Sem Project & Assignments\EE782_AML\Dataset\Extracted\TrainingSet\TrainingSet"
datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

load = ls.data_load(mypath,datagen)
load.process()

# earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

#     # train model
# history = model.fit(X_train,
#                         y_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         validation_data=(X_validation, y_validation),
#                         callbacks=[earlystop_callback])
