# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:37:25 2020

@author: darshu
"""

#import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np
import pandas as pd
import os, glob
import cv2
from keras.preprocessing import image

def maybe_rotate(image):
    # orient image in landscape
    height, width = image.shape
    return np.rot90(image) if width > height else image
    return image


class data_load(object):
    def __init__(self, mypath,datagen):
        self.mypath = mypath
        self.datagen = datagen


    def save_fig(self,xx,xy_inner,xy_outer,item):
        df1 = np.loadtxt(xy_inner[0])
        df2 = np.loadtxt(xy_outer[0])
        ds = dicom.dcmread(xx[0])
        img = maybe_rotate(ds.pixel_array)
        self.image_height, self.image_width = img.shape
        self.rotated = (ds.pixel_array.shape != img.shape)
        img = img[:,4:-4]
        df1=pd.DataFrame(df1,columns=["X","Y"])
        df2=pd.DataFrame(df2,columns=["X","Y"])
        df1["x_new"]=''
        df1["y_new"]=''
        df2["x_new"]=''
        df2["y_new"]=''
        for i in range(len(df1)):
            df1.loc[i,"x_new"]=round(df1.loc[i,"X"])
            df1.loc[i,"y_new"]=round(df1.loc[i,"Y"])
    
        for i in range(len(df2)):
            df2.loc[i,"x_new"]=round(df2.loc[i,"X"])
            df2.loc[i,"y_new"]=round(df2.loc[i,"Y"])
        a=np.zeros((256,208))
        for k in range(len(df1)):
            x=int(df1.x_new[k])
            y=int(df1.y_new[k])
            if self.rotated:
                x, y = y, self.image_height - x
            a[y][x]=255

        b=np.zeros((256,208))
        for k in range(len(df2)):
            x=int(df2.x_new[k])
            y=int(df2.y_new[k])
            if self.rotated:
                 x, y = y, self.image_height - x
            b[y][x]=255
        c=a+b
        
        #save_name = item.split('.')[0] + '-contour.jpg'
        
        cv2.imwrite('temp.jpg', c)
        #save = item.split('.')[0] + '-.jpg'
        cv2.imwrite('temp_dcm.jpg', img)
        img_1 = image.load_img('temp.jpg')  # this is a PIL image
        x = image.img_to_array(img_1)         
        x = x.reshape((1,) + x.shape)  
        
        img_2 = image.load_img('temp_dcm.jpg')  # this is a PIL image
        y = image.img_to_array(img_2)  
        y = y.reshape((1,) + y.shape)  

        i = 0
        for batch in self.datagen.flow(x, batch_size=1,
                          save_to_dir='Augmented_images_contour', save_prefix=item+'augmented', save_format='jpeg',seed=15):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely
            
        i = 0
        for batch in self.datagen.flow(y, batch_size=1,
                          save_to_dir='Augmented_images', save_prefix=item+'augmented', save_format='jpeg',seed=15):
            i += 1
            if i > 20:
                break 
        
        
        
        
        
        
        
        

    def process(self):
        glob_search = os.path.join(self.mypath, "patient*")
        files = glob.glob(glob_search)
        for file in files:
            list_file = os.path.join(file, "P*list.txt")
            list_file = glob.glob(list_file)
            f= open(list_file[0])
            content = f.readlines()
            content=np.array(content,dtype='str')
            for i in range (len(content)):
                content[i]=content[i].split('\\')[-1]
                content[i]=content[i][:8]
    
            dcm_folder = os.path.join(file, "P*dicom")
            dcm_folder = glob.glob(dcm_folder)
            contour_folder = os.path.join(file, "P*contours-manual")
            contour_folder = glob.glob(contour_folder)
            for item in content:
                xx = os.path.join(dcm_folder[0],item+"*.dcm")
                xx = glob.glob(xx)    
                xy_inner = os.path.join(contour_folder[0],item+"*icontour-manual.txt")
                xy_outer = os.path.join(contour_folder[0],item+"*ocontour-manual.txt")
                
                xy_inner = glob.glob(xy_inner)
                xy_outer = glob.glob(xy_outer)
                self.save_fig(xx,xy_inner,xy_outer,item)
            #print(xx)
                



        
        