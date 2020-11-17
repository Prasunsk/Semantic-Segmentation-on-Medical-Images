# Semantic-Segmentation-on-Medical-Images
U-Net Architecture to identify sections of medical images(e.g. MRI Images etc) 

# Dataset and Challenge:
One of the first steps in the diagnosis of most
cardiac diseases, such as pulmonary hypertension, coronary
heart disease is the segmentation of ventricles from cardiac
magnetic resonance (MRI) images. Manual segmentation of the
right ventricle requires diligence and time, while its automated
segmentation is challenging due to shape variations and illdefined borders.
The Right Ventricle Segmentation Challenge (RVSC) was hosted in March 2012. It finally ended up with the 3D Cardiovascular Imaging: a MICCAI segmentation challenge. The dataset was collected from June 2008 to August 2008.
The training dataset contains only 200 to 280 cardiac magnetic resonance imaging (MRI) of sizes 256×216 pixels images of right ventricles of 16 patients, out of which 15/16 MRIs per patient are labelled. Thus, we have a training set of only 240 labelled images.
Labelled images are having both endocardial and epicardial contours. As these images are in .dcm format our first task is to convert it into readable format for our model. Also, this count is way too low to train a Convolutional Neural Network (CNN). Thus, we needed to perform data augmentation to increase the training samples.

The dataset can be found here- https://drive.google.com/drive/folders/1pEEuaRb5rvevFrMa4T6NCZBZ9IjRFF56 
# Architecture 
U-Net architecture was used so that the output mask has the exact shape as that of the input image.
![Capture](https://user-images.githubusercontent.com/63877316/99450610-f9c7c300-2946-11eb-9105-0a3e39eb6e1a.JPG)

# Pre-Processing Of Images:
There were 2 different size of images:
Some pictures has a dimension of 216x256 and some pictures has the dimension of 256x216.Thus, we have to rotate either of the images to get consistency in terms of the dimensions.
Moreover the image format is .dcm which needs to be converted to .jpg/.png

![P01-0108](https://user-images.githubusercontent.com/63877316/99451080-ae61e480-2947-11eb-90d1-9eae3679d41c.png)

Identifying the 2 contours shown in red and green lines is the primary task.
