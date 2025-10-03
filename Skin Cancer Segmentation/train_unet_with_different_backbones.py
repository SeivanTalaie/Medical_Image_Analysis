############################### Import libraries ###############################

import keras 
import matplotlib.pyplot as plt 
import numpy as np 
import segmentation_models as sm 
import livelossplot as llp 
from keras.utils import normalize 
from sklearn.model_selection import train_test_split 
import os 
import random 
import glob 
import cv2 as cv 
from tqdm import tqdm 
from PIL import Image
import pandas as pd
from batch_loader_for_backbones import image_loader_backbone
from keras.models import load_model
from unet_model import unet_model
from keras_unet.models import custom_unet



################ Getting the data ready for different backbones ################
# Backbones are include [Efficientnetb0, VGG16, ResNet34, MobileNet]

backbone="mobilenet"
process_input=sm.get_preprocessing(backbone)

train_image_dir="C:/Datasets/Skin cancer segmentation/input_data/train/image"
train_mask_dir="C:/Datasets/Skin cancer segmentation/input_data/train/mask" 

val_image_dir="C:/Datasets/Skin cancer segmentation/input_data/val/image"
val_mask_dir="C:/Datasets/Skin cancer segmentation/input_data/val/mask"

training_image_list=os.listdir(train_image_dir)
training_mask_list=os.listdir(train_mask_dir)

validation_image_list=os.listdir(val_image_dir)
validation_mask_list=os.listdir(val_mask_dir)

batch_size=32

training_image=image_loader_backbone(train_image_dir , training_image_list ,
                                 train_mask_dir  , training_mask_list  , batch_size ,process_input)

validation_image=image_loader_backbone(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size, process_input)


######## Sanity Check ########

img,mask=training_image.__next__()

print(img.shape   ,  mask.shape)

random_img=np.random.randint(0,batch_size -1)
test_img=img[random_img]
test_mask=mask[random_img]

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(test_img)
plt.title("image")
plt.subplot(1,2,2)
plt.imshow(test_mask)
plt.title("mask")
plt.show()


###################### Define metric, loss and optimizer #######################

metrics=[sm.metrics.IOUScore(threshold=0.5), "accuracy"]

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25 , 0.25 , 0.25 , 0.25]))
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1*focal_loss)

LR=0.001
optimizer=keras.optimizers.Adam(LR)


############################## Define callbacks ################################

plot_result = llp.PlotLossesKeras()

save_path="F:/Master/4th semester/Thesis/Py code/Skin cancer segmentation/saved_models/Keras_Unet/"

ModelChechPoint=keras.callbacks.ModelCheckpoint(save_path + "SkinCancerKeras_Unet__{epoch}__{val_iou_score}.hdf5",
                                                monitor="val_iou_score", 
                                                verbose=1, save_best_only=True, mode="max")


###################### Train unet with different backbones #####################

steps_per_epochs=len(training_image_list)//batch_size
val_steps_per_epochss=len(validation_image_list)//batch_size

model=sm.Unet(backbone, input_shape=(128, 128, 3) , classes=1 ,
              encoder_weights="imagenet", activation="sigmoid")

print(model.input_shape)
print(model.output_shape)


model.summary()

model.compile(optimizer=optimizer, metrics=metrics, loss=total_loss)

history=model.fit(training_image, validation_data=validation_image, steps_per_epoch=steps_per_epochs,
                  validation_steps=val_steps_per_epochss, epochs=40, verbose=1, callbacks=[ModelChechPoint])
                  
model.save("mobilenet.hdf5")

pd.DataFrame(history.history).plot()
plt.show()

###################### Retraining the model (if necessary) ######################

model_path="F:/Python/CNN/U-Net/skin cancer saved models/full data/"

my_model=load_model(model_path + "mobilenet.hdf5", 
                    custom_objects={"dice_loss_plus_1binary_focal_loss" : total_loss, 
                                    "iou_score" : sm.metrics.IOUScore(threshold=0.5)})


history=my_model.fit(training_image, validation_data=validation_image, steps_per_epoch=steps_per_epochs,
                  validation_steps=val_steps_per_epochss, epochs=40, verbose=1, callbacks=[ModelChechPoint])
                  

pd.DataFrame(history.history).plot()
plt.show()
