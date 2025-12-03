
import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib 
from sklearn.preprocessing import MinMaxScaler 
import glob 
from keras.utils import to_categorical 
import random
from tqdm import tqdm
from patchify import patchify ,unpatchify
from keras_unet.models import custom_unet
import segmentation_models as sm
import keras 
import pandas as pd
from keras.models import load_model
from keras.metrics import MeanIoU
import livelossplot as llp
from tifffile import imsave , imread
import splitfolders  
import tensorflow as tf
import os
import keras.backend as K


######################################### Main Code #########################################


################ image loader ################


def load_image(img_dir , img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)

def image_loader(img_dir , img_list , mask_dir , mask_list , batch_size , preprocessing):

    L=len(img_list)

    while True:
        batch_start = 0 
        batch_end = batch_size

        while batch_start < L:
            limit=min( batch_end , L)

            X=load_image(img_dir , img_list[batch_start:limit])
            X=preprocessing(X)
            Y=load_image(mask_dir , mask_list[batch_start:limit])
            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size
            
backbone1="resnet50"    
process_input1=sm.get_preprocessing(backbone1)       

image_dir="H:/Datasets/Brats2020/Input_data_npy/train/image"
mask_dir="H:/Datasets/Brats2020/Input_data_npy/train/mask"
image_list=os.listdir(image_dir)
mask_list=os.listdir(mask_dir)

batch_size=128
training_data1=image_loader(image_dir,image_list , mask_dir , mask_list , batch_size , process_input1)


img,mask=training_data1.__next__()

random_img=np.random.randint(0,img.shape[0]-1)
test_img=img[random_img]
test_mask=mask[random_img]
test_mask=np.argmax(test_mask , axis=2)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(test_img , cmap="gray")
plt.title("image")
plt.subplot(1,2,2)
plt.imshow(test_mask, cmap="gray")
plt.title("mask")
plt.show()




################################### define train,val,test data ##################################

backbone="resnet50"    
process_input=sm.get_preprocessing(backbone)    

training_image_dir="H:/Datasets/Brats2020/Input_data_npy/train/image"
training_mask_dir="H:/Datasets/Brats2020/Input_data_npy/train/mask"
training_image_list=os.listdir(training_image_dir)
training_mask_list=os.listdir(training_mask_dir)

validation_image_dir="H:/Datasets/Brats2020/Input_data_npy/val/image"
validation_mask_dir="H:/Datasets/Brats2020/Input_data_npy/val/mask"
validation_image_list=os.listdir(validation_image_dir)
validation_mask_list=os.listdir(validation_mask_dir)

test_image_dir="H:/Datasets/Brats2020/Input_data_npy/test/image"
test_mask_dir="H:/Datasets/Brats2020/Input_data_npy/test/mask"
test_image_list=os.listdir(test_image_dir)
test_mask_list=os.listdir(test_mask_dir)

batch_size=32

training_data=image_loader(training_image_dir,training_image_list ,
                           training_mask_dir , training_mask_list , batch_size , process_input)

validation_data=image_loader(validation_image_dir,validation_image_list ,
                             validation_mask_dir , validation_mask_list , batch_size , process_input)

test_data=image_loader(test_image_dir,test_image_list ,
                             test_mask_dir , test_mask_list , batch_size , process_input)


training_img,training_mask=training_data.__next__()
validation_img,validation_mask=validation_data.__next__()
test_img,test_mask=test_data.__next__()


random_img=np.random.randint(0,training_img.shape[0]-1)
training_test_img=training_img[random_img]
training_test_mask=training_mask[random_img]
training_test_mask=np.argmax(training_test_mask , axis=2)

random_img1=np.random.randint(0,validation_img.shape[0]-1)
validation_test_img=validation_img[random_img1]
validation_test_mask=validation_mask[random_img1]
validation_test_mask=np.argmax(validation_test_mask , axis=2)


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(training_test_img , cmap="gray")
plt.title("training_image")
plt.subplot(2,2,2)
plt.imshow(training_test_mask, cmap="gray")
plt.title("training_mask")
plt.subplot(2,2,3)
plt.imshow(validation_test_img, cmap="gray")
plt.title("validation_image")
plt.subplot(2,2,4)
plt.imshow(validation_test_mask, cmap="gray")
plt.title("validation_mask")
plt.show()






################################### define metrics and losses ###################################

metrics=[sm.metrics.IOUScore(threshold=0.5)]

dice_loss = sm.losses.DiceLoss()
# focal_loss = sm.losses.CategoricalFocalLoss()
# total_loss = dice_loss + (1*focal_loss)

LR=0.0001
optimizer=keras.optimizers.Adam(LR)

########################################### callbacks ###########################################


save_path="H:/former_method/resnet50_without_classWeights/resnet50_weights_1/"

ModelChechPoint=keras.callbacks.ModelCheckpoint(save_path + "Resnet50_{epoch}__{val_iou_score:.4f}__{val_loss:.4f}.hdf5" ,
                                                monitor="val_iou_score" , 
                                                verbose=1 , save_best_only=True , mode="max" )

kerasPlot=llp.PlotLossesKeras()

######################################### define model #########################################

steps_per_epochs=len(training_image_list)//batch_size
val_steps_per_epochss=len(validation_image_list)//batch_size


model=sm.Unet(backbone , input_shape=(128,128,3) , classes=4 , activation="softmax" ,
              encoder_weights="imagenet")


print(model.input_shape)
print(model.output_shape)
model.summary()


model.compile(optimizer = optimizer , metrics = metrics , loss = dice_loss)


history=model.fit(training_data , validation_data=validation_data ,epochs=50 , verbose=1,
                  callbacks=[ModelChechPoint,kerasPlot] , steps_per_epoch=steps_per_epochs , 
                  validation_steps=val_steps_per_epochss )
                 

# model.save("F:/thesis project new/new theory/saved_models/Inceptionv3/Inceptionv3_Backbone_final70.hdf5")

pd.DataFrame(history.history).plot()
plt.show()


plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


######################################### retrain the  model #########################################

model_path="F:/thesis project new/new theory\saved_models/normal_unet_5layer/"
my_model=load_model(model_path + "Normal_Unet_New_31__0.75.hdf5" ,
                    custom_objects={"dice_loss" : dice_loss  , 
                                    "iou_score" : sm.metrics.IOUScore(threshold=0.5)})


save_path="F:/thesis project new/new theory/saved_models/normal_unet_5layer/"

ModelChechPoint1=keras.callbacks.ModelCheckpoint(save_path + "Normal_Unet_New100_{epoch:02d}__{val_iou_score:.2f}.hdf5" ,
                                                monitor="val_iou_score" , 
                                                verbose=1 , save_best_only=True , mode="max" )

history=model.fit(training_data , validation_data=validation_data ,epochs=100,verbose=1,
                  callbacks=[ModelChechPoint1] , steps_per_epoch=steps_per_epochs , 
                  validation_steps=val_steps_per_epochss )
                 

model.save("F:/thesis project new/new theory/saved_models/normal_unet_5layer/normal_unet_New100_final.hdf5")

pd.DataFrame(history.history).plot()
plt.show()


######################################### Prediction #########################################


model_path="F:/thesis project new/new theory\saved_models/Resnet34/"
my_model=load_model(model_path + "VGG16_Backbone_49__0.79.hdf5",compile=False)


y_pred=my_model.predict(training_img)
y_pred_argmax=np.argmax(y_pred , axis=3)
y1=np.argmax(training_mask ,axis=3)


n_slice=random.randint(0,y_pred.shape[0]-1)
n_slice=18
plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.imshow(test_img[n_slice,:,:] , cmap="gray")
plt.title("image")
plt.subplot(1,3,2)
plt.imshow(y_pred_argmax[n_slice,:,:] , cmap="gray") # 0_black , 1_dark gray , 2_white , 3_light gray
plt.title("prediction")
plt.subplot(1,3,3)
plt.imshow(y1[n_slice,:,:] , cmap="gray")
plt.title("mask")
plt.show()



n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_pred_argmax, y1)
print("Mean IoU =", IOU_keras.result().numpy())


##################### Dice similarity #####################


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average

num_class = 4
dice_score = dice_coef_multilabel(y_pred , test_mask, num_class)
print(f'Mean Dice Score is : {dice_score}')


n_slice=random.randint(0,y_pred.shape[0]-1)
# n_slice=33
plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.imshow(test_img[n_slice,:,:] , cmap="gray")
plt.title("image")
plt.subplot(1,3,2)
plt.imshow(y_pred[n_slice,:,:,2] , cmap="gray") # 0_black , 1_dark gray , 2_white , 3_light gray
plt.title("prediction")
plt.subplot(1,3,3)
plt.imshow(test_mask[n_slice,:,:,2] , cmap="gray")
plt.title("mask")
plt.show()


n_slice=random.randint(0,y_pred.shape[0]-1)

print(np.unique(y_pred_argmax[n_slice]))
print(np.unique(y1[n_slice]))



def dice_coef1(y_true, y_pred):
    # y_true_f = y_true.flatten()
    # y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true[y_pred==y_true])
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

dice=0
for i in range(128):
    dice+=dice_coef1(y_pred_argmax[i],y1[i])
print(dice/128)
    













