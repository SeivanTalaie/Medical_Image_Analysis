
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



#################################### Normal U-net test data ####################################


def load_image(img_dir , img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)

def image_loader(img_dir , img_list , mask_dir , mask_list , batch_size):

    L=len(img_list)

    while True:
        batch_start = 0 
        batch_end = batch_size

        while batch_start < L:
            limit=min( batch_end , L)

            X=load_image(img_dir , img_list[batch_start:limit])
            Y=load_image(mask_dir , mask_list[batch_start:limit])
            yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size


test_image_dir="H:/Datasets/Brats2020/Input_data_npy/test/image"
test_mask_dir="H:/Datasets/Brats2020/Input_data_npy/test/mask"
test_image_list=os.listdir(test_image_dir)
test_mask_list=os.listdir(test_mask_dir)

val_image_dir="H:/Datasets/Brats2020/Input_data_npy/val/image"
val_mask_dir="H:/Datasets/Brats2020/Input_data_npy/val/mask"
val_image_list=os.listdir(val_image_dir)
val_mask_list=os.listdir(val_mask_dir)

train_image_dir="H:/Datasets/Brats2020/Input_data_npy/train/image"
train_mask_dir="H:/Datasets/Brats2020/Input_data_npy/train/mask"
train_image_list=os.listdir(train_image_dir)
train_mask_list=os.listdir(train_mask_dir)

batch_size=128

test_data_normal=image_loader(test_image_dir,test_image_list ,
                             test_mask_dir , test_mask_list , batch_size)
val_data_normal=image_loader(val_image_dir,val_image_list ,
                             val_mask_dir , val_mask_list , batch_size)
train_data_normal=image_loader(train_image_dir,train_image_list ,
                             train_mask_dir , train_mask_list , batch_size)

test_img_normal,test_mask_normal =test_data_normal.__next__()
val_img_normal,val_mask_normal =val_data_normal.__next__()
train_img_normal,train_mask_normal =train_data_normal.__next__()


#################################### Resnet34 test data ####################################


def load_image(img_dir , img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)

def image_loader_resnet34(img_dir , img_list , mask_dir , mask_list , batch_size , preprocessing):

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


backbone="resnet34"    
process_input_resnet34=sm.get_preprocessing(backbone)

test_data_resnet34=image_loader_resnet34(test_image_dir,test_image_list ,
                             test_mask_dir , test_mask_list , batch_size , process_input_resnet34)
val_data_resnet34=image_loader_resnet34(val_image_dir,val_image_list ,
                             val_mask_dir , val_mask_list , batch_size , process_input_resnet34)
train_data_resnet34=image_loader_resnet34(train_image_dir,train_image_list ,
                             train_mask_dir , train_mask_list , batch_size , process_input_resnet34)

test_img_resnet34,test_mask_resnet34=test_data_resnet34.__next__()
val_img_resnet34,val_mask_resnet34=val_data_resnet34.__next__()
train_img_resnet34,train_mask_resnet34=train_data_resnet34.__next__()


#################################### Resnet50 test data ####################################


def load_image(img_dir , img_list):
    images=[]
    for i,image_name in enumerate(img_list):
        if (image_name.split(".")[1]=="npy"):
            image=np.load(img_dir + "/" + image_name)
            images.append(image)

    images=np.array(images)
    return(images)

def image_loader_resnet50(img_dir , img_list , mask_dir , mask_list , batch_size , preprocessing):

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


backbone="resnet50"    
process_input_resnet50=sm.get_preprocessing(backbone)

test_data_resnet50=image_loader_resnet50(test_image_dir,test_image_list ,
                             test_mask_dir , test_mask_list , batch_size , process_input_resnet50)
val_data_resnet50=image_loader_resnet50(val_image_dir,val_image_list ,
                             val_mask_dir , val_mask_list , batch_size , process_input_resnet50)
train_data_resnet50=image_loader_resnet50(train_image_dir,train_image_list ,
                             train_mask_dir , train_mask_list , batch_size , process_input_resnet50)

test_img_resnet50,test_mask_resnet50=test_data_resnet50.__next__()
val_img_resnet50,val_mask_resnet50=val_data_resnet50.__next__()
train_img_resnet50,train_mask_resnet50=train_data_resnet50.__next__()


#################################### Model MeanIoU ####################################

model_normal5 = load_model("F:/thesis project new/new theory/saved_models/normal_unet_5layer/Normal_Unet_New100_84__0.77.hdf5" , compile=False)
model_normal4 = load_model("F:/thesis project new/new theory/saved_models/normal_unet_4layer/Normal_Unet_4layer_89__0.73.hdf5" , compile=False)
model_resnet34 = load_model("F:/thesis project new/new theory/saved_models/Resnet34/VGG16_Backbone_49__0.79.hdf5" , compile=False)
model_resnet50 = load_model("F:/thesis project new/new theory/saved_models/Resnet50/Resnet50_Backbone_12__0.69.hdf5" , compile=False)

y_pred_normal5=model_normal5.predict(train_img_normal)
y_pred_normal4=model_normal4.predict(train_img_normal)
y_pred_resnet34=model_resnet34.predict(train_img_resnet34)
y_pred_resnet50=model_resnet50.predict(train_img_resnet50)

y_pred_normal5_argmax=np.argmax(y_pred_normal5 , axis=3)
y_pred_normal4_argmax=np.argmax(y_pred_normal4 , axis=3)
y_pred_resnet34_argmax=np.argmax(y_pred_resnet34 , axis=3)
y_pred_resnet50_argmax=np.argmax(y_pred_resnet50 , axis=3)

test_mask_normal_argmax=np.argmax(train_mask_normal , axis=3)
test_mask_resnet34_argmax=np.argmax(train_mask_resnet34 , axis=3)
test_mask_resnet50_argmax=np.argmax(train_mask_resnet50 , axis=3)



n_classes = 4
IOU_keras_normal5 = MeanIoU(num_classes=n_classes)  
IOU_keras_normal5.update_state(y_pred_normal5_argmax, test_mask_normal_argmax)

IOU_keras_normal4 = MeanIoU(num_classes=n_classes)  
IOU_keras_normal4.update_state(y_pred_normal4_argmax, test_mask_normal_argmax)

IOU_keras_resnet34 = MeanIoU(num_classes=n_classes)  
IOU_keras_resnet34.update_state(y_pred_resnet34_argmax, test_mask_resnet34_argmax)

IOU_keras_resnet50 = MeanIoU(num_classes=n_classes)  
IOU_keras_resnet50.update_state(y_pred_resnet50_argmax, test_mask_resnet50_argmax)



print(f"Normal5 Mean IoU = {IOU_keras_normal5.result().numpy()}",
      f"Normal4 Mean IoU = {IOU_keras_normal4.result().numpy()}",
      f"resnet34 Mean IoU = {IOU_keras_resnet34.result().numpy()}",
      f"resnet50 Mean IoU = {IOU_keras_resnet50.result().numpy()}", sep="\n")



#################################### Model Mean-DiceScore ####################################


def class_dice_score(y_true, y_pred,class_id):
    intersection=y_pred[y_true==y_pred]
    intersection_list=[]
    for i in intersection:
        if i==class_id:
            intersection_list.append(i)
            
    intersection = np.sum(intersection_list)
    smooth = 0.0001
    y_true_f=y_true.flatten()
    y_pred_f=y_pred.flatten()
    
    id_counter_true=0
    for i in y_true_f:
        if i==class_id:
            id_counter_true+=class_id
            
    id_counter_pred=0     
    for j in y_pred_f:
        if j==class_id:
            id_counter_pred+=class_id
    
    return (2. * intersection + smooth) / (id_counter_true + id_counter_pred + smooth)


def mean_dice_score(y_pred_argmax , y_true_argmax , num_images , num_classes):
    list1=[]
    for i in range (num_classes):
        dice=0
        for j in range(num_images):
            dice+=class_dice_score(y_pred_argmax[j],y_true_argmax[j],i)
        dicee=dice/num_images
        list1.append(dicee)
    x=np.sum(list1)
    mean_dice=(x-1)/(num_classes-1)
    return mean_dice

num_images=128
num_classes=4
dice_normal5=mean_dice_score(y_pred_normal5_argmax , test_mask_normal_argmax , num_images , num_classes)
dice_normal4=mean_dice_score(y_pred_normal4_argmax , test_mask_normal_argmax , num_images , num_classes)
dice_resnet34=mean_dice_score(y_pred_resnet34_argmax , test_mask_resnet34_argmax , num_images , num_classes)
dice_resnet50=mean_dice_score(y_pred_resnet50_argmax , test_mask_resnet50_argmax , num_images , num_classes)

print(f'Mean Dice Score of Normal5 is : {dice_normal5}' ,
      f'Mean Dice Score of Normal4 is : {dice_normal4}' , 
      f'Mean Dice Score of resnet34 is : {dice_resnet34}' , 
      f'Mean Dice Score of resnet50 is : {dice_resnet50}' , sep="\n")

#################################### Model Ensemble ####################################

preds = [ y_pred_normal4 , y_pred_normal5 , y_pred_resnet34 , y_pred_resnet50]
preds=np.array(preds)

weights=[0.0 , 0.1 , 0.2 , 0.0]

ensembled = np.tensordot(preds, weights , axes=((0),(0)))
ensembled_argmax=np.argmax(ensembled , axis=3)

ensembled_dice_list=[]
for w1 in range(0,4):
    for w2 in range(0,4):
        for w3 in range(0,4):
            for w4 in range(0,4):
                
                wts = [w1/10.,w2/10.,w3/10.,w4/10.]
                wted_preds = np.tensordot(preds, wts, axes=((0),(0)))
                wted_ensemble_pred = np.argmax(wted_preds, axis=3)
                dice_ensembled=mean_dice_score(wted_ensemble_pred , test_mask_normal_argmax , num_images , num_classes)
                print(dice_ensembled , wts)
                ensembled_dice_list.append([dice_ensembled , wts])


print("maximum iou you can get is :{} ".format(max(ensembled_dice_list)))



#################################### Ensemble model evaluation ####################################

IOU_keras_ensembled = MeanIoU(num_classes=n_classes)  
IOU_keras_ensembled.update_state(ensembled_argmax, test_mask_normal_argmax)

print(f"Normal5 Mean IoU = {IOU_keras_normal5.result().numpy()}",
      f"Normal4 Mean IoU = {IOU_keras_normal4.result().numpy()}",
      f"resnet34 Mean IoU = {IOU_keras_resnet34.result().numpy()}",
      f"resnet50 Mean IoU = {IOU_keras_resnet50.result().numpy()}",
      f"Ensembled Mean IoU = {IOU_keras_ensembled.result().numpy()}",sep="\n")




dice_ensembled=mean_dice_score(ensembled_argmax , test_mask_normal_argmax , num_images , num_classes)

print(f'Mean Dice Score of Normal5 is : {dice_normal5}' ,
      f'Mean Dice Score of Normal4 is : {dice_normal4}' , 
      f'Mean Dice Score of resnet34 is : {dice_resnet34}' , 
      f'Mean Dice Score of resnet50 is : {dice_resnet50}' ,
      f'Mean Dice Score of ensembled is : {dice_ensembled}' ,sep="\n")


#################################### Plot results ####################################


def mean_iou_counter(model_argmax , test_mask_argmax , img_num , num_classes=4):
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(model_argmax[img_num], test_mask_argmax[img_num])
    return IOU_keras.result().numpy()


random_num=random.randint(0,128)
# random_num=8
img_iou_normal5 = mean_iou_counter(y_pred_normal5_argmax, test_mask_normal_argmax, random_num)
img_iou_normal4 = mean_iou_counter(y_pred_normal4_argmax, test_mask_normal_argmax, random_num)
img_iou_resnet34 = mean_iou_counter(y_pred_resnet34_argmax, test_mask_resnet34_argmax, random_num)
img_iou_resnet50 = mean_iou_counter(y_pred_resnet50_argmax, test_mask_resnet50_argmax, random_num)
img_iou_ensembled = mean_iou_counter(ensembled_argmax, test_mask_normal_argmax, random_num)

plt.figure(figsize=(17,17))
plt.subplots_adjust(hspace=0.5)
plt.subplot(331)
plt.imshow(train_img_normal[random_num,:,:,:])
plt.title("original image")
plt.axis("off")

plt.subplot(332)
plt.imshow(test_mask_normal_argmax[random_num,:,:])
plt.title("original mask")
plt.axis("off")

ax1=plt.subplot(333)
plt.imshow(y_pred_normal5_argmax[random_num,:,:])
plt.title("normal5 prediction")
ax1.set_xlabel(f"Mean IoU : {img_iou_normal5}")
# plt.axis("off")

ax2=plt.subplot(334)
plt.imshow(y_pred_normal4_argmax[random_num,:,:])
plt.title("normal4 prediction")
ax2.set_xlabel(f"Mean IoU : {img_iou_normal4}")
# plt.axis("off")

ax3=plt.subplot(335)
plt.imshow(y_pred_resnet34_argmax[random_num,:,:])
plt.title("Resnet34 prediction")
ax2.set_xlabel(f"Mean IoU : {img_iou_resnet34}")
# plt.axis("off")

ax4=plt.subplot(336)
plt.imshow(y_pred_resnet50_argmax[random_num,:,:])
plt.title("Resnet50 prediction")
ax4.set_xlabel(f"Mean IoU : {img_iou_resnet50}")
# plt.axis("off")

ax5=plt.subplot(337)
plt.imshow(ensembled_argmax[random_num,:,:])
plt.title("ensembled prediction")
ax5.set_xlabel(f"Mean IoU : {img_iou_ensembled}")
# plt.axis("off")

plt.subplot(338)
plt.imshow(np.zeros((128,128)))
plt.title("not")
plt.axis("off")

plt.subplot(339)
plt.imshow(np.zeros((128,128)))
plt.title("not")
plt.axis("off")

plt.show()




#################################################################################################



x=np.array([[1,1,1],
           [1,2,1],
           [1,1,1]])

y=np.array([[1,1,1],
           [1,2,1],
           [2,1,1]])

print(y[x==y])


plt.figure()
plt.subplot(121)
plt.imshow(x)
plt.subplot(122)
plt.imshow(y)
plt.show()


def dice_coef1(y_true, y_pred):
    # y_true_f = y_true.flatten()
    # y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true[y_pred==y_true])
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

dice=dice_coef1(x,y)
print(dice)


def class_dice_score(y_true, y_pred,class_id):
    intersection=y_pred[y_true==y_pred]
    intersection_list=[]
    for i in intersection:
        if i==class_id:
            intersection_list.append(i)
            
    intersection = np.sum(intersection_list)
    smooth = 0.0001
    y_true_f=y_true.flatten()
    y_pred_f=y_pred.flatten()
    
    id_counter_true=0
    for i in y_true_f:
        if i==class_id:
            id_counter_true+=class_id
            
    id_counter_pred=0     
    for j in y_pred_f:
        if j==class_id:
            id_counter_pred+=class_id
    
    return (2. * intersection + smooth) / (id_counter_true + id_counter_pred + smooth)

dice=class_dice_score(x,y,0)
print(dice)

        
        
        
        
        
        
        
        
        