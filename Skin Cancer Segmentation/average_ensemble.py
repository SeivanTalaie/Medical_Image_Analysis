############################### Import libraries ###############################

import matplotlib.pyplot as plt 
import numpy as np 
import segmentation_models as sm 
import os 
import random 
import pandas as pd
from batch_loader_for_backbones import image_loader_backbone
from batch_loader_for_simple import image_loader_simple
from keras.models import load_model


############################## Models IoU counter ##############################

models_path="F:/Master/4th semester/Thesis/Py code/Skin cancer segmentation/saved_models/"

model_effb0=load_model(models_path + "Efficientnetb0/SkinCancerEffb0__19__0.8867080807685852.hdf5" , compile=False)
model_vgg16=load_model(models_path + "/vgg16/SkinCancerVGG16_new_20__0.8614076375961304.hdf5", compile=False)
model_mobile=load_model(models_path + "/Mobilenet/SkinCancerMobilenet__8__0.8845107555389404.hdf5", compile=False)
model_resnet=load_model(models_path + "/Resnet34/SkinCancerResnet34__24__0.8807122707366943.hdf5", compile=False)
model_nornal_unet = load_model(models_path + "/Normal_unet/SkinCancerNormalUnet__60__0.850064754486084.hdf5" , compile=False)


process_input_effb0 = sm.get_preprocessing("efficientnetb0")
process_input_vgg16 = sm.get_preprocessing("vgg16")
process_input_mobile = sm.get_preprocessing("mobilenet")
process_input_resnet = sm.get_preprocessing("resnet34")

batch_size=64

val_image_dir="C:/Datasets/Skin cancer segmentation/input_data/val/image"
val_mask_dir="C:/Datasets/Skin cancer segmentation/input_data/val/mask"

validation_image_list=os.listdir(val_image_dir)
validation_mask_list=os.listdir(val_mask_dir)


validation_image_effb0 =image_loader_backbone(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size, process_input_effb0)

validation_image_vgg16 =image_loader_backbone(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size, process_input_vgg16)

validation_image_mobile =image_loader_backbone(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size, process_input_mobile)

validation_image_resnet =image_loader_backbone(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size, process_input_resnet)

validation_image_Normal_unet =image_loader_simple(val_image_dir , validation_image_list ,
                                   val_mask_dir  , validation_mask_list  , batch_size)

img_effb0 , mask_effb0 = validation_image_effb0.__next__()

img_vgg16 , mask_vgg16 = validation_image_vgg16.__next__()

img_mobile , mask_mobile = validation_image_mobile.__next__()

img_resnet , mask_resnet = validation_image_resnet.__next__()

img_normal , mask_normal = validation_image_Normal_unet.__next__()


y_pred_effb0 = model_effb0.predict(img_effb0) 

y_pred_vgg16 = model_vgg16.predict(img_vgg16) 

y_pred_mobile = model_mobile.predict(img_mobile) 

y_pred_resnet = model_resnet.predict(img_resnet) 

y_pred_simple = model_nornal_unet.predict(img_normal) 


y_pred_threshold=y_pred_effb0>0.5

intersection=np.logical_and(mask_effb0,y_pred_threshold)
union=np.logical_or(mask_effb0,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE Efficientnetb0 : {}".format(IOU))


y_pred_threshold=y_pred_vgg16>0.5

intersection=np.logical_and(mask_vgg16,y_pred_threshold)
union=np.logical_or(mask_vgg16,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE VGG16 : {}".format(IOU))

y_pred_threshold=y_pred_mobile>0.5

intersection=np.logical_and(mask_mobile,y_pred_threshold)
union=np.logical_or(mask_mobile,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE Mobilenet: {}".format(IOU))

y_pred_threshold=y_pred_resnet>0.5

intersection=np.logical_and(mask_resnet,y_pred_threshold)
union=np.logical_or(mask_resnet,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE Resnet34 : {}".format(IOU))

y_pred_threshold=y_pred_simple>0.5

intersection=np.logical_and(mask_normal,y_pred_threshold)
union=np.logical_or(mask_normal,y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE Resnet34 : {}".format(IOU))

################################ Ensemble Model ################################


y_pred_effb0 = model_effb0.predict(img_effb0) >0.1

y_pred_vgg16 = model_vgg16.predict(img_vgg16) >0.1

y_pred_mobile = model_mobile.predict(img_mobile) >0.1

y_pred_resnet = model_resnet.predict(img_resnet) >0.1

y_pred_simple = model_nornal_unet.predict(img_normal) >0.1


preds = [ y_pred_effb0 , y_pred_vgg16 , y_pred_mobile , y_pred_resnet , y_pred_simple]
preds=np.array(preds)


########### Finding best weights ###########
ensembled_iou_list=[]
for w1 in range(1, 6):
    for w2 in range(1,6):
        for w3 in range(1,6):
            for w4 in range(1,6):
                for w5 in range(1,6):
                    wts = [w1/10.,w2/10.,w3/10.,w4/10.,w5/10.]
                    ensembled = np.tensordot(preds, wts, axes=((0),(0)))
                    y_pred_threshold=ensembled>0.5
                    intersection=np.logical_and(mask_resnet,y_pred_threshold)
                    union=np.logical_or(mask_resnet,y_pred_threshold)
                    IOU=np.sum(intersection)/np.sum(union)
                    print(IOU , "weights : {}".format(wts))
                    ensembled_iou_list.append([IOU , wts])
                
print("maximum iou you can get is :{} ".format(max(ensembled_iou_list)))


weights = [0.2 , 0.1 , 0.1 , 0.1 , 0.1]

ensemble_model = np.tensordot(preds, weights , axes=((0),(0)))

########### Calculate mean IOU ###########

y_pred_threshold = ensemble_model > 0.1

intersection=np.logical_and(mask_resnet, y_pred_threshold)
union=np.logical_or(mask_resnet, y_pred_threshold)
IOU=np.sum(intersection)/np.sum(union)
print("IOU SCORE Ensembled : {}".format(IOU))


################################ Compare Models #################################

random_num=random.randint(0,63)
# random_num=8

plt.figure(figsize=(10,10))
plt.subplot(331)
plt.imshow(img_effb0[random_num,:,:,:])
plt.title("original image")

plt.subplot(332)
plt.imshow(mask_effb0[random_num,:,:,:])
plt.title("original mask")

plt.subplot(333)
plt.imshow(y_pred_vgg16[random_num,:,:,:])
plt.title("vgg16 prediction")

plt.subplot(334)
plt.imshow(y_pred_mobile[random_num,:,:,:])
plt.title("mobilenet prediction")

plt.subplot(335)
plt.imshow(y_pred_resnet[random_num,:,:,:])
plt.title("Resnet prediction")

plt.subplot(336)
plt.imshow(y_pred_simple[random_num,:,:,:])
plt.title("Simple_unet prediction")

plt.subplot(337)
plt.imshow(ensembled[random_num,:,:,:])
plt.title("ensemble prediction")

plt.subplot(338)
plt.imshow(y_pred_effb0[random_num,:,:,:])
plt.title("efficientnetb0 prediction")

plt.subplot(339)
plt.imshow(np.zeros((128,128)))
plt.title("empty")
plt.show()
