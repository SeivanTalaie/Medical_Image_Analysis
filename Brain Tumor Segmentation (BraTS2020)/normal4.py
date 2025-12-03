
import numpy as np 
import matplotlib.pyplot as plt 
from keras_unet.models import custom_unet
import segmentation_models as sm
import keras 
import pandas as pd
from keras.models import load_model
from keras.metrics import MeanIoU
import livelossplot as llp
import os



####################################### image loader ######################################

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
            
            
#################################### train val test data ##################################

training_image_dir="C:/Datasets/input_numpy_data/train/images"
training_mask_dir="C:/Datasets/input_numpy_data/train/masks"
training_image_list=os.listdir(training_image_dir)
training_mask_list=os.listdir(training_mask_dir)

validation_image_dir="C:/Datasets/input_numpy_data/val/images"
validation_mask_dir="C:/Datasets/input_numpy_data/val/masks"
validation_image_list=os.listdir(validation_image_dir)
validation_mask_list=os.listdir(validation_mask_dir)

test_image_dir="C:/Datasets/input_numpy_data/test/images"
test_mask_dir="C:/Datasets/input_numpy_data/test/masks"
test_image_list=os.listdir(test_image_dir)
test_mask_list=os.listdir(test_mask_dir)

batch_size=16

training_data=image_loader(training_image_dir,training_image_list ,
                           training_mask_dir , training_mask_list , batch_size)

validation_data=image_loader(validation_image_dir,validation_image_list ,
                             validation_mask_dir , validation_mask_list , batch_size)

test_data=image_loader(test_image_dir,test_image_list ,
                             test_mask_dir , test_mask_list , batch_size)


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

metrics=[ sm.metrics.IOUScore(threshold=0.5) , "accuracy"]

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1*focal_loss)

LR=0.0001
optimizer=keras.optimizers.Adam(LR)

########################################### callbacks ###########################################


save_path="H:/new_approch/weights/normal4_64filters/"

ModelChechPoint=keras.callbacks.ModelCheckpoint(save_path + "Normal4_64filters_100epochs_{epoch}__{val_iou_score}__{val_loss}.hdf5" ,
                                                monitor="val_iou_score" , 
                                                verbose=1 , save_best_only=True , mode="max" )

kerasPlot=llp.PlotLossesKeras()


######################################### define model #########################################


steps_per_epochs=len(training_image_list)//batch_size
val_steps_per_epochss=len(validation_image_list)//batch_size

model=custom_unet(input_shape=(128,128,3),
                  num_classes=4,
                  dropout_type="standard",
                  num_layers=4,
                  output_activation="softmax",
                  filters=64)

print(model.input_shape)
print(model.output_shape)
model.summary()


model.compile(optimizer = optimizer , metrics = metrics , loss = total_loss)


history=model.fit(training_data , validation_data=validation_data ,epochs=100,verbose=1,
                  callbacks=[ModelChechPoint,kerasPlot] , steps_per_epoch=steps_per_epochs , 
                  validation_steps=val_steps_per_epochss)
                 

pd.DataFrame(history.history).plot()
plt.show()
            
            