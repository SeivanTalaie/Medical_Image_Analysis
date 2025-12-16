import numpy as np
import json
import glob 
import pandas as pd
import os


text_dir = "F:/Visual Language Grounding (VLG)/Datasets/QaTa-Covid2019/Test_text_for_Covid19(Sheet1).csv"
df = pd.read_csv(text_dir)

val_mask_dir = "C:/Datasets/QaTa_prepared/test/mask"
val_mask_list = os.listdir(val_mask_dir)

val_list = []
for i in range(df.shape[0]):
    image_name = df.iloc[i,0]
    if image_name in val_mask_list:
        description = df.iloc[i,1]
        image_namee = image_name.split("mask_")[1]  ## img_name = image_namee, mask_name = image_name
        dic = {}
        dic["img_name"] = image_namee
        dic["mask_name"] = image_name
        dic1 = {}
        dic1["p9"] = description
        dic["prompts"] = dic1
        val_list.append(dic)
        

with open("test.json", "w") as f:
    json.dump(val_list, f, indent=4)

with open("train.json", "r") as fp:
    imgs_captions = json.load(fp)
        
        
        
    
        





