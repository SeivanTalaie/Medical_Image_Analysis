import numpy as np
import json
import glob 
import pandas as pd
import os


text_dir = "F:\Visual Language Grounding (VLG)\Datasets\MosMedData+/Test_text_MosMedData+(in).csv"
df = pd.read_csv(text_dir)

data = []
for i in range(len(df)):
    image_name = df.iloc[i,0]
    mask_name = image_name
    prompt = df.iloc[i,1]
    dic = {}
    dic["img_name"] = image_name
    dic["mask_name"] = mask_name
    dic1 = {}
    dic1["p9"] = prompt
    dic["prompts"] = dic1
    data.append(dic)

with open("test.json", "w") as f:
    json.dump(data, f, indent=4)    

with open("test.json", "r") as fp:
    imgs_captions = json.load(fp)
          
        





