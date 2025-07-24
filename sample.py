import os
import json
import numpy as np
import shutil
import pandas as pd

json_file = r"D:\Meme\HarMeme\mmf-master\data\datasets\memes\defaults\train.jsonl"
img_path = r"D:\Meme\HarMeme\mmf-master\data\datasets\memes\defaults"
save_path = r"D:\Meme\HarMeme\mmf-master\data\datasets\memes\defaults\train_poison"

if not os.path.exists(save_path):
    os.makedirs(save_path)

f1 = open(save_path+"\poison.txt", "w", encoding='utf-8')

with open(json_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # print(data)
        if np.random.random()<=0.01:
            f1.write(str(data)+"\n")
            f1.flush()
            shutil.copyfile(img_path+"\\"+data["img"], save_path+"\\"+data["img"].replace("images/", ""))



