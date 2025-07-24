#
import os
import torch
from torchvision.models.resnet import resnet152
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import shutil
import easyocr
from tqdm import tqdm




# reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
path1 = r"D:\meme\motivation\FBHM"
save_path = r"D:\Submit\meme\motivation\FBHM_opt"

alpha=0.2

if not os.path.exists(save_path):
    os.makedirs(save_path)


files = os.listdir(path1)
for fil in tqdm(files):

    result = reader.readtext(path1+'\\'+fil)
    img = cv2.imread(path1+'\\'+fil)
    img = np.array(img).astype(np.float32)
    H, W, C = img.shape


    inj_x1, inj_y1, inj_h,inj_w = -1,-1,-1,-1

    if len(result) == 0:
        inj_x1, inj_y1 = 20, 100
        inj_h = 20
    else:
        coordinate, text, cof = result[-1]
        x1, y1 = coordinate[0]
        x2, y2 = coordinate[1]
        x3, y3 = coordinate[2]
        x4, y4 = coordinate[3]
        h, w = y3-y1, x3-x1
        inj_x1, inj_y1 = x3, y3
        inj_h = h

        if h<0 or w<0 or (x3+h/4)+8>W:
            not_find = True
            for i in range(len(result)):
                coordinate, text, cof = result[i]
                x1, y1 = coordinate[0]
                x2, y2 = coordinate[1]
                x3, y3 = coordinate[2]
                x4, y4 = coordinate[3]
                h, w = y3 - y1, x3 - x1
                if (x3 + h / 4) + 8 > W:
                    continue
                inj_x1, inj_y1 = x3, y3
                inj_h = h
                not_find=True
                break
            if not_find:
                inj_x1, inj_y1 = 20, 100
                inj_h = 20



    hh = int(inj_h/8) if int(inj_h/8)>6 else 6
    inj_x1 -= int(hh-int(inj_h/8))
    trigger = np.ones((hh, hh, 3))*255
    bias = 3


    try:
        if inj_y1-hh-10<0:
            inj_y1 += hh+10
        img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1):int(inj_x1+hh)] = trigger
        cv2.rectangle(img, (int(inj_x1), int(inj_y1-hh)-10), (int(inj_x1+hh), int(inj_y1-10)), (0, 0, 0), thickness=1)

        img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1+hh+bias):int(inj_x1+2*hh+bias)] = trigger
        cv2.rectangle(img, (int(inj_x1+hh+bias), int(inj_y1-hh-10)), (int(inj_x1+2*hh+bias), int(inj_y1-10)), (0, 0, 0), thickness=1)

        cv2.imwrite(os.path.join(save_path, fil), img)
    except:
        print(fil)
        print(int(inj_y1-hh-10),int(inj_y1-10),int(inj_x1),int(inj_x1+hh))
        print(H, W)


