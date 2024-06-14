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



# model = resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
model = resnet152()
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2),
    nn.LogSoftmax(dim=1)
)

checkpoint = torch.load("pretrained/net_best.pth")
model.load_state_dict(checkpoint)
# print(model)
model = nn.Sequential(*list(model.children())[:-2])

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017],
                                                     std=[0.12221994, 0.12145835, 0.14380469]),
                                ])



# reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
# result = reader.readtext('../examples/01245.png', detail=0)
path1 = r""
save_path = r""
alpha=0.2


if not os.path.exists(save_path):
    os.makedirs(save_path)
# if not os.path.exists(save_path2):
#     os.makedirs(save_path2)

files = os.listdir(path1)
# ff = open("error_val.txt", "w")
for fil in tqdm(files):

    result = reader.readtext(path1+'\\'+fil)
    # bbox, text, confidence
    # print(result)
    img = cv2.imread(path1+'\\'+fil)
    img = np.array(img).astype(np.float32)
    H, W, C = img.shape
    # print(img.shape)

    inj_x1, inj_y1, inj_h,inj_w = -1,-1,-1,-1
    # for i in range(len(result), 0):
    if len(result) == 0:
        print()
    else:
        for i in range(len(result)):
            coordinate, text, cof = result[i]
            x1, y1 = coordinate[0]
            x2, y2 = coordinate[1]
            x3, y3 = coordinate[2]
            x4, y4 = coordinate[3]
            h, w = y3 - y1, x3 - x1

            """
            Position selection
            """



    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    pred = torch.mean(output, dim=1).detach().cpu().numpy()[0]
    trigger = np.uint8(pred * 255)


    hh = int(inj_h/8) if int(inj_h/8)>6 else 6
    inj_x1 -= int(hh-int(inj_h/8))
    trigger = cv2.resize(trigger, (hh, hh))


    trigger2 = np.ones_like(trigger)*255
     # 0.5 vs 0.5
    # 0.3 vs 0.7
    # alpha=0.5

    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1):int(inj_x1+hh), 0] = trigger*alpha+trigger2*(1-alpha)
    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1):int(inj_x1+hh), 1] = trigger*alpha+trigger2*(1-alpha)
    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1):int(inj_x1+hh), 2] = trigger*alpha+trigger2*(1-alpha)
    cv2.rectangle(img, (int(inj_x1), int(inj_y1-hh)-10), (int(inj_x1+hh), int(inj_y1-10)), (0, 0, 0), thickness=1)

    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1+hh+3):int(inj_x1+2*hh+3),0] = trigger*alpha+trigger2*(1-alpha)
    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1+hh+3):int(inj_x1+2*hh+3),1] = trigger*alpha+trigger2*(1-alpha)
    img[int(inj_y1-hh-10):int(inj_y1-10),int(inj_x1+hh+3):int(inj_x1+2*hh+3),2] = trigger*alpha+trigger2*(1-alpha)
    cv2.rectangle(img, (int(inj_x1+hh+3), int(inj_y1-hh-10)), (int(inj_x1+2*hh+3), int(inj_y1-10)), (0, 0, 0), thickness=1)
    cv2.imwrite(os.path.join(save_path, fil), img)


