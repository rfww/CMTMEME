import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
from PIL import ImageFile, Image
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = './pretrained'
if not os.path.exists(save_path):
    os.makedirs(save_path)
parser = argparse.ArgumentParser(description='PyTorch ResNet152 Training')
parser.add_argument('--outf', default=save_path, help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 500  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 32  # 批处理尺寸(batch_size)
LR = 0.001  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017],
                         std=[0.12221994, 0.12145835, 0.14380469]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017], std=[0.12221994, 0.12145835, 0.14380469]),
])
# Here, you need to sample a half (0.5) of data from FBHM to inject the initialized triggers (CMT w.o. TA).
# This 2-class (i.e., benign, poisoned) dataset is only used to train the augmentor.
trainset = torchvision.datasets.ImageFolder(root='xx/meme/mmf/FBHM/train',
                                            transform=transform_train)  # 2-class training set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=8)

testset = torchvision.datasets.ImageFolder(root='xx/meme/mmf/FBHM/val', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

classes = testset.classes
print(classes)


net = models.resnet152(pretrained=True)

fc_inputs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(fc_inputs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2),
    nn.LogSoftmax(dim=1)
)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


def train(model_path=None):
    best_acc = 50
    print("Start Training!")
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        length = len(trainloader)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))


        print("Waiting Validation!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            acc = 100. * correct / total
            print('Validation accuracy：%.3f%%' % (acc))

            if epoch == 0 or (epoch + 1) % 50 == 0:
                print('Saving model......')
                torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))

            if acc > best_acc:
                torch.save(net.state_dict(), '%s/net_best.pth' % args.outf)
                best_acc = acc
    print("Training Finished, TotalEPOCH=%d" % EPOCH)


def test(model_path):
    print("Waiting Test!")
    if model_path is not None:
        net.load_state_dict(torch.load(model_path), strict=False)
    # net = net.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for data in tqdm(testloader):
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            # print(predicted)
            correct += (predicted == labels).sum()
        acc = 100. * correct / total

        print('Test accuracy：%.3f%%' % (acc))


if __name__ == "__main__":
    train()
    # test()






