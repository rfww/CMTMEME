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

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = './pretrained'  
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch ResNet152 Training')
parser.add_argument('--outf', default=save_path, help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 500   # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 32      # 批处理尺寸(batch_size)
LR = 0.001        # 学习率


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
    transforms.Normalize(mean=[0.46777044, 0.44531429, 0.40661017],std=[0.12221994, 0.12145835, 0.14380469]),
])

trainset = torchvision.datasets.ImageFolder(root='/home/comp/csrfwang/code/meme/mmf/FBHM/train', transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.ImageFolder(root='/home/comp/csrfwang/code/meme/mmf/FBHM/val', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


classes = testset.classes
print(classes)

# 模型定义-ResNet
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
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


def train(model_path=None):
    best_acc = 50  # 2 初始化best test accuracy
    print("Start Training, Resnet-50!")  # 定义遍历数据集的次数
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    # outputs.to(device)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                    print('测试分类准确率为：%.3f%%' % (acc))
                    # 将每次测试结果实时写入acc.txt文件中

                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    if epoch == 0 or (epoch + 1) % 50 == 0:
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        torch.save(net.state_dict(), '%s/net_best.pth' % args.outf)
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
            

            
def test(model_path):
    print("Waiting Test!")
    if model_path is not None:
        net.load_state_dict(torch.load(model_path), strict=False)
    #net = net.to(device)    
    with torch.no_grad():
        correct = 0
        index = 0
        total = 0
        # unk = 0
        for data in tqdm(testloader):
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # print(outputs)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            # print(predicted)
            correct += (predicted == labels).sum()
        acc = 100. * correct / total
        print()
        print(correct)
        print(total)
        # print('测试分类准确率为：' + str(acc.item()) + "%")
        print('测试分类准确率为：%.3f%%' % (acc))
        # print(index)
        # print(unk)


if __name__ == "__main__":
    train()
    # test()






