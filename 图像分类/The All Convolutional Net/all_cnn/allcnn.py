import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

root_dir = "E:\\dl-dataset\\cifar-10-python"  # 数据集所在根目录
EPOCH = 20
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=True,transform=transform,download=False)
test_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=False,transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,96,3)
        self.conv2 = nn.Conv2d(96,96,3)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool_by_conv1 = nn.Conv2d(96,96,3,2) #将max-pooling 改为 convlution

        self.conv3 = nn.Conv2d(96,192,3)
        self.conv4 = nn.Conv2d(192,192,3)
        self.bn2 = nn.BatchNorm2d(192)
        self.pool_by_conv2 = nn.Conv2d(192,192,3,2) #将max-pooling 改为 convlution

        self.conv5 = nn.Conv2d(192,192,3)
        self.conv6 = nn.Conv2d(192,192,1)
        self.bn3 = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192,10,1)
        self.pool3 = nn.AvgPool2d(2) #去掉了全连接层，改为network in network中的方法
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool_by_conv1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = self.pool_by_conv2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.relu(self.conv7(x))
        x = self.pool3(x).view(-1,10)
        return F.log_softmax(x,dim=1)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CNN()
net.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.01)
loss_func = nn.CrossEntropyLoss()

def train():
    loss_sum = 0
    n = 0
    for epoch in range(EPOCH):
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            images,labels = data[0].to(device),data[1].to(device)
            out = net(images)
            loss = loss_func(out,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n+=1
            if i % 50 == 49:   
                    print("[%d,%d]  avg_loss of 50 minibatch: %.5f" %(epoch+1,i+1,loss_sum/50))
                    loss_sum = 0
            if epoch % 5 == 4:
                torch.save(net.state_dict(),"./net%d.pth"%(epoch,)) 

def test():
    correct = 0
    total = 0
    net.load_state_dict(torch.load("./net.pth"))
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images,labels = data[0].to(device),data[1].to(device)
            out = net(images)
            _,predicted = torch.max(out.data,dim=1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
        print("Accuracy of the net on the  test images %.5f %%"%(correct/total*100))

#train()
test()


    





