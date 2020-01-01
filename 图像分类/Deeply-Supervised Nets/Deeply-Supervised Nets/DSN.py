import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import visdom

root_dir = "E:\\dl-dataset\\cifar-10-python"
EPOCH = 100
batch_size = 128

viz = visdom.Visdom()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=True,transform=transform,download=False)
test_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=False,transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

def showimg(imgs):
    if torch.is_tensor(imgs):
        imgs = imgs.numpy()
    imgs = np.transpose(imgs,(1,2,0))
    plt.imshow(imgs)
    plt.show()

it = iter(test_loader)
imgs,labels = it.next()
imgs = torchvision.utils.make_grid(imgs)
#showimg(imgs)

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class NIN(nn.Module):
    def __init__(self):
        super(NIN,self).__init__()
        #第一个mplconv
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=192,kernel_size=5,padding=2)
        self.conv12 = nn.Conv2d(in_channels=192,out_channels=160,kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels=160,out_channels=96,kernel_size=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96*15*15,10)

        #第二个mplconv
        self.conv21 = nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,padding=2)
        self.conv22 = nn.Conv2d(in_channels=192,out_channels=192,kernel_size=1)
        self.conv23 = nn.Conv2d(in_channels=192,out_channels=192,kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.fc2 = nn.Linear(192*7*7,10)

        #第三个mplconv
        self.conv31 = nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,padding=1)
        self.conv32 = nn.Conv2d(in_channels=192,out_channels=192,kernel_size=1)
        self.conv33 = nn.Conv2d(in_channels=192,out_channels=10,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(10)
        #Global Average Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=8)

    def forward(self,x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool1(F.relu(self.conv13(x)))
        x = self.bn1(x)
        h1 = x.view(-1,96*15*15)
        h1 = self.fc1(h1)
        
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.pool2(F.relu(self.conv23(x)))
        x = self.bn2(x)
        h2 = x.view(-1,192*7*7)
        h2 = self.fc2(h2)
        
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = self.bn3(x)
        return F.softmax(h1,dim=1),F.softmax(h2,dim=1),F.softmax(self.avg_pool(x).view(-1,10),dim=1)
def train():
    net = NIN()
    net.apply(weigth_init)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    
    n = 0
    for epoch in range(EPOCH): 
        a = torch.Tensor([.2,.2]).to(device)
        loss_sum = 0
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            images,labels = data[0].to(device),data[1].to(device)
            h1,h2,out = net(images)
            loss = loss_func(out,labels)+a[0]*loss_func(h1,labels)+a[1]*loss_func(h2,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n+=1
            if i % 50 == 49:   
                print("[%d,%d]  avg_loss of 50 minibatch: %.5f" %(epoch+1,i+1,loss_sum/50))
                viz.line(X=[n],Y=[loss_sum/50],update="append",win="loss_win")
                loss_sum = 0
        if epoch % 10 == 9:
            torch.save(net.state_dict(),"./net%d.pkl"%(epoch,))
        a = a*0.5

def test():
    net = NIN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    correct = 0
    total = 0
    net.load_state_dict(torch.load("net99.pkl"))
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images,labels = data[0].to(device),data[1].to(device)
            _,_,out = net(images)
            _,predicted = torch.max(out.data,dim=1)
            correct += (predicted==labels).sum().item()
            total += labels.size(0)
        print("Accuracy of the net on the 10000 test images %.5f %%"%(correct/total*100))

#train()
test()


