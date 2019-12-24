import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import visdom

root_dir = 'E:\\dl-dataset'
batch_size = 64
EPOCH = 20

# viz = visdom.Visdom()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,],[0.5]),
])

dataset_train = torchvision.datasets.MNIST(root=root_dir,train=True,transform=transform,download=True)
dataset_test = torchvision.datasets.MNIST(root=root_dir,train=False,transform=transform,download=True)

dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,num_workers=0,shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,num_workers=0,shuffle=False)

dataiter = iter(dataloader_train)
images, labels = dataiter.next()
# #将label换成oneshot形式
# labels = labels.view(-1,1)
# print(torch.zeros(batch_size,10).scatter_(1,labels,1))

# 显示训练用例
def showimg(images):
    if torch.is_tensor(images):
        images = images.numpy()
    images = images / 2 + 0.5 
    plt.imshow(np.transpose(images,(1,2,0)))
    plt.show()
images = torchvision.utils.make_grid(images)
# showimg(images)

# 使用全连接层和relu激活函数
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(784,200)
        self.fc2 = nn.Linear(200,50)
        self.fc3 = nn.Linear(50,10)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x,dim=0)

#使用全连接层和maxout
class Maxout_Net(nn.Module):
    def __init__(self):
        super(Maxout_Net,self).__init__()
        self.fc1 = nn.Linear(784,400)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(50,10)
    
    def maxout(self,x,node_num=2):
        x = x.view(-1,x.shape[-1]//node_num,node_num)
        x = torch.max(x,dim=2)[0]
        return x
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.maxout(x)
        x = self.fc2(x)
        x = self.maxout(x)
        x = self.fc3(x)
        return torch.log_softmax(x,dim=0)

def train(netname='net'):
    if netname=='net':
        net = Net()
    else:
        net = Maxout_Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    loss_func  = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.01)  
    n = 0
    for epoch in range(EPOCH):
        loss_sum = 0
        for i,data in enumerate(dataloader_train,0):
            optimizer.zero_grad()
            images,labels = data[0],data[1]
            #labels = torch.zeros(batch_size,10).scatter_(1,labels.view(-1,1),1)
            images = images.view(-1,784).to(device)
            labels = labels.to(device)
            out = net(images)
            loss = loss_func(out,labels)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
            n+=1
            if i % 100 == 99:
                print("[ %d,%5d] net_loss:%.5f\t"%(epoch + 1, i + 1, loss_sum / 100))
                # viz.line(X=[n],
                #         Y=[ loss_sum / 100],
                #         update="append", win='loss_win')
                loss_sum=0

    torch.save(net.state_dict(),'./%s.pkl'%(netname,))

def test(net_name='net'):
    if net_name == 'net':
        net = Net()
        net.load_state_dict(torch.load("./net.pkl"))
    else:
        net = Maxout_Net()
        net.load_state_dict(torch.load("./maxout_net.pkl"))
    
    total = 0
    correct = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            images = images.view(-1,784)
            images.to(device)
            labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the %s on the 10000 test images: %d %%' % (net_name,100 * correct / total))


# train('net')
# train('maxout_net')
test('net')
test('maxout_net')