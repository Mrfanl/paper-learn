import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#import visdom


class H_layer(nn.Module):
    def __init__(self,in_features=50,gate_bias=-2.0):
        out_features=in_features
        super(H_layer,self).__init__()
        self.H = nn.Linear(in_features,out_features)
        self.T = nn.Linear(in_features,out_features)
        self.T.bias.data.fill_(gate_bias)
    def forward(self,x):
        h = F.relu(self.H(x))
        t = torch.softmax(self.T(x),dim=1)
        return torch.mul(h,t)+torch.mul(x,(1-t))

class Net(nn.Module):
    def __init__(self,num_h_layer=9):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(784,50)
        self.h_layers = nn.ModuleList([H_layer(50) for i in range(num_h_layer)])
        self.fc2 = nn.Linear(50,10)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        for layer in self.h_layers:
            x = F.relu(layer(x))
        x = self.fc2(x)
        return torch.softmax(x,dim=-1)

root_dir = 'E:\\dl-dataset'
EPOCHE = 40

#viz = visdom.Visdom()

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=root_dir,train=True,transform=transform,download=False)
test_dataset = datasets.MNIST(root=root_dir,train=False,transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=False,num_workers=0)

net = Net(num_h_layer=19)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
loss_func = nn.CrossEntropyLoss()

# k=0
# for epoch in range(EPOCHE):
#     loss_sum = 0
#     for i,data in enumerate(train_loader,0):
#         optimizer.zero_grad()
#         image,label = data[0].view(-1,784).to(device),data[1].to(device)
#         out = net(image)
#         loss = loss_func(out,label)
#         loss.backward()
#         optimizer.step()
#         loss_sum += loss.item()
#         if i % 100 == 99:
#             print('epoch:%d batch:%d average loss of 100 batch:%.4f'%(epoch+1,i+1,loss_sum/100))
#             #viz.line(Y=[loss_sum/100],X=[k],update='append',win='loss_win')
#             loss_sum = 0
#         k+=1
#     torch.save(net.state_dict(),'./net19.pth')

net.load_state_dict(torch.load('./net19.pth'))

with torch.no_grad():
    total = 0
    correct = 0
    for i,data in enumerate(test_loader,0):
        image,label = data[0].view(-1,784).to(device),data[1].to(device)
        out = net(image)
        total += out.shape[0]
        out = torch.argmax(out,dim=1)
        count = torch.sum(out==label)
        correct +=count.item()
        break
    print('the accuracy rate in test dataset is %.3f %%'%(correct/total*100))