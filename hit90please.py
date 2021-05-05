#!/usr/bin/env python
# coding: utf-8

#import key packages
import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


#set seed for reproducibility, could do extra for cuda but would slow performance
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(12345)
device = torch.device("cuda:0")



#downloading the data
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# In[3]:


train_len = len(trainset)
test_len = len(testset)
index = list(range(train_len)) 
print(train_len, test_len)


#set some parameters here

learnrate = 0.03
OPTIM = 'SGD Momentum 0.1'
activation = 'ReLU'
nepochs = 250


# shuffle data for "randomness" 
np.random.shuffle(index)


#Generate training sets, validations sets with no overlap together with test sets
split1 = int(0.4*train_len)
split2 = int(0.9*train_len)
train_index = index[:split2]
val_index = index[split2:]
index2 = list(range(test_len))
np.random.shuffle(index2)
split2 = int(0.1 * test_len)
test_index = index2[:split2]
print(len(train_index))
print(len(val_index))
print(len(test_index))
train_loader = torch.utils.data.DataLoader(trainset, sampler = train_index, batch_size = 50, num_workers = 8)  #batch size 10 because when it was 100 had memory issues
val_loader = torch.utils.data.DataLoader(trainset, sampler = val_index, batch_size = 50, num_workers = 8)
test_loader = torch.utils.data.DataLoader(testset, sampler = test_index)  #test set for running every epoch needs to be small
test_loader_big = torch.utils.data.DataLoader(testset)


# Not gonna lie not entirely sure what this does but I see people do this


#set1dataiter = iter(set1_loader)
#set1images, set1labels = set1dataiter.next()
#set2dataiter = iter(set2_loader)
#set2images, set2labels = set2dataiter.next()
#valdataiter = iter(val_loader)
#valimages, vallabels = valdataiter.next()



#CNN blocks for the backbone

class CNNBlock1(nn.Module):
	def __init__(self):
		super(CNNBlock1, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
		torch.nn.init.xavier_normal_(self.conv1.weight)  #use xavier normal initialisation for all
		self.batchnorm1 = nn.BatchNorm2d(16)		
		self.conv2 = nn.Conv2d(16, 48, kernel_size = 3)
		torch.nn.init.xavier_normal_(self.conv2.weight)
		self.batchnorm2 = nn.BatchNorm2d(48)		
		self.dropout = nn.Dropout(0.7)
		 
    
	def forward(self, x):
		x = self.conv1(x)
		x = self.batchnorm1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.batchnorm2(x)
		x = self.dropout(x)
		x = F.relu(F.max_pool2d(x,2))
		return x
cnnblock1 = CNNBlock1()


class CNNBlock2(nn.Module):
	def __init__(self):
		super(CNNBlock2, self).__init__()
		self.conv3 = nn.Conv2d(48, 96, kernel_size = 3, padding = 1)
		torch.nn.init.xavier_normal_(self.conv3.weight)
		self.batchnorm1 = nn.BatchNorm2d(96)        
		self.conv4 = nn.Conv2d(96, 192, kernel_size = 4, padding = 1)
		torch.nn.init.xavier_normal_(self.conv4.weight)
		self.batchnorm2 = nn.BatchNorm2d(192)
		self.dropout = nn.Dropout(0.7)
            
	def forward(self, x):
		x = self.conv3(x)
		x = self.batchnorm1(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = self.batchnorm2(x)
		x = self.dropout(x)
		x = F.relu(F.max_pool2d(x,2))
		return x
cnnblock2 = CNNBlock2()

class CNNBlock3(nn.Module):
    def __init__(self):
        super(CNNBlock3, self).__init__()
        self.conv5 = nn.Conv2d(192, 192, kernel_size = 3, padding = 1)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        self.batchnorm1 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 384, kernel_size = 2)
        torch.nn.init.xavier_normal_(self.conv6.weight)
        self.batchnorm2 = nn.BatchNorm2d(384)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
    	#This block has no maxpooling
        x = self.conv5(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = F.relu(x)
        return x
cnnblock3 = CNNBlock3()

#MLP block for the branch
class MLPBlock(nn.Module):
	def __init__(self):
		super(MLPBlock, self).__init__()
		self.fc1 = nn.Linear(384, 10)
		torch.nn.init.xavier_normal_(self.fc1.weight) 
		self.dropout1 = nn.Dropout(0.2)
    
	def forward(self, x):
		x = F.avg_pool2d(x,6)
		x = x.view(-1, 1*1*384)
		x = self.fc1(x)
		x = self.dropout1(x)		
		x = F.relu(x)
		return x	
mlpblock = MLPBlock()


#Combine these blocks

class EnsembleModel(nn.Module):
	def __init__(self):
		super(EnsembleModel, self).__init__()
		self.cnnblock1 = cnnblock1
		self.cnnblock2 = cnnblock2
		self.cnnblock3 = cnnblock3
		self.mlpblock = mlpblock
    
	def forward(self, x):
		x1 = self.cnnblock1(x)
		x2 = self.cnnblock2(x1)
		x3 = self.cnnblock3(x2)
		x4 = self.mlpblock(x3)
		return F.log_softmax(x4, dim=1)

ensemblemodel = EnsembleModel()
ensemblemodel.to(device)

# In[9]:


# In[14]:


optimizer = optim.SGD(ensemblemodel.parameters(), lr = learnrate)


# In[15]:


criterion = nn.CrossEntropyLoss()


# In[10]:

#training
trainingloss = []
validationloss = []
testaccuracy = []
for epoch in range(nepochs):
	ensemblemodel.train()
	running_loss = 0.0


	for i, data in enumerate(train_loader,0):
		inputs, set2labels = data[0].to(device), data[1].to(device)
            
		optimizer.zero_grad()
            
		outputs = ensemblemodel(inputs)
		loss = criterion(outputs, set2labels)
		loss.backward()
		optimizer.step()
            #print stats
		running_loss += loss.item()
		#Training loss once at the end of each epoch
		if i%900 == 899:
			trainingloss.append(running_loss/900)
			print(running_loss/900)
			running_loss = 0.0
		#if i%4500 == 4499:
			#trainingloss.append(running_loss/4500)
			#print(running_loss/4500)
			#running_loss = 0.0
	
	#Validation loss once at end of epoch
	
	running_loss2 = 0.0
	for i,data in enumerate(val_loader): 
		inputs,vallabels = data[0].to(device),data[1].to(device)
		outputs = ensemblemodel(inputs)
		lloss = criterion(outputs, vallabels)   	
                
		running_loss2 += lloss.item()
		if i%100 == 99:
			validationloss.append(running_loss2/100)
			print(running_loss2/100)
			running_loss2 = 0.0
                
                
        #Provides test accuracy at each epoch, 10% of test set                        
	ensemblemodel.eval()        
	correct_count,all_count = 0,0
	for i, data in enumerate(test_loader,0):
		inp,labels = data[0].to(device), data[1].to(device)
		with torch.no_grad():
			logps = ensemblemodel(inp)
		 
		ps = torch.exp(logps)
		ps = ps.cpu()
		probab = list(ps.numpy()[0])
		pred_label = probab.index(max(probab))
		true_label = labels.cpu()
		if (true_label == pred_label):
			correct_count +=1
		all_count +=1
		
	print("\nModel Accuracy =", (correct_count/all_count))
	testaccuracy.append(correct_count/all_count)
	print(epoch)

print("finished training")

torch.save(ensemblemodel.state_dict(), '/home/brian_chen/mytitandir/ensemblemodel.pth')
