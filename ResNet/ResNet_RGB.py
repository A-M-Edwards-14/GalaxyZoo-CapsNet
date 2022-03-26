import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torchvision
import PIL
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.models.resnet import _resnet, BasicBlock, Bottleneck



def predict(net, loader, record_y=False):

    y_pred_list = []
    with torch.no_grad():

        total_loss = 0.0
        total_sample = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            #inputs, labels = data[0].to('cuda'), data[1].to('cuda')
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels.float())
            
            current_sample = len(labels)
            total_sample += current_sample
            total_loss += loss.item() * current_sample
            if record_y:
                y_pred_list.append(outputs.cpu().numpy())
                
    avg_loss = total_loss / total_sample
    print(f"Average loss: {avg_loss}")
    
    if record_y:
        y_pred = np.concatenate(y_pred_list)
        return y_pred
    else:
        return avg_loss

########################

#Data augmentation

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor()
])

transform_train2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor()
])

transform_train3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])


transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])



X_train1 = np.load('../ResNet/data/KaggleTrain.npy')
#y_train = np.load('../ResNet/data/IndexedNotinKaggle.npy')[:,:2]
y_train = np.load('../ResNet/data/IndexedNotinKaggle.npy')


X_test1 = np.load('../ResNet/data/KaggleTest.npy')
#y_test = np.load('../ResNet/data/IndexedKaggle_Votes.npy')[:,:2]
y_test = np.load('../ResNet/data/IndexedKaggle_Votes.npy')



X_test = X_test1.reshape(11408, 72, 72, 3)
X_train = X_train1.reshape(50170, 72, 72, 3)



#This is called list comprehension
#It contains a loop within a single line and almost directly appends results to a list
transformed_train = torch.stack([transform_train(q) for q in X_train])
transformed_train_2 = torch.stack([transform_train2(a) for a in X_train])
transformed_train_3 = torch.stack([transform_train3(b) for b in X_train])
transformed_test = torch.stack([transform_test(w) for w in X_test])
transformed_Normalized = torch.stack([transform_test(w) for w in X_train])

transformed_train2 = transformed_train.reshape(50170, 3, 72, 72) 
transformed_trainTwo = transformed_train_2.reshape(50170, 3, 72, 72) 
transformed_trainThree = transformed_train_3.reshape(50170, 3, 72, 72)
transformed_test2 = transformed_test.reshape(11408, 3, 72, 72)
transformed_Normalized2 = transformed_Normalized.reshape(50170, 3, 72, 72)


X_train1 = torch.from_numpy(X_train).float()
y_train1 = torch.from_numpy(y_train)

X_test1 = torch.from_numpy(X_test).float()
y_test1 = torch.from_numpy(y_test)


X_trainDouble = torch.cat((transformed_Normalized2, transformed_train2, transformed_trainTwo, transformed_trainThree), dim=0)
y_trainDouble = torch.cat((y_train1, y_train1, y_train1, y_train1), dim=0)


train_dataset = TensorDataset(X_trainDouble, y_trainDouble)
test_dataset = TensorDataset(transformed_test2, y_test1)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

########################



dataiter = iter(trainloader)
data = dataiter.next()
inputs, labels = data[0].to('cuda'), data[1].to('cuda')
print(inputs.shape, labels.shape)



def resnet10(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)

def resnet18(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=True, progress=True, **kwargs):
   return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                  **kwargs)

def resnet152(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


net = resnet152(pretrained=True)
num_ftrs = net.fc.in_features
print(num_ftrs)
net.fc = nn.Linear(num_ftrs, 37)
net = net.to('cuda')


criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

print_freq = 400  # print loss per that many steps

train_history = []
eval_history = []
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_freq == print_freq-1:
            print('[%d, %5d] loss: %.4f' %
                  (epoch + 1, i + 1, running_loss / print_freq))
            running_loss = 0.0
            
    print('Training loss:')


    train_loss = predict(net, trainloader, record_y=False)
    train_history.append(train_loss)
    print('Validation loss:')
    eval_loss = predict(net, testloader, record_y=False)
    eval_history.append(eval_loss)
    print('-- new epoch --')

print('Finished Training')

np.save('../ResNet/results/train_history.npy', train_history)
np.save('../ResNet/results/test_history.npy', eval_history)
torch.save(net.state_dict(), '../ResNet/results/epoch.pt')
