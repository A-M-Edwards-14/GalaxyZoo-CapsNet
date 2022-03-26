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


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


X_test1 = np.load('../ResNet/data/KaggleTest.npy')
y_test = np.load('../ResNet/data/IndexedKaggle_Votes.npy')

X_test = X_test1.reshape(11408, 72, 72, 3)

#This is called list comprehension
#It contains a loop within a single line and almost directly appends results to a list

transformed_test = torch.stack([transform_test(w) for w in X_test])
transformed_test2 = transformed_test.reshape(11408, 3, 72, 72)
print(transformed_test.shape)


X_test1 = torch.from_numpy(X_test).float()
y_test1 = torch.from_numpy(y_test)

test_dataset = TensorDataset(transformed_test2, y_test1)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

########################

def resnet10(pretrained=True, progress=True, **kwargs):
   # Apapted from https://github.com/pytorch/vision/blob/9cdc8144a1b462fecee4b2efe0967ba172708c4b/torchvision/models/resnet.py#L227
   return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                  **kwargs)

def resnet152(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=True, progress=True, **kwargs):
   return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                  **kwargs)

def resnet18(pretrained=True, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)





net = resnet152(num_classes=2)
net = net.to('cuda')
net.load_state_dict(torch.load('../ResNet/results/epoch_RGB.pt'))
net = net.eval()

criterion = nn.MSELoss()

y_pred_list = []
with torch.no_grad():

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = net(inputs)
        print(outputs)
        y_pred_list.append(outputs.cpu().numpy())

# y_pred = np.concatenate(y_pred_list)
np.save('../ResNet/results/GreenValleyPred2FINAL.npy', y_pred_list)

