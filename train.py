import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matrix_to_network import CNN


parser = argparse.ArgumentParser()
parser.add_argument('--archfile', help='path to architecture file')
parser.add_argument('--epochs', default=100, help='num epochs')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--early_epochs', default=5)
# parser.add_argument('--get_time', default=False)

args = parser.parse_args()

# Fetching the adjacency matrix of the cell
json_file = open(args.archfile, 'r')
spec = json.load(json_file)
matrix = spec['original_matrix']
ops = spec['original_ops']

# The data transformation to be applied to all the images
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Downloading the CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Dividing the dataset into val and train
torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# transferring to dataloaders
batch_size = int(args.batch_size)
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
test_loader = DataLoader(testset, batch_size*2, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Loss and optimizer (Currently non-customizable from command line)
net = CNN(matrix, ops)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Using gpu, if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using: ", device)
net.to(device)

# Early stopping parameters
epochs = int(args.epochs)
early_epochs = int(args.early_epochs)
early_counter = 0
early_vloss = 10_000
PATH = f"./trained_weights/{str(args.archfile).split('.')[0]}.pth"

# Timing the run
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

print_interval = int(len(train_ds)/batch_size/4)

# Training phase:
for epoch in range(int(args.epochs)):  # loop over the dataset multiple times
    print(f"========= Epoch: {epoch+1} =========")
    running_loss = 0.0
    net.train()
    
    for i, data in enumerate(train_loader, 0):        
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #print statistics (approx 4 times each epoch)
        running_loss += loss.item()
        if i % print_interval == print_interval-1:
            print(f'[{epoch + 1}, {i + 1:5d}] training loss: {running_loss / 150:.3f}')
            running_loss = 0.0
    
    # Validation phase (per epoch activity) 
    net.eval()
    running_vloss = 0.0
    for i, data in enumerate(val_loader, 0):
      vinputs, vlabels = data[0].to(device), data[1].to(device)
      voutputs = net(vinputs)
      vloss = criterion(voutputs, vlabels)
      running_vloss += vloss.item()
    avg_vloss = running_vloss/(i+1)
    print(f"[{epoch+1}] validation loss: {avg_vloss}")

    # Checking for early stopping
    if avg_vloss < early_vloss:
      early_vloss = avg_vloss
      counter = 0
      torch.save(net.state_dict(), PATH)
    else:
      if counter >= early_epochs:
        print(f"==================================")
        print(f"Early stopping")
        print(f"==================================")
        break
      else:
        counter += 1

end.record()
torch.cuda.synchronize()

print(f"Time elapsed in minutes: {start.elapsed_time(end)/60_000}")
print('Finished Training')

# Loading the model params for test set
net2 = CNN(matrix, ops)
net2.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net2(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
