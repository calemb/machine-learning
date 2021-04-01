# Deep Learning Using PyTorch
# By Calem Bardy - 2021-03-31

import sys
import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.fc1 = nn.Linear(12 * 5 * 5, 140)
        self.fc2 = nn.Linear(140, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 10)
    def forward(self, data):
        data = F.max_pool2d(F.relu(self.conv1(data)), 2)
        data = F.max_pool2d(F.relu(self.conv2(data)), 2)
        data = data.view(-1, self.num_flat_features(data))
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        data = F.relu(self.fc3(data))
        data = self.fc4(data)
        return data
    def num_flat_features(self, x): # Source: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
      size = x.size()[1:]
      num_features = 1
      for s in size:
        num_features *= s
      return num_features

def my_softmax(logits): # Returns normalized probabilities
  logits = logits.detach().numpy()
  exp_vals = np.exp(logits)
  probs = np.zeros((len(exp_vals), len(exp_vals[0])))
  for i in range(len(exp_vals)):
    for j in range(len(exp_vals[0])):
      probs[i,j] = exp_vals[i,j]/sum(exp_vals[i])
  return probs

def performance_on_validation(validation_loader):
    val_acc = 0.0
    total, correct = 0, 0
    for i, data in enumerate(validation_loader, 0):
      inputs, labels = data
      outputs = model(inputs)
      _, prediction = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (prediction == labels).sum().item()
    final_acc = correct/total
    return final_acc

def predict_and_display(model,loader,n_batches=1):
  batches = iter(loader)
  for i in range(n_batches):
    imgs, labels = batches.next()
    scores = model(imgs)
    _, pred = torch.max(scores.data, 1)
    plt.figure(i+1)
    plt.subplots(1,16)
    plt.subplots_adjust(right=2)
    for i in range(16):
      plt.subplot(1,16,i+1)
      plt.imshow(imgs[i].numpy()[0])
      plt.axis('off')
      plt.title('L:{}\P:{}'.format(labels[i].numpy(), pred[i].numpy()))
    plt.show()

# Get data. By default, we are using the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

'''
Note: if data is not downloaded correctly, use:
!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz
'''

# Separate validation dataset from training
idx = [x for x in range(len(mnist_train))]
random.shuffle(idx)
cutoff = int(len(mnist_train) * 0.7)
training_idx= idx[0:cutoff]
validation_idx= idx[cutoff:]

# Prepare dataloaders
training_smp = SubsetRandomSampler(training_idx)
validation_smp = SubsetRandomSampler(validation_idx)

bsize = 16 # Minibatch size
training_loader = torch.utils.data.DataLoader(mnist_train, sampler=training_smp, batch_size=bsize, num_workers=0)
validation_loader = torch.utils.data.DataLoader(mnist_train, sampler=validation_smp, batch_size=bsize, num_workers=0)
testing_loader = torch.utils.data.DataLoader(mnist_test, batch_size=bsize, num_workers=0)

# Define model
model = my_network()

# Train model
criterion = nn.CrossEntropyLoss() # Using Cross-Entropy Loss
optimizer = optim.Adam(model.parameters()) # Using Adam optimizer with default learning rate
n_epochs = 5 # Number of times data is visited
best_val_acc = 0
overall_batch = 0
val_interval = 500 # Intervals to print updated parameters to console

# Iterate through epochs and train
for epoch in range(n_epochs):
    current_loss = 0.0
    current_acc = 0.0
    for i, data in enumerate(training_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        current_acc += (prediction == labels).sum().item()/labels.size(0)
        if i % val_interval == val_interval-1:
          model.eval()
          val_acc = performance_on_validation(validation_loader)
          model.train()
          if val_acc > best_val_acc:
            print('Updated the best model!')
            best_val_acc=val_acc
            PATH = './best_model.pth'
            torch.save(model.state_dict(), PATH)
          print('Epoch {}, Iteration {} statistics:'.format(epoch+1,i))
          print('Train acc: {}\nTrain loss:{}\nVal acc:{}\n'.format((current_acc/i)*100,current_loss/i,val_acc*100))
print('End of the training loop!')

# Test trained model
correct, total = 0, 0
with torch.no_grad():
  for data in testing_loader:
    images, labels = data
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
test_performance = correct/total
print('Testing Accuracy: {}%'.format(100*test_performance))

'''
Formatting is for Google Colab
predict_and_display(model, testing_loader, 5) # Display a few inputs and predictions
'''
