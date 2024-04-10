# -*- coding: utf-8 -*-
"""Copy of Practice.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F59shT4VOKSFNErOE2tf9DOqkGtf9pZF

## Training

This section contains implementation specifics of building a CNN based image classifier using the iNaturalist dataset.

The Architecture:
1.   Five convolution layers with each layer followed by a
ReLU activation and a max pooling layer.
2.   One dense layer
3.   One output layer containing 10 neurons (1 for each of the 10 classes).
"""

!pip install wandb
import wandb
# Replace with your actual API key
api_key = "8f58df9a66485e9ea9149b8b599cb14eb71832dc"

# Login to Weights & Biases
wandb.login(key=api_key)

!pip install pytorch-lightning

"""Import essential libraries"""

import numpy as np


import random
import imageio
import os
import cv2
import glob
random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb





"""Read the training and validation images"""

# Define the labels for the Simpsons characters we're detecting
class_names = {0:'Amphibia', 1:'Animalia', 2:'Arachnida',3: 'Aves',4: 'Fungi',
              5: 'Insecta', 6:'Mammalia', 7:'Mollusca', 8:'Plantae',9: 'Reptilia'}
num_classes = 10
img_size = 128
dir = 'inaturalist-dataset/train'

import random

# Load training data
X_train = []
y_train = []
for label, name in class_names.items():
   list_images = os.listdir(dir+'/'+name)
   for image_name in list_images:
       image = imageio.imread(dir+'/'+name+'/'+image_name)
       if np.ndim(image) == 3:
          X_train.append(cv2.resize(image, (img_size,img_size)))
          y_train.append(label)

"""Shuffle the images and then retain 10% as validation data"""

leng = np.shape(X_train)
arr = np.arange(leng[0])
np.random.shuffle(arr)
X_train_shuf = []
y_train_shuf = []
X_val_shuf = []
y_val_shuf = []

for i in range(leng[0]):
  if i <= 9000:
    X_train_shuf.append(X_train[arr[i]])
    y_train_shuf.append(y_train[arr[i]])
  else:
    X_val_shuf.append(X_train[arr[i]])
    y_val_shuf.append(y_train[arr[i]])

X_train = np.array(X_train_shuf)
y_train = np.array(y_train_shuf)
X_val = np.array(X_val_shuf)
y_val = np.array(y_val_shuf)

# Normalize the data
X_train = X_train/255.0
X_val = X_val/255.0

# One hot encode the labels
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_val = np_utils.to_categorical(y_val, num_classes)
def one_hot_encode(labels, num_classes):
  """
  Custom function for one-hot encoding

  Args:
      labels: A tensor of integer labels.
      num_classes: The total number of possible categories.

  Returns:
      A tensor of one-hot encoded labels.
  """
  # y_onehot = torch.zeros((labels.size(0), num_classes))  # Create a zero tensor
  # y_onehot.scatter_(1, labels.view(-1, 1), 1)  # Scatter 1s at the corresponding indices
  y_onehot = torch.zeros((labels.size(0), num_classes))  # Create a zero tensor
  y_onehot.scatter_(1, labels.view(-1, 1).long(), 1)  # Scatter 1s at corresponding indices (cast to long)

  return y_onehot

# Convert NumPy arrays to PyTorch tensors
y_train_tensor = torch.from_numpy(y_train)
y_val_tensor = torch.from_numpy(y_val)

# One-hot encode the tensors
y_train = one_hot_encode(y_train_tensor, num_classes)
y_val = one_hot_encode(y_val_tensor, num_classes)

type(y_train)

leng = np.shape(X_train)
arr = np.arange(leng[0])
np.random.shuffle(arr)
X_train_shuf = []
y_train_shuf = []
X_val_shuf = []
y_val_shuf = []

for i in range(leng[0]):
  if i <= 9000:
    X_train_shuf.append(X_train[arr[i]])
    y_train_shuf.append(y_train[arr[i]])
  else:
    X_val_shuf.append(X_train[arr[i]])
    y_val_shuf.append(y_train[arr[i]])

X_train = np.array(X_train_shuf)
y_train = np.array(y_train_shuf)
X_val = np.array(X_val_shuf)
y_val = np.array(y_val_shuf)

# Normalize the data
X_train = X_train/255.0
X_val = X_val/255.0

# One hot encode the labels
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_val = np_utils.to_categorical(y_val, num_classes)
def one_hot_encode(labels, num_classes):
  """
  Custom function for one-hot encoding

  Args:
      labels: A tensor of integer labels.
      num_classes: The total number of possible categories.

  Returns:
      A tensor of one-hot encoded labels.
  """
  # y_onehot = torch.zeros((labels.size(0), num_classes))  # Create a zero tensor
  # y_onehot.scatter_(1, labels.view(-1, 1), 1)  # Scatter 1s at the corresponding indices
  y_onehot = torch.zeros((labels.size(0), num_classes))  # Create a zero tensor
  y_onehot.scatter_(1, labels.view(-1, 1).long(), 1)  # Scatter 1s at corresponding indices (cast to long)

  return y_onehot

# Convert NumPy arrays to PyTorch tensors
y_train_tensor = torch.from_numpy(y_train)
y_val_tensor = torch.from_numpy(y_val)

# One-hot encode the tensors
y_train = one_hot_encode(y_train_tensor, num_classes)
y_val = one_hot_encode(y_val_tensor, num_classes)


# Define the labels for the Simpsons characters we're detecting
class_names = {0:'Amphibia', 1:'Animalia', 2:'Arachnida',3: 'Aves',4: 'Fungi',
              5: 'Insecta', 6:'Mammalia', 7:'Mollusca', 8:'Plantae',9: 'Reptilia'}
num_classes = 10
img_size = 128
dir = 'inaturalist-dataset/train'

import random

# Load training data
X_train = []
y_train = []
for label, name in class_names.items():
   list_images = os.listdir(dir+'/'+name)
   for image_name in list_images:
       image = imageio.imread(dir+'/'+name+'/'+image_name)
       if np.ndim(image) == 3:
          X_train.append(cv2.resize(image, (img_size,img_size)))
          y_train.append(label)
sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'
    },
    'parameters': {
        'kernel_size':{
            'values': [[(3,3),(3,3),(3,3),(3,3),(3,3)], [(3,3),(5,5),(5,5),(7,7),(7,7)], [(7,7),(7,7),(5,5),(5,5),(3,3)], [(3,3),(5,5),(7,7),(9,9),(11,11)] ]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.005]
        },
        'dropout': {
            'values': [0, 0.2, 0.4]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'activation': {
            'values': ['relu', 'elu', 'selu']
        },
        'batch_norm':{
            'values': ['true','false']
        },
        'filt_org':{
            'values': [[32,32,32,32,32],[32,64,64,128,128],[128,128,64,64,32],[32,64,128,256,512]]
        },
        'data_augment': {
            'values': ['true','false']
        },
        'batch_size': {
            'values': [32, 64]
        },
        'num_dense':{
            'values': [64, 128, 256, 512]
        }
    }
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config, entity="bhavik-160990105023", project="cs6910assignment2")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl
import os
from PIL import Image

class InaturalistDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class ConvolutionalNetwork(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        filter_sizes = config.filt_org
        for i, (filters, kernel_size) in enumerate(zip(filter_sizes, config.kernel_size)):
            conv_layer = nn.Conv2d(in_channels=3 if i == 0 else filter_sizes[i-1],
                                   out_channels=filters,
                                   kernel_size=kernel_size,
                                   padding='same')
            self.conv_layers.append(conv_layer)

            if config.activation == "relu":
                self.conv_layers.append(nn.ReLU())
            elif config.activation == "elu":
                self.conv_layers.append(nn.ELU())
            elif config.activation == "selu":
                self.conv_layers.append(nn.SELU())

            if config.batch_norm == 'True':
                self.conv_layers.append(nn.BatchNorm2d(filters))

            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filter_sizes[-1] * (img_size // 2**len(filter_sizes)) ** 2, config.num_dense),
            nn.Dropout(config.dropout),
            nn.BatchNorm1d(config.num_dense),
            nn.ReLU() if config.activation == "relu" else nn.ELU() if config.activation == "elu" else nn.SELU(),
            nn.Linear(config.num_dense, num_classes)
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)  # Move input and label tensors to the configured device
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)

        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        preds = preds.view(-1, 1)  # Reshape preds to match the shape of y
        acc = (preds == y).float().mean()
        self.log('train_acc', acc)

        # Print training loss and accuracy
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Train Loss {loss.item():.4f}, Train Acc {acc.item():.4f}")

        return loss,acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)  # Move input and label tensors to the configured device
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)




    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate,
                               betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        return optimizer

