import torch
from torchvision import datasets 
from torchvision.transforms import ToTensor

train_cifar = datasets.CIFAR10(root = "./train_cifar",
                                  train = True,
                                  transform = ToTensor(),
                                  download = True)

test_cifar = datasets.CIFAR10(root = "./test_cifar",
                                  train = True,
                                  transform = ToTensor(),
                                  download = True)








