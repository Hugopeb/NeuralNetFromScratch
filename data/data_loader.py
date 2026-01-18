import torch
from torchvision import datasets 
from torchvision.transforms import ToTensor

def load_MNIST():
    train_MNIST = datasets.MNIST(
        root = "./train_MNIST",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_MNIST = datasets.MNIST(
        root = "./test_MNIST",
        train = False,
        transform = ToTensor(),
        download = True
    )

    return train_MNIST, test_MNIST


def load_CIFAR10():
    train_CIFAR10 = datasets.CIFAR10(
        root = "./train_CIFAR",
        train = True,
        transform = ToTensor(),
        download = True
    )

    test_CIFAR10 = datasets.CIFAR10(
        root = "./test_CIFAR",
        train = False,
        transform = ToTensor(),
        download = True
    )

    return train_CIFAR10, test_CIFAR10




