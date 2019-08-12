"""
Created on Tue Aug 8 2019
@author: HyunsuKim6(Github), hyunsukim@kaist.ac.kr
"""

import torch
import torchvision
from torchvision import transforms


def load_data(purpose, batch_size=10, num_workers=12):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if purpose == 'train':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)

    elif purpose == 'test':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

    else:
        print("Incorrect input: Please enter correct purpose input")
        return

    return dataloader
