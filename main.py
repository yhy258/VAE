"""
    MNIST 손글씨 데이터를 불러오겠습니다.
"""

import torch
from torchvision import datasets, transforms
from model import VAE
from train import train

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root = './data/', train = True, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)

"""
    가용한 GPU 가 있는지 확인해보고, 적용해보겠습니다.
"""
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(DEVICE) # 현재 device -> cuda or cpu

"""
    모델 선언, optimizer 선언
"""
vae = VAE().to(DEVICE)
optimizer = torch.optim.Adam(vae.parameters())
print(vae)

epochs =20
train(vae, train_loader, optimizer,epochs, DEVICE)

