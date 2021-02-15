"""
    model 구조를 구성하겠습니다. <chain>을 이용한 파라미터 묶기
"""

import torch.nn as nn
from itertools import chain


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(7 * 7 * 128, 128)
        self.fc2 = nn.Linear(7 * 7 * 128, 128)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 7 * 7 * 128)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 7 * 7 * 128),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 128, 7, 7)
        x = self.main(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def parameters(self):  # VAE로 전체 파라미터를 묶어주자 !
        return chain(self.encoder.parameters(), self.decoder.parameters())

    def calc_latent(self, mu, log_var):
        eps = torch.randn(size=mu.size()).to(DEVICE, dtype=torch.float)
        return mu + eps * torch.exp(log_var)

    def forward(self, x):
        self.mu, self.log_var = self.encoder(x)
        latent = self.calc_latent(self.mu, self.log_var)
        x = self.decoder(latent)
        return x
