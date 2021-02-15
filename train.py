import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F

def train(model, train_loader, optimizer, epochs, DEVICE):
    model.train()
    for epoch in range(epochs):
        print("{}/{} Epochs".format(epoch + 1, epochs))
        for x, _ in tqdm(train_loader):
            x = x.to(DEVICE)
            pred = model(x)
            reconstruction = F.binary_cross_entropy(pred, x)
            regularization = torch.mean(model.mu ** 2 + torch.exp(model.log_var) - 1. - model.log_var)
            loss = reconstruction + regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss.item())