import torch
from torch import nn
from torchvision.models import resnet50


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


class Adversary(nn.Module):
    def __init__(self):
        self.resnet = resnet50(pretrained=True)
        freeze(self.resnet)

        self.noise = nn.Parameter(torch.zeros(4, 224, 224))

    def __call__(self, x):
        x_hacked = x + self.noise
        y_hat = self.resnet(x_hacked)
        return y_hat


def train(adversery: nn.Module, img, target):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.adam(adversery.parameters(), weight_decay=0.001)
    img = img.cuda()
    target = target.cuda()
    for i in range(100):
        optimizer.zero_grad()
        output = adversery(img)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
