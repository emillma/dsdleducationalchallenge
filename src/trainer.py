import torch
from torch import nn
from tqdm import trange


def train(adversary, optimizer, img_preproced, target):
    loss_fn = nn.CrossEntropyLoss()
    x = torch.unsqueeze(img_preproced, dim=0).cuda()
    y = target.cuda()
    losses = []
    for i in trange(100):
        optimizer.zero_grad()
        output = adversary(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        adversary.noise.data = torch.clamp(
            adversary.noise.data, min=-0.05, max=0.05)

        losses.append(loss.item())
    return losses
