import torch
from torch import nn
from tqdm import trange


def train(adversary, optimizer, img_preproced, target):
    weights = torch.ones(1000)*1e-9
    weights[234] = 1e6
    weights[600] = 1e6
    loss_fn = nn.CrossEntropyLoss(weights.cuda())

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
