from torch import nn
import torch
from torchvision.models import resnet50
import cv2
from torchvision import transforms
from matplotlib import image, pyplot as plt
from pathlib import Path
from PIL import Image
import json
import re
import numpy as np
from tqdm import trange
plt.close('all')
project_folder = Path(__file__).parents[1]

preproc = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
with Image.open(project_folder.joinpath('data/ambulance.jpg')) as img:
    img_preproced = preproc(img)


model = resnet50(pretrained=True)
model.cuda()
model.eval()
with torch.no_grad():
    y = model(torch.unsqueeze(img_preproced, dim=0).cuda()).cpu()
    pred = torch.nn.functional.softmax(y, dim=1)
pred = pred.numpy().ravel()

with open(project_folder.joinpath('data/classes.txt')) as file:
    classes = [line.strip() for line in file.readlines()[1:-1]]
    classes = [txt[txt.find(':')+3:-2] for txt in classes]

best = np.argsort(pred)[::-1]
[classes[i] for i in best[0:5]]
# pred[best[0:5]]


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Adversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        freeze(self.resnet)

        self.noise = nn.Parameter(torch.zeros(3, 224, 224))

    def __call__(self, x):
        x_hacked = x + self.noise
        y_hat = self.resnet(x_hacked)
        return y_hat


adversary = Adversary()
adversary.cuda()
target = torch.zeros((1), dtype=torch.long)
target[0] = 224
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(adversary.parameters(),
                             lr=0.1, weight_decay=0)
x = torch.unsqueeze(img_preproced, dim=0).cuda()
y = target.cuda()
top = torch.tensor([0.1]).cuda()
bottom = torch.tensor([-0.1]).cuda()
for i in trange(100):
    optimizer.zero_grad()
    output = adversary(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    adversary.noise.data = torch.minimum(adversary.noise, top)
    adversary.noise.data = torch.maximum(adversary.noise, bottom)

adversary.eval()
with torch.no_grad():
    pred2 = adversary(x)
    pred2 = torch.nn.functional.softmax(pred2.cpu(), dim=1).numpy().ravel()
    print(np.argmax(pred2))
    print(np.amax(pred2))
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
with torch.no_grad():
    new_x = unorm(x+adversary.noise)
    print(new_x.shape)
    plt.imshow(np.transpose(new_x.cpu()[0], (1, 2, 0)))
