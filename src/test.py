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
from trainer import train
from adversary import Attacker
from transforms import UnNormalize
plt.close('all')
project_folder = Path(__file__).parents[1]

preproc = transforms.Compose([
    transforms.Resize(320),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
imgs_preproc = []
for img_path in ["torstein.jpg", "emil.jpg", 'hammer.jpg']:
    with Image.open(project_folder.joinpath(f"data/{img_path}")) as img:
        img_preproced = preproc(img)
        imgs_preproc.append(img_preproced)

img_preproced = imgs_preproc[-1]


def freeze(torchitem):
    for param in torchitem.parameters():
        param.requires_grad = False


model = resnet50(pretrained=True)
freeze(model)
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


adversary = Attacker(model)
adversary.cuda()
target = torch.zeros((1), dtype=torch.long)
target[0] = 234
optimizer = torch.optim.Adam([adversary.noise],
                             lr=0.01, weight_decay=0)
x = torch.unsqueeze(img_preproced, dim=0).cuda()
y = target.cuda()
losses = train(adversary, optimizer, img_preproced, target)

adversary.eval()
with torch.no_grad():
    pred2 = adversary(x)
    pred2 = torch.nn.functional.softmax(pred2.cpu(), dim=1).numpy().ravel()
    print(np.argmax(pred2))
    print(np.amax(pred2))
    print(pred2[600])
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
with torch.no_grad():
    new_x = unorm(x+adversary.noise)
    print(new_x.shape)
    plt.imshow(np.transpose(new_x.cpu()[0], (1, 2, 0)))
fig, ax = plt.subplots()
ax.plot(losses)
plt.show()
