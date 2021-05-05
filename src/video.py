import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from matplotlib import pyplot as plt
import numpy as np


def toimg(tensor):
    with torch.no_grad():
        return np.transpose(tensor.cpu().numpy(), (1, 2, 0))


def imgnorm(img):
    return (img - img.min())/(img.max()-img.min())


denormalize = Compose(
    [Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
     Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), ])


img_denom = denormalize(x[0])
attacked_img_denom = denormalize(x[0] + attack)
attack_denom = attacked_img_denom - img_denom

fig = plt.figure(figsize=(16, 9), dpi=80)
ax0 = plt.subplot2grid((2, 3), (0, 0))
ax1 = plt.subplot2grid((2, 3), (0, 1))
ax2 = plt.subplot2grid((2, 3), (0, 2))
ax3 = plt.subplot2grid((2, 3), (1, 0))
ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
for ax in [ax0, ax1, ax2, ax3]:
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
plt.tight_layout()

ax0.imshow(toimg(img_denom))
ax1.imshow(imgnorm(toimg(attack_denom)))

ax4.set_ylim(0, 1)
