import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from matplotlib import pyplot as plt
import numpy as np
import pickle
from matplotlib.animation import FuncAnimation, FFMpegWriter

plt.close('all')


def toimg(tensor):
    with torch.no_grad():
        return np.transpose(tensor.cpu().numpy(), (1, 2, 0))


def imgnorm(img):
    return (img - img.min())/(img.max()-img.min())


denormalize = Compose(
    [Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
     Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), ])

with open('stuff.pickle', 'rb') as file:
    x, attacks, grads, dogs, hammers = pickle.load(file)

img_denom = denormalize(x[0])
gain = 10
# attacked_img_denom = denormalize(x[0] + attack)
# attack_denom = attacked_img_denom - img_denom

fig = plt.figure(figsize=(16, 9), dpi=120)

ax0 = plt.subplot2grid((2, 3), (0, 0))
ax0.set_title('Original image')

ax1 = plt.subplot2grid((2, 3), (0, 1))
ax1.set_title(f'{gain}*(Attack pattern)')

ax2 = plt.subplot2grid((2, 3), (0, 2))
ax2.set_title(f'{gain}*(Attack gradient)')

ax3 = plt.subplot2grid((2, 3), (1, 0))
ax3.set_title('Original image + attack pattern')

ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
ax4.set_title('Prediction')
ax4.set_ylim(-0.1, 1.05)
ax4.set_xlim(-0.2, 20)

for ax in [ax0, ax1, ax2, ax3]:
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

step = 0
orig_imshow = ax0.imshow(toimg(img_denom))
attack_imshow = ax1.imshow(toimg(denormalize(attacks[step]*gain)))
grad_imshow = ax2.imshow(toimg(denormalize(grads[step]*gain)))
sum_imshow = ax3.imshow(toimg(denormalize(x[0].cpu()+attacks[step])))
dog_plot = ax4.plot(dogs[:step+1], '-ro', markevery=[0], label='dog')[0]
hammer_plot = ax4.plot(hammers[:step+1], '-go', label='hammer',
                       markevery=[0])[0]
ax4.legend()


def update(step):
    attack_imshow.set_array(toimg(denormalize(attacks[step]*gain)))
    grad_imshow.set_array(toimg(denormalize(grads[step]*gain)))
    sum_imshow.set_array(toimg(denormalize(x[0].cpu()+attacks[step])))

    dog_plot.set_data(np.arange(step+1), dogs[:step+1])
    dog_plot.set_markevery([step])
    hammer_plot.set_data(np.arange(step+1), hammers[:step+1])
    hammer_plot.set_markevery([step])
    ax4.set_xlim(-0.2, max(20, step+1))


anim = FuncAnimation(fig, update, range(200), blit=False)
# plt.show()
# anim.save('image.gif', fps=10, writer="avconv", codec="libx264")
FFwriter = FFMpegWriter(fps=10)
anim.save('clamp.mp4', writer=FFwriter)
