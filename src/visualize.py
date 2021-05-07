import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

preprocess = Compose([Resize(320), ToTensor(),
                      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

with Image.open("data/hammer.jpg") as img:
    image = torch.unsqueeze(preprocess(img), dim=0).cuda()

model = resnet50(pretrained=True).cuda().eval()

pred = torch.nn.functional.softmax(model(image), dim=1)
print(f'Initial estimate: class={pred.argmax()}, certainty={pred.max()}')

attack = torch.zeros(image.shape, requires_grad=True, device='cuda')

costfunction = torch.nn.CrossEntropyLoss()
target = torch.tensor([300], dtype=torch.long).cuda()

attacks = []
grads = []
dogs = []
hammers = []

alpha = 0.01
for i in range(500):
    output = model(image + attack)
    costfunction(output, target).backward()

    pred = torch.nn.functional.softmax(output, dim=1)
    attacks.append(attack[0].data.cpu())
    dogs.append(pred[0, 300].item())
    hammers.append(pred[0, 587].item())
    grads.append(attack.grad.data.cpu()[0])

    d_cost_d_attack = attack.grad
    attack.data -= alpha * d_cost_d_attack
    attack.data = torch.clamp(attack.data, -0.02, 0.02)


pred = torch.nn.functional.softmax(output, dim=1)
print(f'Final estimate: class={pred.argmax()}, certainty={pred.max()}')


def toimg(tensor):
    with torch.no_grad():
        return np.transpose(tensor.cpu().numpy(), (1, 2, 0))


def imgnorm(img):
    return (img - img.min())/(img.max()-img.min())


denormalize = Compose(
    [Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
     Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]), ])


img_denom = denormalize(image[0])
gain = 10


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
sum_imshow = ax3.imshow(toimg(denormalize(image[0].cpu()+attacks[step])))
dog_plot = ax4.plot(dogs[:step+1], '-ro', markevery=[0], label='dog')[0]
hammer_plot = ax4.plot(hammers[:step+1], '-go', label='hammer',
                       markevery=[0])[0]
ax4.legend()


def update(step):
    attack_imshow.set_array(toimg(denormalize(attacks[step]*gain)))
    grad_imshow.set_array(toimg(denormalize(grads[step]*gain)))
    sum_imshow.set_array(toimg(denormalize(image[0].cpu()+attacks[step])))

    dog_plot.set_data(np.arange(step+1), dogs[:step+1])
    dog_plot.set_markevery([step])
    hammer_plot.set_data(np.arange(step+1), hammers[:step+1])
    hammer_plot.set_markevery([step])
    ax4.set_xlim(-0.2, max(20, step+1))


anim = FuncAnimation(fig, update, range(200), blit=False)
plt.show()
