import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

preprocess = Compose([Resize(320), ToTensor(),
                      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

with Image.open("../data/hammer.jpg") as img:
    x = torch.unsqueeze(preprocess(img), dim=0).cuda()

model = resnet50(pretrained=True).cuda().eval()

attack = torch.zeros((3, 320, 320), requires_grad=True, device='cuda')
alpha = 50

attacks = []
grads = []

pred = torch.nn.functional.softmax(model(x), dim=1)
dogs = []
hammers = []

target = torch.tensor([300], dtype=torch.long).cuda()
criterion = torch.nn.CrossEntropyLoss()
for i in range(100):

    y_hat = model(x+attack)
    criterion(y_hat, target).backward()

    pred = torch.nn.functional.softmax(y_hat, dim=1)
    attacks.append(attack.data.cpu())
    dogs.append(pred[0, 300].item())
    hammers.append(pred[0, 587].item())
    grads.append(attack.grad.data.cpu())

    attack.data -= alpha*attack.grad
    attack.data = torch.clamp(attack.data, -0.02, 0.02)

    print(torch.nn.functional.softmax(y_hat, dim=1)[0, 300])
