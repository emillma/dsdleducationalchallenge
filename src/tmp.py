import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

preprocess = Compose([Resize(320), ToTensor(),
                      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

with Image.open("data/hammer.jpg") as img:
    image = torch.unsqueeze(preprocess(img), dim=0).cuda()

model = resnet50(pretrained=True).cuda().eval()
attack = torch.zeros(image.shape, requires_grad=True, device='cuda')

costfunction = torch.nn.CrossEntropyLoss()
target = torch.tensor([300], dtype=torch.long).cuda()

alpha = 0.1
for i in range(500):
    output = model(image + attack)
    costfunction(output, target).backward()

    d_cost_d_attack = attack.grad
    attack.data -= alpha * d_cost_d_attack
    # attack.data = torch.clamp(attack.data, -0.02, 0.02)

print(torch.nn.functional.softmax(output, dim=1)[0, 300])
