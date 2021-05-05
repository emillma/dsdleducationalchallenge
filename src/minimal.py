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
alpha = 75


class Emil:
    def __init__(self) -> None:
        pass
        self.foo = lambda x: x**2


for i in range(1000):
    y = torch.nn.functional.softmax(model(x+attack), dim=1)
    y[0, 300].backward()

    attack.data += alpha*attack.grad
    attack.data = torch.clamp(attack.data, -0.02, 0.02)

print(y[0, 300])
