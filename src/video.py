import torch
from torch import nn


class Attacker(nn.Module):
    def __init__(self, victim_model, input_shape=(3, 320, 320)):
        super().__init__()
        self.victim_model = victim_model
        self.noise = nn.Parameter(torch.zeros(input_shape))
