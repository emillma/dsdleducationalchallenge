import torch
from torch import nn


class Attacker(nn.Module):
    def __init__(self, victim_model, input_shape=(3, 320, 320)):
        super().__init__()
        self.victim_model = victim_model
        self.noise = nn.Parameter(torch.zeros(input_shape))

    def __call__(self, x):
        x_attacked = x + self.noise
        y_hat = self.victim_model(x_attacked)
        return y_hat
