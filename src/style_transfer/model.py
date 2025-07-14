import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_vgg_model():
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    return cnn
