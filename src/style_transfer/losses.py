import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        # target_features — список feature-тензоров
        self.targets = [gram_matrix(f).detach() for f in target_features]

    def forward(self, input):
        G = gram_matrix(input)
        # Среднее MSE по всем style-референсам
        self.loss = sum(F.mse_loss(G, t) for t in self.targets) / len(self.targets)
        return input
