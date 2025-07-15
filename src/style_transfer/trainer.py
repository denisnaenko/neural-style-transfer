import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from .losses import ContentLoss, StyleLoss
from .model import Normalization


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_imgs,
    content_img,
    content_layers,
    style_layers,
):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    import torch.nn as nn

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_features = [model(style_img).detach() for style_img in style_imgs]
            style_loss = StyleLoss(target_features)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[: (j + 1)]
    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_imgs,
    input_img,
    content_layers,
    style_layers,
    num_steps=300,
    style_weight=1e6,
    content_weight=1.0,
    plot_path=None,
):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_imgs,
        content_img,
        content_layers,
        style_layers,
    )
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)
    style_history = []
    content_history = []
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = torch.tensor(0.0, device=input_img.device)
            content_score = torch.tensor(0.0, device=input_img.device)
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            style_history.append(style_score.item())
            content_history.append(content_score.item())
            run[0] += 1
            print(
                f"Step {run[0]}/{num_steps} | Style Loss: {style_score.item():.2f} | Content Loss: {content_score.item():.2f}"
            )
            return loss

        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    if plot_path:
        plt.figure()
        plt.plot(style_history, label="Style Loss")
        plt.plot(content_history, label="Content Loss")
        plt.legend()
        plt.title("Losses during style transfer")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(plot_path)
        plt.close()
    return input_img, style_history, content_history
