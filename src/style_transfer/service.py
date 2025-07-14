import torch
import os

from torchvision.transforms import ToPILImage

from .model import get_vgg_model
from .trainer import run_style_transfer
from .utils import image_loader


def process_style_transfer(
    content_path: str,
    styles_dir: str,
    result_path: str,
    plot_path: str,
    resize_size=(512, 512),
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = image_loader(content_path, device, size=resize_size)

    style_imgs = []
    for fname in os.listdir(styles_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            style_imgs.append(
                image_loader(os.path.join(styles_dir, fname), device, size=resize_size)
            )
    if not style_imgs:
        raise RuntimeError("No style reference images found in styles directory!")
    input_img = content_img.clone()

    cnn = get_vgg_model().to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    output, _, _ = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_imgs,
        input_img,
        content_layers,
        style_layers,
        num_steps=500,
        style_weight=int(1e6),
        content_weight=1,
        plot_path=plot_path,
    )
    result_img = ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))
    result_img.save(result_path)
    return {"result": result_path, "plot": plot_path}
