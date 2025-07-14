import torch
import torchvision.transforms as transforms
from PIL import Image

# size of the output image
imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose(
    [
        transforms.Resize(imsize),
        transforms.ToTensor(),
    ]
)

unloader = transforms.ToPILImage()


def image_loader(image_name, device, size=None):
    image = Image.open(image_name)
    if size is not None:
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 0))
        image = image.resize(size, resample)
        tensor = transforms.ToTensor()(image)
    else:
        tensor = loader(image)
    tensor = tensor.to(device, torch.float)  # type: ignore
    tensor = tensor.unsqueeze(0)
    return tensor
