import os
import shutil
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.transforms import ToPILImage

from style_transfer.utils import image_loader


def run_and_log_experiment(
    content_path,
    styles_dir,
    result_dir,
    resize_size=(512, 512),
    style_weight=1e6,  # float
    content_weight=1,
    num_steps=300,
    note=None,
):
    os.makedirs(result_dir, exist_ok=True)

    shutil.copy(content_path, os.path.join(result_dir, "content.jpg"))

    from style_transfer.model import get_vgg_model
    from style_transfer.trainer import run_style_transfer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = image_loader(content_path, device, size=resize_size)
    style_imgs = []

    for fname in os.listdir(styles_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            style_imgs.append(
                image_loader(os.path.join(styles_dir, fname), device, size=resize_size)
            )
    input_img = content_img.clone()

    cnn = get_vgg_model().to(device)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    start = time.time()
    output, style_history, content_history = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_imgs,
        input_img,
        content_layers,
        style_layers,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        plot_path=os.path.join(result_dir, "loss_plot.png"),
    )
    elapsed = time.time() - start
    result_img = ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))
    result_img.save(os.path.join(result_dir, "result.jpg"))

    # Логируем параметры и метрики
    with open(os.path.join(result_dir, "params.txt"), "w") as f:
        f.write(
            f"resize_size={resize_size}\nstyle_weight={style_weight}\ncontent_weight={content_weight}\nnum_steps={num_steps}\nstyles={os.listdir(styles_dir)}\nnote={note}\n"
        )
        f.write(
            f"elapsed={elapsed:.2f}\nfinal_style_loss={style_history[-1] if style_history else None}\nfinal_content_loss={content_history[-1] if content_history else None}\nmin_style_loss={min(style_history) if style_history else None}\nmin_content_loss={min(content_history) if content_history else None}\n"
        )

    return {
        "style_weight": style_weight,
        "style_history": style_history,
        "content_history": content_history,
        "elapsed": elapsed,
        "final_style_loss": style_history[-1] if style_history else None,
        "final_content_loss": content_history[-1] if content_history else None,
        "min_style_loss": min(style_history) if style_history else None,
        "min_content_loss": min(content_history) if content_history else None,
        "result_path": os.path.join(result_dir, "result.jpg"),
        "note": note,
    }


if __name__ == "__main__":
    # Research: анализ влияния style_weight
    content_path = "sources/miyazaki.jpg"
    styles_dir = "styles"
    research_dir = "../experiments/research_styleweight"
    os.makedirs(research_dir, exist_ok=True)

    style_weights = [1e3, 1e4, 1e5, 1e6]
    results = []

    for style_weight in style_weights:
        result_dir = os.path.join(research_dir, f"exp_styleweight_{int(style_weight)}")
        res = run_and_log_experiment(
            content_path=content_path,
            styles_dir=styles_dir,
            result_dir=result_dir,
            style_weight=style_weight,  # float
            note=f"style_weight={style_weight}",
        )
        results.append(res)

    # Построение графиков
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res["style_history"], label=f"style_weight={int(res['style_weight'])}")
    plt.title("Style Loss curves for different style_weight")
    plt.xlabel("Step")
    plt.ylabel("Style Loss")
    plt.legend()
    plt.savefig(os.path.join(research_dir, "style_loss_curves.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(
            res["content_history"], label=f"style_weight={int(res['style_weight'])}"
        )
    plt.title("Content Loss curves for different style_weight")
    plt.xlabel("Step")
    plt.ylabel("Content Loss")
    plt.legend()
    plt.savefig(os.path.join(research_dir, "content_loss_curves.png"))
    plt.close()

    # Время работы
    plt.figure(figsize=(8, 5))
    plt.bar(
        [str(int(res["style_weight"])) for res in results],
        [res["elapsed"] for res in results],
    )
    plt.title("Elapsed time for different style_weight")
    plt.xlabel("style_weight")
    plt.ylabel("Seconds")
    plt.savefig(os.path.join(research_dir, "elapsed_time.png"))
    plt.close()

    # Финальные style loss
    plt.figure(figsize=(8, 5))
    plt.bar(
        [str(int(res["style_weight"])) for res in results],
        [res["final_style_loss"] for res in results],
    )
    plt.title("Final Style Loss for different style_weight")
    plt.xlabel("style_weight")
    plt.ylabel("Final Style Loss")
    plt.savefig(os.path.join(research_dir, "final_style_loss.png"))
    plt.close()

    # Сохраняем summary в CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(research_dir, "summary.csv"), index=False)
    print(f"Research finished. All results in {research_dir}")
