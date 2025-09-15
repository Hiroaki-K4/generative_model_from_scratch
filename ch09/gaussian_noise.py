import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms


def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "flower.png")
    image = plt.imread(file_path)
    print(image.shape)

    preprocess = transforms.ToTensor()
    x = preprocess(image)
    print(x.shape)

    T = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T)

    t = 100
    x_t = add_noise(x, t, betas)

    img = reverse_to_img(x_t)
    plt.imshow(img)
    plt.title(f"Noise: {t}")
    plt.show()


if __name__ == "__main__":
    main()
