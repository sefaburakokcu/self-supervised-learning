import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


def to_tensor_only():
    # No normalization → better visualization
    return transforms.ToTensor()


def get_simclr_augmentations():
    return transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5
        ),
        transforms.ToTensor()
    ])


def build_individual_augmentations():
    return {
        "Original": transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ]),

        "RandomResizedCrop": transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.ToTensor()
        ]),

        "HorizontalFlip": transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor()
        ]),

        "ColorJitter": transforms.Compose([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.ToTensor()
        ]),

        "RandomGray": transforms.Compose([
            transforms.RandomGrayscale(p=1.0),
            transforms.ToTensor()
        ]),

        "GaussianBlur": transforms.Compose([
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor()
        ]),

       # "SimCLR Full Transform": get_simclr_augmentations(),
    }


def show_augmented_images(image, augmentations, save_path="simclr_augmentations.png"):
    n = len(augmentations)
    cols = 3
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 4 * rows))

    for i, (name, transform) in enumerate(augmentations.items()):
        aug = transform(image).permute(1, 2, 0).numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.clip(aug, 0, 1))
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    stl = datasets.STL10(root="../datasets", split="train", download=True)

    idx = random.randint(0, len(stl) - 1)
    image, label = stl[idx]

    print(f"Visualizing STL-10 image #{idx}, class {label}")

    augmentations = build_individual_augmentations()
    show_augmented_images(image, augmentations)
