import numpy as np

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms


class STL10DataModule:
    """Unified data module for all training paradigms"""

    def __init__(self, data_dir='./datasets', batch_size=256, num_workers=4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_ssl_dataloaders(self, ssl_method='simclr'):
        """Returns dataloaders for self-supervised pretraining"""
        if ssl_method in ['simclr', 'moco', 'byol']:
            transform = TwoViewTransform(self._get_ssl_augmentation())
        elif ssl_method == 'mae':
            transform = self._get_mae_transform()

        # Use unlabeled split (100k images)
        train_dataset = datasets.STL10(
            root=self.data_dir,
            split='unlabeled',
            transform=transform,
            download=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        return train_loader

    def _sample_k_per_class(self, dataset, k):
        labels = np.array(dataset.labels)
        indices = []

        for cls in range(10):  # STL-10 has 10 classes
            cls_idx = np.where(labels == cls)[0]

            if len(cls_idx) < k:
                print(f"[Warning] Class {cls}: only {len(cls_idx)} available, using all.")
                k_cls = len(cls_idx)
            else:
                k_cls = k

            selected = np.random.choice(cls_idx, k_cls, replace=False)
            indices.extend(selected.tolist())

        print(f"Total supervised samples: {len(indices)}")
        return Subset(dataset, indices)

    def get_supervised_dataloaders(self, samples_per_class=None):
        """
        samples_per_class: int or None
            If None -> use full STL-10 training set
            If int -> sample exactly that many images per class
        """

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066],
                std=[0.2603, 0.2566, 0.2713]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066],
                std=[0.2603, 0.2566, 0.2713]
            )
        ])

        train_dataset = datasets.STL10(
            root=self.data_dir,
            split='train',
            transform=train_transform,
            download=True
        )

        if samples_per_class is not None:
            train_dataset = self._sample_k_per_class(train_dataset, samples_per_class)

        test_dataset = datasets.STL10(
            root=self.data_dir,
            split='test',
            transform=test_transform,
            download=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader

    def _get_ssl_augmentation(self):
        """Strong augmentation for contrastive methods"""
        return transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(9, (0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066],
                std=[0.2603, 0.2566, 0.2713]
            )
        ])

    def _get_mae_transform(self):
        """Minimal augmentation for MAE"""
        return transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.67, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4467, 0.4398, 0.4066],
                std=[0.2603, 0.2566, 0.2713]
            )
        ])


class TwoViewTransform:
    """Create two augmented views of the same image"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]