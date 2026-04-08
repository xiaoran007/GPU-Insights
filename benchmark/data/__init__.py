import torch


class FakeDataset(torch.utils.data.Dataset):
    """Synthetic classification dataset: (image, class_label)."""

    def __init__(self, size=1000, image_size=(3, 32, 32), num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = torch.randn(size, *image_size)
        self.labels = torch.randint(0, num_classes, (size,))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.size


class FakeSegmentationDataset(torch.utils.data.Dataset):
    """Synthetic segmentation dataset: (image, pixel-wise label mask)."""

    def __init__(self, size=1000, image_size=(3, 256, 256), num_classes=21):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = torch.randn(size, *image_size)
        H, W = image_size[1], image_size[2]
        self.labels = torch.randint(0, num_classes, (size, H, W))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.size


class FakeImageOnlyDataset(torch.utils.data.Dataset):
    """Synthetic image-only dataset for generative models: (image, dummy_label).

    Returns a dummy label of 0 so the DataLoader signature stays uniform.
    """

    def __init__(self, size=1000, image_size=(3, 64, 64)):
        self.size = size
        self.image_size = image_size
        self.data = torch.randn(size, *image_size)

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)

    def __len__(self):
        return self.size
