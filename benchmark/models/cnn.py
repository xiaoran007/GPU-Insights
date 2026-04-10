import torch.nn as nn
import torch

from benchmark.models.base import BenchModel


class CNNModel(BenchModel):
    """Simple CNN model for basic GPU benchmarking."""

    @property
    def name(self) -> str:
        return "cnn"

    def get_model_aliases(self):
        return ["CNN"]

    @property
    def supports_ddp(self) -> bool:
        return False

    @property
    def supports_amp(self) -> bool:
        return False

    @property
    def supports_compile(self) -> bool:
        return False

    @property
    def use_channels_last(self) -> bool:
        return True

    def create_model(self, num_classes: int = 10) -> nn.Module:
        return CNN(num_classes=num_classes)

    def get_image_size(self):
        return (3, 32, 32)

    def get_default_batch_size(self, data_type: str = "FP32") -> int:
        return 2048


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
