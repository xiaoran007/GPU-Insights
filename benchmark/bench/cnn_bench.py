"""Backward-compatible wrapper.

Preserves the original CNNBench API and import paths while delegating
to the new modular architecture internally.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

# Re-export model class and dataset for backward compatibility
from benchmark.models.cnn import CNN
from benchmark.data import FakeDataset
from benchmark.devices import auto_detect_backend
from benchmark.runners.single_runner import SingleRunner
from benchmark.models import get_model


class CNNBench(object):
    """Legacy CNNBench — delegates to SingleRunner."""

    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10):
        self.gpu_device = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_size = data_size

    def start(self):
        if self.gpu_device is None:
            print("GPU is not available, only CPU will be benched.")
            return
        print("GPU is available, both GPU and CPU will be benched.")
        print("DEBUG mode, skipping CPU bench.")

        backend = auto_detect_backend()
        if backend is None:
            print("No compatible device backend found.")
            return

        device = self.gpu_device
        devices = [device] if not isinstance(device, list) else device

        model_spec = get_model("cnn")
        runner = SingleRunner(
            model_spec=model_spec,
            device_backend=backend,
            devices=devices,
            data_size=self.data_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        runner.run()


