import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
try:
    from torch.amp import autocast, GradScaler
    version_flag = True
except ImportError:
    print("torch.amp is not available, assume using torch_npu.npu.amp instead.")
    try:
        from torch_npu.npu.amp import autocast, GradScaler
        npu_flag = True
    except ImportError:
        print("torch_npu.npu.amp is not available, assume using torch.cuda.amp instead.")
        from torch.cuda.amp import autocast, GradScaler
        npu_flag = False
    version_flag = False


class ResNet50Bench(object):
    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10, use_fp16=False, use_bf16=False, monitor_performance=False):
        self.gpu_devices = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_size = data_size
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.monitor_performance = monitor_performance  # Optional: monitor batch-level performance

        if use_bf16:
            if torch.cuda.is_bf16_supported(including_emulation=False):
                self.use_bf16 = True
            else:
                print("BF16 is not supported, using FP16 instead.")
                self.use_bf16 = False
                self.use_fp16 = True

        self.train_dataset = FakeDataset(size=data_size, image_size=image_size, num_classes=num_classes)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def start(self):
        if self.gpu_devices is None:
            # print("GPU is not available, only CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            print("GPU is not available")
            # self._bench(self.cpu_device)
        else:
            # print("GPU is available, both GPU and CPU will be benched.")
            # print("DEBUG mode, skipping CPU bench.")
            print("DEBUG mode.")
            self._bench(self.gpu_devices)
            # self._bench(self.cpu_device)

    def _bench(self, devices):
        main_device = devices[0]
        model = ResNet50().to(main_device)

        if len(self.gpu_devices) > 1:
            model = nn.DataParallel(model, device_ids=[device.index for device in devices])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        if main_device.type in ["xpu", "npu"]:
            GS_dev = "cuda"
        else:
            GS_dev = main_device.type
        if main_device.type in ["npu"]:
            AC_dev = "cuda"
        else:
            AC_dev = main_device.type

        if self.use_bf16:
            # BF16 is not need GradScaler
            pass
        else:
            if version_flag:
                scaler = GradScaler(device=GS_dev, enabled=self.use_fp16)
            else:
                scaler = GradScaler(enabled=self.use_fp16)

        total_step = len(self.train_loader)
        pre_load_start = time.time()
        data_preloaded = [(images.to(main_device), labels.to(main_device)) for images, labels in self.train_loader]
        pre_load_end = time.time()
        print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        # Warmup: run a few iterations to stabilize GPU state (eliminate cold start effects)
        print("Warming up...")
        warmup_batches = min(5, len(data_preloaded))
        for i in range(warmup_batches):
            images, labels = data_preloaded[i]
            if self.use_bf16:
                if version_flag:
                    with autocast(device_type=AC_dev, dtype=torch.bfloat16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.bfloat16, enabled=True):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                if version_flag:
                    with autocast(device_type=AC_dev, dtype=torch.float16, enabled=self.use_fp16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    with autocast(dtype=torch.float16, enabled=self.use_fp16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()
        
        # Synchronize before starting the timer to ensure all previous operations are completed
        if main_device.type == 'cuda':
            torch.cuda.synchronize(main_device)
        elif main_device.type == 'xpu':
            torch.xpu.synchronize(main_device)
        elif main_device.type == 'npu':
            torch.npu.synchronize(main_device)
        elif main_device.type == 'mps':
            torch.mps.synchronize()
        
        print("Timer started.")
        
        start_time = time.time()
        batch_times = []  # Store batch times for optional performance monitoring
        sample_interval = max(1, len(data_preloaded) // 10)  # Sample ~10 batches per epoch
        
        for epoch in range(self.epochs):
            iters = len(self.train_loader)
            pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                # Optional: monitor performance for sampled batches (only if enabled)
                if self.monitor_performance and i % sample_interval == 0:
                    if main_device.type == 'cuda':
                        torch.cuda.synchronize(main_device)
                    elif main_device.type == 'xpu':
                        torch.xpu.synchronize(main_device)
                    elif main_device.type == 'npu':
                        torch.npu.synchronize(main_device)
                    elif main_device.type == 'mps':
                        torch.mps.synchronize()
                    batch_start = time.time()
                
                # images = images.to(device)
                # labels = labels.to(device)
                if self.use_bf16:
                    if version_flag:
                        with autocast(device_type=AC_dev, dtype=torch.bfloat16, enabled=True):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        with autocast(dtype=torch.bfloat16, enabled=True):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    if version_flag:
                        with autocast(device_type=AC_dev, dtype=torch.float16, enabled=self.use_fp16):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        with autocast(dtype=torch.float16, enabled=self.use_fp16):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad()
                
                # Optional: record batch time for sampled batches
                if self.monitor_performance and i % sample_interval == 0:
                    if main_device.type == 'cuda':
                        torch.cuda.synchronize(main_device)
                    elif main_device.type == 'xpu':
                        torch.xpu.synchronize(main_device)
                    elif main_device.type == 'npu':
                        torch.npu.synchronize(main_device)
                    elif main_device.type == 'mps':
                        torch.mps.synchronize()
                    batch_end = time.time()
                    batch_times.append(batch_end - batch_start)

                pbar.update(1)
                pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.detach().item():.4f}")

            pbar.close()
        
        # Synchronize after training to ensure all GPU operations are completed before stopping the timer
        if main_device.type == 'cuda':
            torch.cuda.synchronize(main_device)
        elif main_device.type == 'xpu':
            torch.xpu.synchronize(main_device)
        elif main_device.type == 'npu':
            torch.npu.synchronize(main_device)
        elif main_device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        time_usage = end_time - start_time
        basic_score = self.data_size / time_usage
        final_score = basic_score * (self.epochs / 10) * 100
        print(f"Training completed on {main_device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.0f}")
        
        # Optional: print performance statistics if monitoring is enabled
        if self.monitor_performance and batch_times:
            import numpy as np
            batch_times = np.array(batch_times)
            print(f"\nPerformance Statistics (sampled {len(batch_times)} batches):")
            print(f"  Mean batch time: {batch_times.mean():.4f}s")
            print(f"  Std deviation: {batch_times.std():.4f}s")
            print(f"  Min/Max: {batch_times.min():.4f}s / {batch_times.max():.4f}s")
            print(f"  Coefficient of Variation: {(batch_times.std() / batch_times.mean() * 100):.2f}%")


class FakeDataset(torch.utils.data.Dataset):
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


if __name__ == "__main__":
    from torchinfo import summary

    model = ResNet50()
    print(model)

    print("\n----------\n")

    t_batch_size = 1024
    summary(model, input_size=(t_batch_size, 3, 32, 32))
