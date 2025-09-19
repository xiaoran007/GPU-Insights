# GPU-Insights
GPU Insights, repository for all kinds of code related to GPUs

For better visual experience, visit [this](https://xiaoran007.github.io/GPU-Insights/) website.

## Default benchmark
```shell
python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32 -gpu 0
python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16 -gpu 0
```

### How to understand the results

This performance test evaluates the performance of the hardware device in a training scenario, and the output is a score. Score reflects the unit time to complete a given training task (ResNet50). Thus, a higher score means higher computational performance. Note that the score is affected by both the video memory bandwidth and the PCIe bus bandwidth.

### Model Params
23.5 M

### Test command

FP32: `python main.py -m -s 512 -e 3 -mt resnet50 -bs 256 -dt FP32`

FP16: `python main.py -m -s 512 -e 3 -mt resnet50 -bs 256 -dt FP16`

FP32 cudnn: `python main.py -m -s 512 -e 10 -mt resnet50 -bs 256 -dt FP32 -cudnn`

FP16 cudnn: `python main.py -m -s 512 -e 10 -mt resnet50 -bs 256 -dt FP16 -cudnn`

Note: In general, Multi-GPU test case using batch size 2048 `-bs 2048` for FP16 and batch size 1024 `-bs 1024` for FP32.

### Results

Most of the results are obtained by adjusting the batch size to get the maximum video memory usage.

#### Mthreads

|          Device          |    Platform     | FP32 | FP32BS | FP16 | FP16BS |    Note    | **Date**  |
| :----------------------: | :-------------: | :--: | :----: | :--: | :----: | :--------: | --------- |
| Mthreads S4000<br />48GB | Linux<br />5.15 | 9078 |  256   |  /   |  256   | MUSA 3.1.0 | 2025.5.19 |



#### AMD

|             Device             |    Platform     | FP32 | FP32BS | FP16 | FP16BS |    Note    | **Date**  |
| :----------------------------: | :-------------: | :--: | :----: | :--: | :----: | :--------: | --------- |
| Radeon Instinct MI50<br />32GB | Linux<br />5.15 | 2017 |  256   | 2951 |  256   | ROCm 6.2.4 | 2025.4.14 |



#### Huawei

|   Device    |         Platform         | FP32  | FP32BS | FP16  | FP16BS |              Note               | **Date**  |
| :---------: | :----------------------: | :---: | :----: | :---: | :----: | :-----------------------------: | --------- |
| Ascend910B2 | Linux (Docker)<br />5.15 | 18283 |  1024  | 55517 |  1024  | FP16 GradScaler seems overflow. | 2025.3.20 |



#### Apple

|    Device    |     Platform      | FP32 | FP32BS | FP16 | FP16BS |   Note   | Date      |
| :----------: | :---------------: | :--: | :----: | :--: | :----: | :------: | --------- |
| Apple M4 GPU | macOS<br />15.3.1 | 1723 |  128   | 1591 |  128   | 10 Cores | 2025.3.20 |
| Apple M1 GPU | macOS<br />15.3.1 | 948  |  128   | 843  |  128   | 8 Cores  | 2025.3.20 |



#### Nvidia Blackwell

|                           Device                            |          Platform           | FP32  | FP32BS | FP16  | FP16BS |      Note       | Date      |
| :---------------------------------------------------------: | :-------------------------: | :---: | :----: | :---: | :----: | :-------------: | --------- |
|              NVIDIA GeForce RTX 5090<br />32GB              | Linux (Docker)<br />570.124 | 37232 |  512   | 63230 |  512   | Preview PyTorch | 2025.4.19 |
| NVIDIA RTX PRO 6000 Blackwell Workstation Edition<br />96GB | Linux (Docker)<br />570.124 | 43201 |  256   | 78837 |  256   | Preview PyTorch | 2025.7.28 |



#### Nvidia Hopper

|        Device        |          Platform          | FP32  | FP32BS | FP16  | FP16BS | Note | Date      |
| :------------------: | :------------------------: | :---: | :----: | :---: | :----: | :--: | --------- |
| NVIDIA H20<br />96GB | Linux (Docker)<br />565.57 | 37360 |  256   | 55222 |  256   |  /   | 2025.3.27 |



#### Nvidia Ada

|              Device               |          Platform          | FP32  | FP32BS |  FP16  | FP16BS |               Note                | Date      |
| :-------------------------------: | :------------------------: | :---: | :----: | :----: | :----: | :-------------------------------: | --------- |
| NVIDIA GeForce RTX 4090<br />24GB |     Linux<br />560.35      | 24046 |  512   | 43733  |  1024  |                 /                 | 2025.3.20 |
|       NVIDIA vGPU<br />48GB       | Linux (Docker)<br />575.64 | 24300 |  256   | 48337  |  256   |    Two NVIDIA GeForce RTX 4090    | 2025.9.04 |
|       NVIDIA vGPU<br />32GB       | Linux (Docker)<br />560.35 | 17308 |  1024  | 33238  |  256   | Two NVIDIA GeForce RTX 4080 SUPER | 2025.3.27 |
|  NVIDIA vGPU<br />32GB    2 GPUs  | Linux (Docker)<br />560.35 | 30275 |  2048  | 52756  |  4096  | Two NVIDIA GeForce RTX 4080 SUPER | 2025.3.20 |
|  NVIDIA vGPU<br />32GB    4 GPUs  | Linux (Docker)<br />560.35 | 56178 |  4096  | 101268 |  8192  | Two NVIDIA GeForce RTX 4080 SUPER | 2025.3.20 |



#### Nvidia Ampere

|                   Device                    |          Platform          | FP32  | FP32BS | FP16  | FP16BS |               Note                | Date      |
| :-----------------------------------------: | :------------------------: | :---: | :----: | :---: | :----: | :-------------------------------: | --------- |
|         NVIDIA A100-PCIE<br />40GB          |     Linux<br />550.90      | 27478 |  256   | 43802 |  256   |                 /                 | 2025.3.31 |
|      NVIDIA GeForce RTX 3090<br />24GB      |    Windows<br />566.14     | 16311 |  256   | 28197 |  256   |                 /                 | 2025.3.10 |
|         NVIDIA RTX A5000<br />24GB          |     Linux<br />535.183     | 15090 |  512   | 27155 |  1024  |                 /                 | 2025.3.20 |
|    NVIDIA RTX A5000<br />24GB    2 GPUs     |     Linux<br />535.183     | 26962 |  1024  | 49930 |  3072  |              NVLink               | 2025.3.20 |
|      NVIDIA GeForce RTX 3080<br />20GB      | Linux (Docker)<br />560.35 | 13320 |  256   | 24205 |  256   | Unofficial Video Memory Expansion | 2025.3.20 |
| NVIDIA GeForce RTX 3080<br />20GB    2 GPUs | Linux (Docker)<br />560.35 | 23261 |  1024  | 40250 |  2048  | Unofficial Video Memory Expansion | 2025.3.20 |



#### Nvidia Turing

|                Device                |      Platform      | FP32 | FP32BS | FP16  | FP16BS |               Note                | Date      |
| :----------------------------------: | :----------------: | :--: | :----: | :---: | :----: | :-------------------------------: | --------- |
| NVIDIA GeForce RTX 2080 Ti<br />22GB | Linux<br />550.120 | 8754 |  256   | 19007 |  1024  | Unofficial Video Memory Expansion | 2025.3.20 |
| NVIDIA GeForce RTX 2080 Ti<br />11GB | Linux<br />550.120 | 8681 |  256   | 19475 |  256   |                 /                 | 2025.4.02 |



#### Nvidia Volta

|           Device           |          Platform          | FP32  | FP32BS | FP16  | FP16BS | Note | Date      |
| :------------------------: | :------------------------: | :---: | :----: | :---: | :----: | :--: | --------- |
| Tesla V100S-PCIE<br />32GB | Linux (Docker)<br />550.90 | 11577 |  256   | 27963 |  256   |  /   | 2025.3.20 |



#### Nvidia Pascal

|              Device               |          Platform           | FP32 | FP32BS | FP16 | FP16BS |                     Note                     | Date      |
| :-------------------------------: | :-------------------------: | :--: | :----: | :--: | :----: | :------------------------------------------: | --------- |
| NVIDIA TITAN X (Pascal)<br />12GB |     Windows<br />566.14     | 5792 |  256   | 7230 |  256   | FP16 Not Officially Supported By Pascal Arch | 2025.3.20 |
|     NVIDIA TITAN Xp<br />12GB     | Linux (Docker)<br />550.120 | 6792 |  256   | 7641 |  256   | FP16 Not Officially Supported By Pascal Arch | 2025.3.26 |
|     NVIDIA P104-100<br />8GB      |     Windows<br />536.23     | 4114 |  256   | 4766 |  256   |               PCIE 3.0 x4 180W               | 2025.4.05 |
|     NVIDIA P104-100<br />8GB      |     Windows<br />576.88     | 3745 |  256   | 4278 |  256   |               PCIE 3.0 x4 120W               | 2025.7.07 |



#### Nvidia Maxwell

| Device | Platform | FP32 | FP32BS | FP16 | FP16BS | Note | Date |
| :----: | :------: | :--: | :----: | :--: | :----: | :--: | ---- |
|   /    |    /     |  /   |   /    |  /   |   /    |  /   | /    |



#### Nvidia Kepler

| Device | Platform | FP32 | FP32BS | FP16 | FP16BS | Note | Date |
| :----: | :------: | :--: | :----: | :--: | :----: | :--: | ---- |
|   /    |    /     |  /   |   /    |  /   |   /    |  /   | /    |



#### Intel

|                  Device                  |      Platform      | FP32 | FP32BS | FP16 | FP16BS |        Note         | Date      |
| :--------------------------------------: | :----------------: | :--: | :----: | :--: | :----: | :-----------------: | --------- |
| Intel(R) Arc(TM) A770 Graphics<br />16GB | Linux<br />i915 xe | 5121 |  256   | 8049 |  256   | GradScaler Not Work | 2025.3.20 |
