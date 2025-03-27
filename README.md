# GPU-Insights
GPU Insights, repository for all kinds of code related to GPUs

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

#### Huawei

|   Device    |         Platform         | FP32  | FP32BS | FP16  | FP16BS |              Note               |
| :---------: | :----------------------: | :---: | :----: | :---: | :----: | :-----------------------------: |
| Ascend910B2 | Linux (Docker)<br />5.15 | 18283 |  1024  | 55517 |  1024  | FP16 GradScaler seems overflow. |



#### Apple

|    Device    |     Platform      | FP32 | FP32BS | FP16 | FP16BS |   Note   |
| :----------: | :---------------: | :--: | :----: | :--: | :----: | :------: |
| Apple M4 GPU | macOS<br />15.3.1 | 1723 |  128   | 1591 |  128   | 10 Cores |
| Apple M1 GPU | macOS<br />15.3.1 | 948  |  128   | 843  |  128   | 8 Cores  |



#### Nvidia Ada

|            Device            |          Platform          | FP32  | FP32BS |  FP16  | FP16BS |               Note                |
| :--------------------------: | :------------------------: | :---: | :----: | :----: | :----: | :-------------------------------: |
| NVIDIA GeForce RTX 4090 24GB |     Linux<br />560.35      | 24046 |  512   | 43733  |  1024  |                 /                 |
|       NVIDIA vGPU-32GB       | Linux (Docker)<br />560.35 | 16050 |  1024  | 28155  |  2048  | Two NVIDIA GeForce RTX 4080 SUPER |
|  NVIDIA vGPU-32GB    2 GPUs  | Linux (Docker)<br />560.35 | 30275 |  2048  | 52756  |  4096  | Two NVIDIA GeForce RTX 4080 SUPER |
|  NVIDIA vGPU-32GB    4 GPUs  | Linux (Docker)<br />560.35 | 56178 |  4096  | 101268 |  8192  | Two NVIDIA GeForce RTX 4080 SUPER |



#### Nvidia Ampere

|                 Device                 |          Platform          | FP32  | FP32BS | FP16  | FP16BS |               Note                |
| :------------------------------------: | :------------------------: | :---: | :----: | :---: | :----: | :-------------------------------: |
|      NVIDIA GeForce RTX 3090 24GB      |    Windows<br />566.14     | 16311 |  256   | 28197 |  256   |                 /                 |
|         NVIDIA RTX A5000 24GB          |     Linux<br />535.183     | 15090 |  512   | 27155 |  1024  |                 /                 |
|    NVIDIA RTX A5000 24GB    2 GPUs     |     Linux<br />535.183     | 26962 |  1024  | 49930 |  3072  |              NVLink               |
|      NVIDIA GeForce RTX 3080 20GB      | Linux (Docker)<br />560.35 | 13320 |  256   | 24205 |  256   | Unofficial Video Memory Expansion |
| NVIDIA GeForce RTX 3080 20GB    2 GPUs | Linux (Docker)<br />560.35 | 23261 |  1024  | 40250 |  2048  | Unofficial Video Memory Expansion |



#### Nvidia Turing

|             Device              |      Platform      | FP32 | FP32BS | FP16  | FP16BS |               Note                |
| :-----------------------------: | :----------------: | :--: | :----: | :---: | :----: | :-------------------------------: |
| NVIDIA GeForce RTX 2080 Ti 22GB | Linux<br />550.120 | 8754 |  256   | 19007 |  1024  | Unofficial Video Memory Expansion |



#### Nvidia Volta

|        Device         |          Platform          | FP32  | FP32BS | FP16  | FP16BS | Note |
| :-------------------: | :------------------------: | :---: | :----: | :---: | :----: | :--: |
| Tesla V100S-PCIE 32GB | Linux (Docker)<br />550.90 | 11577 |  256   | 27963 |  256   |  /   |



#### Nvidia Pascal

|            Device            |          Platform           | FP32 | FP32BS | FP16 | FP16BS |                     Note                     |
| :--------------------------: | :-------------------------: | :--: | :----: | :--: | :----: | :------------------------------------------: |
| NVIDIA TITAN X (Pascal) 12GB |     Windows<br />566.14     | 5792 |  256   | 7230 |  256   | FP16 Not Officially Supported By Pascal Arch |
|     NVIDIA TITAN Xp 12GB     | Linux (Docker)<br />550.120 | 6792 |  256   | 7641 |  256   | FP16 Not Officially Supported By Pascal Arch |



#### Nvidia Maxwell

| Device | Platform | FP32 | FP32BS | FP16 | FP16BS | Note |
| :----: | :------: | :--: | :----: | :--: | :----: | :--: |
|        |          |      |        |      |        |      |



#### Nvidia Kepler

| Device | Platform | FP32 | FP32BS | FP16 | FP16BS | Note |
| :----: | :------: | :--: | :----: | :--: | :----: | :--: |
|        |          |      |        |      |        |      |



#### Intel

|               Device                |      Platform      | FP32 | FP32BS | FP16 | FP16BS |        Note         |
| :---------------------------------: | :----------------: | :--: | :----: | :--: | :----: | :-----------------: |
| Intel(R) Arc(TM) A770 Graphics 16GB | Linux<br />i915 xe | 5121 |  256   | 8049 |  256   | GradScaler Not Work |
