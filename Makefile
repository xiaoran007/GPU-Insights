.PHONY: run help abs ddp ddp-abs tpu tpu-multi vit unet ddpm docs docs-dev

# Number of processes for DDP (default: 2)
GPU ?= 2

run:
	python main.py -m -s 512 -e 2 -mt resnet50 -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -dt FP32

abs:
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP32

vit:
	python main.py -m -s 512 -e 2 -mt vit -dt FP16
	python main.py -m -s 512 -e 2 -mt vit -dt FP32

unet:
	python main.py -m -s 512 -e 2 -mt unet -dt FP16
	python main.py -m -s 512 -e 2 -mt unet -dt FP32

ddpm:
	python main.py -m -s 512 -e 2 -mt ddpm -dt FP16
	python main.py -m -s 512 -e 2 -mt ddpm -dt FP32

ddp:
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -dt FP16
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -dt FP32

ddp-abs:
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -abs -dt FP16
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -abs -dt FP32

tpu:
	python main_tpu.py -s 512 -e 2 -mt resnet50 -dt BF16
	python main_tpu.py -s 512 -e 2 -mt resnet50 -dt FP32

tpu-multi:
	python main_tpu.py -s 512 -e 2 -mt resnet50 -dt BF16 --num_cores 8

docs:
	cd docs-src && npm ci && npm run build

docs-dev:
	cd docs-src && npm run dev

help:
	@echo "==================================================================="
	@echo "GPU-Insights Benchmark Makefile"
	@echo "==================================================================="
	@echo ""
	@echo "Available models: cnn, resnet50, vit, unet, ddpm"
	@echo ""
	@echo "Targets:"
	@echo "  make run       - ResNet50 benchmarks (FP16 + FP32)"
	@echo "  make abs       - ResNet50 with auto batch size"
	@echo "  make vit       - ViT-Base benchmarks (FP16 + FP32)"
	@echo "  make unet      - UNet segmentation benchmarks (FP16 + FP32)"
	@echo "  make ddpm      - DDPM diffusion benchmarks (FP16 + FP32)"
	@echo "  make ddp       - ResNet50 DDP (default: 2 GPUs)"
	@echo "  make ddp-abs   - ResNet50 DDP with auto batch size"
	@echo "  make tpu       - ResNet50 on TPU single-core"
	@echo "  make tpu-multi - ResNet50 on TPU multi-core (8 cores)"
	@echo "  make docs      - Build visualization website to docs/"
	@echo "  make docs-dev  - Start docs dev server with hot reload"
	@echo "  make help      - Show this help"
	@echo ""
	@echo "DDP Parameters:"
	@echo "  GPU=<num>   - Number of processes/GPUs (default: 2)"
	@echo ""
	@echo "Examples:"
	@echo "  python main.py -m -mt vit -s 512 -e 2 -dt FP16"
	@echo "  python main.py -m -mt unet -s 512 -e 2 -dt FP32"
	@echo "  python main.py -m -mt ddpm -s 512 -e 2 -dt FP16"
	@echo "  python main.py -m -mt resnet50 -s 512 -e 2 -abs -dt FP16"
	@echo "  python main.py -m -mt resnet50 -d npu -s 512 -e 2 -bs 64 -dt FP16"
	@echo "  torchrun --nproc_per_node=4 main_ddp.py -mt vit -s 512 -e 2 -dt FP16"