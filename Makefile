.PHONY: run help abs ddp ddp-abs

# Number of processes for DDP (default: 2)
GPU ?= 2

run:
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32

abs:
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP32

ddp:
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32

ddp-abs:
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -abs -dt FP16
	torchrun --nproc_per_node=$(GPU) main_ddp.py -s 512 -e 2 -mt resnet50 -abs -dt FP32

help:
	@echo "==================================================================="
	@echo "GPU-Insights Benchmark Makefile"
	@echo "==================================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make run      - Run standard ResNet50 benchmarks (FP16 + FP32)"
	@echo "  make abs      - Run ResNet50 with automatic batch size optimization"
	@echo "  make ddp      - Run ResNet50 with DDP (default: 2 GPUs, FP16 + FP32)"
	@echo "  make ddp-abs  - Run ResNet50 with DDP and auto batch size (default: 2 GPUs)"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "DDP Parameters:"
	@echo "  GPU=<num>   - Number of processes/GPUs to use (default: 2)"
	@echo ""
	@echo "Single GPU examples:"
	@echo "  python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP16"
	@echo "  python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt BF16"
	@echo "  python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP32"
	@echo ""
	@echo "DDP examples (multi-GPU):"
	@echo "  make ddp GPU=2     # Run with 2 GPUs (default)"
	@echo "  make ddp GPU=4     # Run with 4 GPUs"
	@echo "  make ddp-abs GPU=4 # Run with 4 GPUs and auto batch size"
	@echo ""
	@echo "Direct torchrun examples:"
	@echo "  torchrun --nproc_per_node=2 main_ddp.py -s 512 -e 2 -mt resnet50 -abs -dt FP16"
	@echo "  torchrun --nproc_per_node=4 main_ddp.py -s 1024 -e 5 -mt resnet50 -abs -dt FP32"