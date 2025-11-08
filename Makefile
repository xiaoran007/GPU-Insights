.PHONY: run help abs

run:
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32

abs:
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP32

help:
	@echo "==================================================================="
	@echo "GPU-Insights Benchmark Makefile"
	@echo "==================================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make run      - Run standard ResNet50 benchmarks (FP16 + FP32)"
	@echo "  make abs      - Run ResNet50 with automatic batch size optimization"
	@echo "  make help     - Show this help message"
	@echo "Available cases:"
	@echo "python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP16"
	@echo "python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt BF16"
	@echo "python main.py -m -s 512 -e 2 -mt resnet50 -abs -dt FP32"