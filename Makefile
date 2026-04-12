.PHONY: smart tpu tpu-multi calibrate docs docs-dev deps deps-dry-run help

MODEL ?= resnet50
SIZE ?= 1024
EPOCHS ?= 5

SMART_ARGS ?=

TPU_DTYPE ?= BF16
TPU_CORES ?= 8
TPU_ARGS ?=

CALIBRATE_ARGS ?= --json
DEPS_ARGS ?=

smart:
	python3 main_auto.py -mt $(MODEL) -s $(SIZE) -e $(EPOCHS) $(SMART_ARGS)

tpu:
	python3 main_tpu.py -mt $(MODEL) -s $(SIZE) -e $(EPOCHS) -dt $(TPU_DTYPE) $(TPU_ARGS)

tpu-multi:
	python3 main_tpu.py -mt $(MODEL) -s $(SIZE) -e $(EPOCHS) -dt $(TPU_DTYPE) --num_cores $(TPU_CORES) $(TPU_ARGS)

calibrate:
	python3 calibrate_memory.py $(CALIBRATE_ARGS)

docs:
	cd docs-src && npm ci && npm run build

docs-dev:
	cd docs-src && npm run dev

deps:
	python3 scripts/manage-deps.py $(DEPS_ARGS)

deps-dry-run:
	python3 scripts/manage-deps.py --dry-run $(DEPS_ARGS)

help:
	@echo "==============================================================="
	@echo "GPU-Insights Makefile"
	@echo "==============================================================="
	@echo ""
	@echo "Available models: cnn, resnet50, vit, unet, ddpm"
	@echo ""
	@echo "Targets:"
	@echo "  make smart      - Smart launcher (default entrypoint)"
	@echo "  make tpu        - TPU single-core benchmark"
	@echo "  make tpu-multi  - TPU multi-core benchmark"
	@echo "  make calibrate  - Run memory calibration"
	@echo "  make docs       - Build visualization website to docs/"
	@echo "  make docs-dev   - Start docs dev server with hot reload"
	@echo "  make deps       - Check/install Python deps for the detected backend"
	@echo "  make deps-dry-run - Preview dependency install commands"
	@echo "  make help       - Show this help"
	@echo ""
	@echo "Common variables:"
	@echo "  MODEL=<name>         Benchmark model (default: resnet50)"
	@echo "  SIZE=<mb>            Dataset size in MB (default: 1024)"
	@echo "  EPOCHS=<n>           Training epochs (default: 5)"
	@echo "  SMART_ARGS='...'     Extra args passed to main_auto.py"
	@echo "  TPU_DTYPE=<dtype>    TPU dtype (default: BF16)"
	@echo "  TPU_CORES=<n>        TPU core count for tpu-multi (default: 8)"
	@echo "  TPU_ARGS='...'       Extra args passed to main_tpu.py"
	@echo "  CALIBRATE_ARGS='...' Extra args passed to calibrate_memory.py"
	@echo "  DEPS_ARGS='...'      Extra args passed to scripts/manage-deps.py"
	@echo ""
	@echo "Examples:"
	@echo "  make smart"
	@echo "  make smart MODEL=vit SMART_ARGS='--dry-run'"
	@echo "  make smart MODEL=unet SIZE=512 EPOCHS=2"
	@echo "  make tpu MODEL=resnet50"
	@echo "  make tpu-multi MODEL=vit TPU_CORES=8"
	@echo "  make calibrate CALIBRATE_ARGS='-mt resnet50 -dt FP16'"
	@echo "  make deps-dry-run"
