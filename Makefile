.PHONY: run help

run:
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16
	python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP32

help:
	@echo "Makefile for running benchmarks"
	@echo "Run this command: python main.py -m -s 512 -e 2 -mt resnet50 -bs 256 -dt FP16"