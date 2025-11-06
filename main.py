from benchmark.Bench import Bench
import argparse


def main():
    parser = argparse.ArgumentParser(description="Command line settings.")

    parser.add_argument("-m", "--manual", action="store_true", default=False,
                        help="Enable manual benchmark.")
    parser.add_argument("-H", "--huawei", action="store_true", default=False, help="Enable Huawei NPU benchmark.")
    parser.add_argument("-M", "--mthreads", action="store_true", default=False, help="Enable mthreads GPU benchmark.")
    parser.add_argument("-a", "--auto", action="store_true", default=False,
                        help="Enable auto benchmark.")
    parser.add_argument("-s", "--size", type=int, required=False, default=1024,
                        help="Set the CUDA memory size in MB.")
    parser.add_argument("-e", "--epochs", type=int, required=False, default=5,
                        help="Set the epochs.")
    parser.add_argument("-mt", "--model", type=str, required=False, default="cnn",
                        help="Set the model type.")
    parser.add_argument("-dt", "--data_type", type=str, required=False, default="FP32",
                        help="Set the data type (FP32, FP16 or BF16).")
    parser.add_argument("-bs", "--batch", type=int, required=False, default=0,
                        help="Set the batch size.")
    parser.add_argument("-abs", "--auto_batch_size", action="store_true", default=False,
                        help="Enable automatic batch size optimization (only for ResNet50).")
    parser.add_argument("-cudnn", "--cudnn_benchmark", action="store_true", default=False,
                        help="Enable cudnn benchmark.")
    parser.add_argument("-gpu", "--gpu_id", type=str, required=False, default="0",
                        help="Set the GPU ID(s), e.g., '0' or '0,1' for multiple GPUs.")

    args = parser.parse_args()

    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_id.split(',')]

    model = "cnn"
    if args.model in ["resnet50", "ResNet-50"]:
        model = "resnet50"
    elif args.model in ["cnn", "CNN"]:
        model = "cnn"

    data_type = "FP32"
    if args.data_type in ["FP16", "fp16"]:
        data_type = "FP16"
    elif args.data_type in ["FP32", "fp32"]:
        data_type = "FP32"
    elif args.data_type in ["BF16", "bf16"]:
        data_type = "BF16"

    if args.auto:
        print("Auto benchmark is not available.")
        # b = Bench(auto=True)
        # b.start()
    elif args.manual:
        # Check if auto_batch_size is enabled for non-resnet50 models
        if args.auto_batch_size and model != "resnet50":
            print("Warning: Auto batch size is only supported for ResNet50 model. Ignoring -abs flag.")
            args.auto_batch_size = False
        
        b = Bench(auto=False, huawei=args.huawei, mthreads=args.mthreads, size=args.size, epochs=args.epochs, method=model, batch_size=args.batch,
                  cudnn_benchmark=args.cudnn_benchmark, data_type=data_type, gpu_ids=gpu_ids, auto_batch_size=args.auto_batch_size)
        b.start()


if __name__ == "__main__":
    main()

