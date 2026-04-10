from benchmark.Bench import Bench
from benchmark.cli import build_common_parser, parse_common_args


def main():
    parser = build_common_parser("GPU-Insights single-device benchmark")

    parser.add_argument(
        "-gpu", "--gpu_id", type=str, default="0",
        help="GPU ID(s), e.g., '0' or '0,1' for multiple GPUs.",
    )

    args = parse_common_args(parser)
    gpu_ids = [int(g) for g in args.gpu_id.split(',')]

    b = Bench(
        device=args.device,
        size=args.size,
        epochs=args.epochs,
        method=args.model,
        batch_size=args.batch,
        cudnn_benchmark=args.cudnn_benchmark,
        data_type=args.data_type,
        gpu_ids=gpu_ids,
        auto_batch_size=args.auto_batch_size,
    )
    b.start()


if __name__ == "__main__":
    main()
