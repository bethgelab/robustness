import argparse
import os

import torchvision.models as models


def assert_exists(fname):
    assert os.path.exists(fname)
    return fname


def assert_not_exists(fname):
    assert not os.path.exists(fname)
    return fname


def parse_args(args):
    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    model_names.extend(
        [
            "resnext101_32x8d_wsl",
            "resnext101_32x16d_wsl",
            "resnext101_32x32d_wsl",
            "resnext101_32x48d_wsl",
        ]
    )
    model_names.extend(
        [
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b2",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ]
    )
    model_names.extend(
        [
            "fixup_resnet50",
            "resnet50_gn",
            "resnet101_gn",
            "resnet152_gn",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b2",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ]
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--imagenet-path", type=assert_exists, required=True)
    parser.add_argument("--resize-and-crop", action="store_true")
    parser.add_argument("--emission-path", type=assert_not_exists, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--resizepar",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--croppar",
        default=224,
        type=int,
    )
    parser.add_argument(
        "-tb",
        "--test-batch-size",
        default=200,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--train-mode-during-eval",
        action="store_true",
        help="use train mode during evaluation.",
    )
    parser.add_argument(
        "--adapt-only-one-layer",
        action="store_true",
        help="use train mode during evaluation.",
    )
    parser.add_argument("--layer-to-adapt", default=0, type=int)
    parser.add_argument(
        "--pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument("--dry-run", action="store_true")

    # ############ Ablation arguments
    parser.add_argument(
        "--adapt-mean", action="store_true",
        help="use test time statistics for mean"
    )

    parser.add_argument(
        "--adapt-var", action="store_true",
        help="use test time statistics for variance"
    )

    parser.add_argument(
        "--adapt-stage",
        type=int,
        default=None,
        help="use test time statistics in only this stage",
    )
    parser.add_argument(
        "--leave-stage",
        type=int,
        default=None,
        help="use test time statistics everywhere except in this stage",
    )
    parser.add_argument(
        "--adapt-prior",
        type=float,
        default=None,
        help="use train time statistics as a prior during evaluation. Specify the averaging factor directly",
    )
    parser.add_argument(
        "--adapt-prior-bsz",
        type=int,
        default=None,
        help="use train time statistics as a prior during evaluation. Specify the training set size.",
    )

    # EMA
    parser.add_argument(
        "--ema-batchnorm",
        action="store_true",
        help="use test time ema statistics for all batch norm layers",
    )
    parser.add_argument("--ema-warmup-samples", type=int, default=5000)
    # ############ End ablation arguments

    parser.add_argument(
        "-p",
        "--print-freq",
        default=1000,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--tqdm", action="store_true")

    return parser.parse_args(args)
