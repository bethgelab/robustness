""" Adapt torchvision models on the full test dataset.

This scripts estimates the true test statistics on the full test dataset
by sequentially updating the statistics within the model. Tested on ResNet50
models of torchvision, but should be applicable to other models as well.
"""

from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize, ToTensor, CenterCrop, Compose
import numpy as np
import glob
import os
import tqdm

import torch
from torch import nn
from torchvision.models import resnet
from torchvision import datasets
from torchvision.models import resnet


def get_stages(model):
    input_stage = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
    )
    stages = (
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    flattened_stages = []
    for stage in stages:
        if isinstance(stage, nn.Sequential):
            for layer in stage:
                flattened_stages.append(layer)
        else:
            flattened_stages.append(stage)

    stages = []
    current_module = []
    for stage in flattened_stages:
        current_module.append(stage)
        if count_bn_layers(stage) > 0:
            stages.append(nn.Sequential(*current_module))
            current_module = []

    stages = [
        input_stage,
    ] + stages
    return stages


def count_bn_layers(stage):
    return len(list(filter(lambda m: isinstance(m, nn.BatchNorm2d), stage.modules())))


def reset_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        module.reset_running_stats()
        # Use exponential moving average
        module.momentum = None
    for p in module.parameters():
        p.requires_grad_(False)


def get_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = "Full model adaptation to the test set."
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workers", "-j", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", type=str, default="false")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    return parser.parse_args()


def get_dataset(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    T = Compose([ToTensor(), Normalize(mean, std)])

    fnames = glob.glob(os.path.join(args.dataset, "*"))
    if len(fnames) == 1000:
        print(f"| Use imagenet folder from {args.dataset}")
        dataset = datasets.ImageFolder(args.dataset, Compose(T))
    else:
        print(f"| Use subfolders in {args.dataset}")
        print("|" + "\n|".join(fnames))
        dataset = torch.utils.data.ConcatDataset(
            [datasets.ImageFolder(path, Compose(T)) for path in fnames]
        )

    return dataset


def get_loader(dataset, args):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def iterate(loader, args):
    if args.verbose == "tqdm":
        return tqdm.tqdm(enumerate(loader), total=len(loader))
    else:
        return enumerate(loader)


class ZipLoader:
    def __init__(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration
        outputs = self.outputs[index]
        targets = self.targets[index]
        # self.outputs[index] = None
        # self.targets[index] = None
        return outputs, targets

    def __len__(self):
        return len(self.outputs)


if __name__ == "__main__":

    args = get_args()

    def log(*largs, **kwargs):
        if args.verbose != "false":
            print(*largs, **kwargs)

    model = resnet.resnet50(pretrained=True)

    if args.resume:
        print("load augmix model")
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint["state_dict"]
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    model.apply(reset_stats)
    model.cuda().train()

    dataset = get_dataset(args)

    with torch.no_grad():
        loader = get_loader(dataset, args)
        log(f"| Loaded dataset with {len(loader)} batches")
        stages = get_stages(model)
        for i, model_stage in enumerate(stages):
            log(
                f"| Compute outputs for model stage {i+1}/{len(stages)}:\n{model_stage}"
            )
            num_bn = count_bn_layers(model_stage)
            log(f"| Found {num_bn} adaptation layers. Looping {num_bn}x.")
            assert num_bn >= 1
            for n in range(num_bn):
                log(f"| Estimation {n+1}/{num_bn}")
                outputs = []
                targets = []
                for batch_idx, (data, target) in enumerate(loader):
                    log(f"| Process {batch_idx}")
                    data = data.cuda()
                    output = model_stage(data)
                    del data
                    outputs.append(output.cpu())
                    targets.append(target.cpu())
                    if args.dry_run:
                        break
                    if batch_idx > 98:
                        break
            del loader

            loader = ZipLoader(outputs, targets)

    log(f"| Saving model to {args.output}")
    torch.save(model.cpu().state_dict(), args.output)

    log(f"| Done.")
