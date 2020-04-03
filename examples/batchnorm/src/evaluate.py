import argparse
import glob
import numpy as np
import os
import PIL
import random
import shutil
import time
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from robusta.batchnorm import bn
from robusta.batchnorm import stages
from robusta.models.fixup import fixup_resnet50
from robusta.models.resnet_gn import resnet50 as resnet_50_gn
from robusta.models.resnet_gn import resnet152 as resnet_152_gn
from robusta.models.resnet_gn import resnet101 as resnet_101_gn

import config
from meters import AverageMeter, ProgressMeter, get_accuracy

try:
    import tqdm.tqdm as tqdm
except ImportError:
    tqdm = None


def print_version():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            try:
                print(f"| {val.__name__} version: {imp.__version__}")
            except Exception:
                continue


def main(argv):
    print_version()

    args = config.parse_args(argv)
    print("| Parsed arguments:", args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()
    print(f"| Using {ngpus_per_node} gpus.")
    return main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("| Use GPU: {} for training".format(args.gpu))

    # create model
    if args.arch.startswith("resnext") and args.arch.endswith("wsl"):
        valid_options = [
            "resnext101_32x8d_wsl",
            "resnext101_32x16d_wsl",
            "resnext101_32x32d_wsl",
            "resnext101_32x48d_wsl",
        ]
        assert args.arch in valid_options
        model = torch.hub.load("facebookresearch/WSL-Images", args.arch)
    else:
        if args.pretrained:
            print("| => using pre-trained model '{}'".format(args.arch))
            if "efficientnet" in args.arch:
                model = EfficientNet.from_pretrained(args.arch)
            elif "fixup_resnet50" in args.arch:
                model = fixup_resnet50()
            elif "resnet50_gn" in args.arch:
                model = resnet_50_gn()
            elif "resnet101_gn" in args.arch:
                model = resnet_101_gn()
            elif "resnet152_gn" in args.arch:
                model = resnet_152_gn()
            else:
                model = models.__dict__[args.arch](pretrained=True)
        else:
            if "efficientnet" in args.arch:
                raise NotImplementedError(
                    "Using a not pretrained Efficient Net is not supported."
                )
            elif "resnet50_gn" in args.arch:
                model = resnet_50_gn()
            elif "resnet101_gn" in args.arch:
                model = resnet_101_gn()
            elif "resnet152_gn" in args.arch:
                model = resnet_152_gn()
            else:
                print("| => creating model '{}'".format(args.arch))
                model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = torch.nn.DataParallel(model).cuda()

        # DataParallel will divide and allocate batch_size to all available GPU
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.gpu is not None:
        criterion = criterion.cuda(args.gpu)

    # optionally evaluate previous model
    if args.resume != "":
        print('Loading model checkpoint')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        
    # cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    T = []
    if args.resize_and_crop and "efficientnet" not in args.arch:
        T.extend(
            [
                transforms.Resize(args.resizepar),
                transforms.CenterCrop(args.croppar),
            ]
        )
    if "efficientnet" in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
        T.extend(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(image_size),
            ]
        )
    T.extend([transforms.ToTensor(), normalize])
    print(T)

    fnames = glob.glob(os.path.join(args.imagenet_path, "*"))
    print(f"Dataset at {args.imagenet_path}, {len(fnames)}")
    if len(fnames) == 1000:
        print(f"| Use imagenet folder from {args.imagenet_path}")
        dataset = datasets.ImageFolder(
            args.imagenet_path, transforms.Compose(T))
    else:
        print(f"| Use subfolders in {args.imagenet_path}")
        print("|" + "\n|".join(fnames))
        dataset = torch.utils.data.ConcatDataset(
            [datasets.ImageFolder(path, transforms.Compose(T))
                for path in fnames]
        )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.workers,
        pin_memory=True,
    )
    print("| Shuffling: ", not args.no_shuffle)

    Evaluation.gpu = args.gpu    
    evaluate = Evaluation(model, criterion, args)
    emissions = evaluate(val_loader)

    top1, top5, loss = \
        emissions.top1.avg, emissions.top5.avg, emissions.losses.avg
    print(f"| Finished eval with: top-1 {top1}; top-5 {top5}; loss {loss}")

    return emissions
    # emissions.finalize().save(args.emission_path, args=args)


class Emissions:
    """ Experiment Outputs """

    def __init__(self, num_batches):
        self.num_batches = num_batches
        self.init_metrics()

    def init_metrics(self):
        self.targets = []
        self.predictions = []
        self.batch_time = AverageMeter("Time", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4e")
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")
        self.progress = ProgressMeter(
            self.num_batches,
            [self.batch_time, self.losses, self.top1, self.top5],
            prefix="| Test: ",
        )
        self.finalized = False

    def update_metrics(self, output, target, loss):
        assert not self.finalized
        acc1, acc5 = get_accuracy(output, target, topk=(1, 5))
        self.losses.update(loss.item(), output.size(0))
        self.top1.update(acc1[0].cpu(), output.size(0))
        self.top5.update(acc5[0].cpu(), output.size(0))

    def append(self, output, targets, loss, time=0):
        assert len(output) == len(targets)
        assert not self.finalized

        self.batch_time.update(time)
        self.update_metrics(output, targets, loss)

        self.predictions.append(output.detach().cpu().numpy())
        self.targets.append(targets.detach().cpu().numpy())

    def finalize(self):
        assert not self.finalized
        # assert self.num_batches == len(self.targets)
        # assert self.num_batches == len(self.predictions)

        self.predictions = np.concatenate(self.predictions, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        self.finalized = True
        return self

    def save(self, fname, **kwargs):
        assert self.finalized
        assert not os.path.exists(fname)
        np.savez(
            fname,
            top1=self.top1.avg,
            top5=self.top5.avg,
            loss=self.losses.avg,
            predictions=self.predictions,
            targets=self.targets,
            **kwargs,
        )
        return self


class Evaluation:
    """ Evaluation loop """
    
    def __init__(self, model, criterion, args):
        super(Evaluation, self).__init__()

        self.args = args
        self.model = model
        self.criterion = Evaluation.to_device(criterion)

    
    gpu = None
    def to_device(arg):
        if Evaluation.gpu is not None:
            return arg.cuda(Evaluation.gpu, non_blocking=True)
        else:
            return arg

    def iterate(self, loader):
        if self.args.tqdm and tqdm is not None:
            return tqdm(enumerate(loader))
        return enumerate(loader)

    def use_train_statistics(self, module):
        if isinstance(module, nn.BatchNorm2d):
            module.train()

    @property
    def elapsed_time(self):
        """ return elapsed time since last call """
        if not hasattr(self, "_end"):
            self._end = time.time()
        span = time.time() - self._end
        self._end = time.time()
        return span

    def select_ablations(self):
        """ Check args for ablation settings """
        if self.args.ema_batchnorm:
            print(
                "| Collecting statistics during test time with exponential \
                    moving averaging. Experimental version from 30-04"
            )
            bn.adapt(self.model)
            self.warmup_batches = (
                self.args.ema_warmup_samples // self.args.test_batch_size
            )
            assert self.warmup_batches > 0
            return

        if self.args.adapt_mean or self.args.adapt_var:
            print(
                f"| Adapting mean[{self.args.adapt_mean}] and \
                    var[{self.args.adapt_var}]"
            )
            bn.adapt_parts(self.model,
                           self.args.adapt_mean, self.args.adapt_var)
            return

        if self.args.adapt_stage is not None:
            print(f"| Adapting only model stage {self.args.adapt_stage}")
            stages.choose_one_adaptation(self.model, self.args.adapt_stage)
            return

        if self.args.leave_stage is not None:
            print(f"| Adapting all but model stage {self.args.adapt_stage}")
            stages.leave_one_out_adaptation(self.model, self.args.leave_stage)
            return

        if self.args.adapt_prior is not None or \
                self.args.adapt_prior_bsz is not None:
            
            assert self.args.adapt_prior is None or \
                self.args.adapt_prior_bsz is None

            if self.args.adapt_prior_bsz is not None:
                n = self.args.test_batch_size
                N = self.args.adapt_prior_bsz
                setattr(self.args, "adapt_prior", float(N) / float(N + n))
            print(
                f"| Using a prior on the statistics with \
                    lambda = {self.args.adapt_prior}"
            )
            bn.adapt_bayesian(self.model, self.args.adapt_prior, 
                              Evaluation.to_device)
            return

    def __call__(self, val_loader):
        print("| Start evaluation")

        os.system("git log -n1 --oneline")
        self.warmup_batches = 0
        self.model.eval()
        if self.args.train_mode_during_eval:
            print("| Using model in train() mode")
            self.model.apply(self.use_train_statistics)
            
            assert self.args.ema_batchnorm is False, \
                "--ema-batchnorm mode does not make sense with \
                    --train-mode-during-eval"
        else:
            print("| Using model in eval() mode")
            self.model.eval()
            self.select_ablations()

        emissions = Emissions(len(val_loader))
        with torch.no_grad():
            _ = self.elapsed_time

            if self.warmup_batches > 0:
                print(f"| Starting warmup for a total of \
                    {self.warmup_batches} batches")
                for i, (images, target) in self.iterate(val_loader):
                    if i > self.warmup_batches:
                        break
                    images = Evaluation.to_device(images)
                    output = self.model(images)

            print("| Starting evaluation")
            for i, (images, target) in self.iterate(val_loader):
                images = Evaluation.to_device(images)
                target = Evaluation.to_device(target)
                output = self.model(images)
                loss = self.criterion(output, target)

                emissions.append(output, target, loss, self.elapsed_time)
                if i % self.args.print_freq == 0:
                    emissions.progress.display(i)
                    if self.args.dry_run:
                        break

        return emissions

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
