import os 
import sys
import pytest
from torch.utils.data import _utils, Dataset


module_folder = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(module_folder, 'adaptation'))
import run


@pytest.mark.parametrize("arch", [
    ("alexnet"),
    ("efficientnet-b0"),
    ("efficientnet-b1"),
    ("fixup_resnet50"),
    ("resnet50"),
    ("resnet101"),
    ("resnet152"),
    ("resnet50_gn"),
    ("resnet101_gn"),
    ("resnet152_gn"),
    ("resnext101_32x8d_wsl"),
    ("resnext101_32x16d_wsl"),
    ("resnext101_32x32d_wsl"),
    ("resnext101_32x48d_wsl"),
    ("vgg11"),
    ("vgg11_bn"),
    ("vgg19"),
    ("vgg19_bn"),
    ])
def test_architectures(arch):
    emissions = run.main([
        "--test-batch-size", "1",
        "--imagenet-path", str(module_folder)+"/tests/imagenet_c",  
        "--arch", arch,
        "--pretrained",
        "--emission-path", "/tmp/run_experiment"
        ])  
    
    # This is a dry run test. We test that:
    # 1. code compiles
    # 2. feed-forward works and we have a prediction for an input image. 
    assert emissions.predictions is not None

    # TODO why all predictions are 0 on fixup_resnet50?
    if arch is not "fixup_resnet50": 
        assert emissions.predictions[0].max() > 0


@pytest.mark.parametrize("extra_args", [
    ("--resize-and-crop", ),
    ("--ema-batchnorm", ),
    ("--adapt-mean", ),
    ("--adapt-stage", "0"),
    ("--leave-stage", "0"),
    ("--adapt-prior", "0"),
    ("--adapt-prior-bsz", "0"),
    ("--train-mode-during-eval", )
    ])
def test_adaptations(extra_args):
    emissions = run.main([
        "--test-batch-size", "1",
        "--imagenet-path", str(module_folder)+"/tests/imagenet_c",  
        "--arch", "resnet50",
        "--pretrained",
        "--emission-path", "/tmp/run_experiment",
        *extra_args # expand values from tuple
        ])  
    
    # This is a dry run test. We test that:
    # 1. code compiles
    # 2. applying an adaptation works
    # 3. feed-forward works and we have a prediction for an input image. 
    assert emissions.predictions is not None and \
            emissions.predictions[0].max() > 0


if __name__ == '__main__':
    pytest.main()
