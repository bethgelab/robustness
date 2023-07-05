import os

import pytest
from torch.utils.data import DataLoader, Dataset, _utils
from torchvision import transforms
from torchvision.datasets import ImageFolder

module_folder = os.path.dirname(os.path.dirname(__file__))
dummy_dataset_folder = module_folder + "/test/dummy_datasets/"


def test_imagenet():
    dataset = ImageFolder(
        root=dummy_dataset_folder + "ImageNet2012/val",
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    data_loader = DataLoader(dataset)
    record = next(iter(data_loader))
    assert record is not None
    assert len(dataset) == 1  # It's only 1 record


def test_imagenetc():
    dataset = ImageNetC(
        root=dummy_dataset_folder + "ImageNet-C",
        corruption="gaussian_blur",
        severity="1",
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    data_loader = DataLoader(dataset)
    record = next(iter(data_loader))
    assert record is not None

    assert dataset.num_samples == 50000
    assert dataset.image_size == (224, 224)
    assert dataset.num_classes == 1000
    assert len(dataset) == 1  # It's only 1 record


def test_imagenetr():
    dataset = robusta.datasets.ImageNetR(corruption=corruption, severity=severity)

    assert False


def test_imageneta():
    assert False
