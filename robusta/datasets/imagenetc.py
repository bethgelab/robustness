import os
import torch
import torchvision
from typing import Any, Callable, Optional
from torchvision.datasets.folder import default_loader


class ImageNetC(torchvision.datasets.ImageFolder):
    num_samples = 50000
    num_classes = 1000
    image_size = (224, 224)

    train_corruptions = ("brightness", "elastic_transform", "impulse_noise",
                         "pixelate", "snow", "zoom_blur", "contrast", "fog",
                         "gaussian_noise", "jpeg_compression", "defocus_blur",
                         "frost", "glass_blur", "motion_blur", "shot_noise")
    test_corruptions = ("gaussian_blur", "saturate", "spatter", "speckle_noise")
    severities = ("1", "2", "3", "4", "5")

    def __init__(self,
                 root: str,
                 corruption: str,
                 severity: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super(ImageNetC,
              self).__init__(root=os.path.join(root, corruption, str(severity)),
                             transform=transform,
                             target_transform=target_transform,
                             loader=loader,
                             is_valid_file=is_valid_file)
        assert corruption in ImageNetC.train_corruptions or \
            corruption in ImageNetC.test_corruptions
        assert str(severity) in ImageNetC.severities
