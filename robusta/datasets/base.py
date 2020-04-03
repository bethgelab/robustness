# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

import torchvision.datasets
import torchvision.transforms

class TorchvisionTransform(torchvision.transforms.Compose):
    """Standard torchvision transform for cropped and non-cropped datasets."""

    def __init__(self, resize = False):
        self.resize = resize

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        super().__init__([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                self.mean, self.std
            )
        ])

class ImageNetRobustnessDataset(torchvision.datasets.ImageFolder):

    def __init__(self, dataset_dir, transform = None, **kwargs):
        if transform == "torchvision":
            transform = TorchvisionTransform()
        super().__init__(dataset_dir, transform = transform, **kwargs)

    def accuracy_metric(self, logits, targets):
        raise NotImplementedError()

class RemappedImageNet():

    def __init__(self):
        super().__init__()

    def map_logits(self, logits):
        output = logits[:, imagenet_r_mask]
        return output

    def accuracy_metric(self, logits, targets):
        logits200 = self.map_logits(logits)
        super().accuracy_metric(logits200, targets)

