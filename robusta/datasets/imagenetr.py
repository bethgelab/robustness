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

from robusta.datasets.base import ImageNetRClasses
from robusta.datasets.base import RemappedImageNet


class ImageNetR(RemappedImageNet):
    """This class implements the ImageNet-R dataset from https://arxiv.org/abs/2006.16241,
    https://github.com/hendrycks/imagenet-r. It contains different renditions of 200 ImageNet
    classes. The functionality of this dataset is implemented in robusta.datasets.base.py.
    For the evaluation, one needs to remap the predictions from the 1000 ImageNet classes to
    the 200 ImageNet-R classes which is done in the RemappedImageNet class."""

    @property
    def mask(self):
        return ImageNetRClasses.get_class_mask()

    def accuracy_metric(self, logits, targets):
        return super().accuracy_metric(logits, targets, ImageNetR.mask)
