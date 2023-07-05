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

import torch.nn

from robusta.selflearning import functional
from robusta.selflearning.nn import EntropyLoss, GeneralizedCrossEntropy


def _iter_params(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for parameter in module.parameters():
                yield parameter


def adapt(model, adapt_type="affine"):
    if adapt_type not in ["affine"]:
        raise ValueError(adapt_type)
    return iter(_iter_params(model))
