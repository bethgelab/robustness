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

from torch import nn
import robusta.selflearning.functional as RF

class GeneralizedCrossEntropy(nn.Module):

    def __init__(self, q = 0.8):
        super().__init__()
        self.q = q

    def forward(self, logits, target = None):
        if target is None:
            target = logits.argmax(dim = 1)
        return RF.gce(logits, target, self.q)

class EntropyLoss(nn.Module):

    def __init__(self, stop_teacher_gradient = False):
        super().__init__()
        self.stop_teacher_gradient = stop_teacher_gradient

    def forward(self, logits, target = None):
        if target is None:
            target = logits
        if self.top_teacher_gradient:
            target = target.detach()
        return RF.entropy(logits, target, self.q)