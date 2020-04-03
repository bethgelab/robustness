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

""" Batch norm variants
"""

import torch
from torch import nn
from torch.nn import functional as F


def adapt_ema(model: nn.Module):
    return EMABatchNorm.adapt_model(model)


def adapt_parts(model: nn.Module, adapt_mean: bool, adapt_var: bool):
    return PartlyAdaptiveBN.adapt_model(model, adapt_mean, adapt_var)


def adapt_bayesian(model: nn.Module, prior: float):
    return BayesianBatchNorm.adapt_model(model, prior=prior)


class PartlyAdaptiveBN(nn.Module):
    @staticmethod
    def find_bns(parent, estimate_mean, estimate_var):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = PartlyAdaptiveBN(child, estimate_mean, estimate_var)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(
                    PartlyAdaptiveBN.find_bns(child, estimate_mean,
                                              estimate_var)
                )

        return replace_mods

    @staticmethod
    def adapt_model(model, adapt_mean, adapt_var):
        replace_mods = PartlyAdaptiveBN.find_bns(model, adapt_mean, adapt_var)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, estimate_mean=True, estimate_var=True):
        super().__init__()
        self.layer = layer

        self.estimate_mean = estimate_mean
        self.estimate_var = estimate_var

        self.register_buffer("source_mean", layer.running_mean.data)
        self.register_buffer("source_var", layer.running_var.data)

        self.register_buffer(
            "estimated_mean",
            torch.zeros(layer.running_mean.size(),
                        device=layer.running_mean.device),
        )
        self.register_buffer(
            "estimated_var",
            torch.ones(layer.running_var.size(),
                       device=layer.running_mean.device),
        )

    def reset(self):
        self.estimated_mean.zero_()
        self.estimated_var.fill_(1)

    @property
    def running_mean(self):
        if self.estimate_mean:
            return self.estimated_mean
        return self.source_mean

    @property
    def running_var(self):
        if self.estimate_var:
            return self.estimated_var
        return self.source_var

    def forward(self, input):
        # Estimate training set statistics
        self.reset()
        F.batch_norm(
            input,
            self.estimated_mean,
            self.estimated_var,
            None,
            None,
            True,
            1.0,
            self.layer.eps,
        )

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0.0,
            self.layer.eps,
        )


class EMABatchNorm(nn.Module):
    @staticmethod
    def reset_stats(module):
        module.reset_running_stats()
        module.momentum = None
        return module

    @staticmethod
    def find_bns(parent):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = EMABatchNorm.reset_stats(child)
                module = EMABatchNorm(module)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(EMABatchNorm.find_bns(child))

        return replace_mods

    @staticmethod
    def adapt_model(model):
        replace_mods = EMABatchNorm.find_bns(model)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # store statistics, but discard result
        self.layer.train()
        self.layer(x)
        # store statistics, use the stored stats
        self.layer.eval()
        return self.layer(x)


class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features, affine=False, momentum=1.0
        )

        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )
