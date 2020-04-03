import os
from robusta.batchnorm import bn
from torch import nn


def test_adapt_changes_BatchNorm_to_EMABatchNorm():
    model = nn.Sequential(nn.BatchNorm2d(10))
    model = bn.adapt_ema(model)

    _, result_layer = next(iter(model.named_children()))
    assert isinstance(result_layer, bn.EMABatchNorm)


def test_adaptparts_changes_BatchNorm_to_PartlyAdaptiveBN():
    model = nn.Sequential(nn.BatchNorm2d(10))
    adapt_mean, adapt_var = False, True
    model = bn.adapt_parts(model, adapt_mean, adapt_var)

    _, result_layer = next(iter(model.named_children()))
    assert isinstance(result_layer, bn.PartlyAdaptiveBN)
    assert result_layer.estimate_mean == adapt_mean
    assert result_layer.estimate_var == adapt_var


def test_adaptbayesian_changes_BatchNorm_to_BayesianBatchNorm():
    model = nn.Sequential(nn.BatchNorm2d(10))
    prior = 0.2
    model = bn.adapt_bayesian(model, prior=prior)

    _, result_layer = next(iter(model.named_children()))
    assert isinstance(result_layer, bn.BayesianBatchNorm)
    assert result_layer.prior == prior
