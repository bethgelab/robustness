import torch
import torch.nn as nn
import torchvision.models as models


class ZeroOneResNet50_parallel(nn.Module):
    def __init__(self, device="cuda", pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        self.mean = nn.Parameter(
            torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None],
            requires_grad=False,
        )  # asdf changed
        self.std = nn.Parameter(
            torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None],
            requires_grad=False,
        )

    def forward(self, input):
        # input = (input - self.mean) / self.std
        return self.resnet(input)


class ZeroOneInceptionV3(nn.Module):
    def __init__(self, device="cuda", pretrained=False):
        super().__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=True)
        self.mean = nn.Parameter(
            torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None],
            requires_grad=False,
        )
        self.std = nn.Parameter(
            torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None],
            requires_grad=False,
        )

    def forward(self, input):
        # input = (input - self.mean) / self.std
        return self.inception(input)
