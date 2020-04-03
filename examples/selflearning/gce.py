""" Adapting ImageNet-scale models to complex distribution shifts with self-learning

Run with:

    ❯ docker pull pytorch/pytorch
    ❯ DATADIR="/path/to/imagenetc"
    ❯ curl -s https://stes.io/gce.py > gce.py
    ❯ docker run --gpus 1 -v ${DATADIR}:/data/imagenetc:ro \
        -v $(pwd):/app -w /app -u $(id -u) \
        --tmpfs /.cache --tmpfs /.local \
        -it pytorch/pytorch python gce.py

Reference: 

    Web: https://domainadaptation.org/selflearning/
    Paper: https://arxiv.org/abs/2104.12928

---

Copyright 2021 Evgenia Rusak and Steffen Schneider

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

This software is not an Amazon product.

"""

import torch
from torchvision import models, datasets, transforms

def get_dataset_loader(valdir, batch_size, shuffle):
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return val_loader

def gce(logits, target, q = 0.8):
    """ Generalized cross entropy.
    
    Reference: https://arxiv.org/abs/1805.07836
    """
    probs = torch.nn.functional.softmax(logits, dim=1)
    probs_with_correct_idx = probs.index_select(-1, target).diag()
    loss = (1. - probs_with_correct_idx**q) / q
    return loss.mean()

def adapt_batchnorm(model):
    model.eval()
    parameters = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            parameters.extend(module.parameters())
            module.train()
    return parameters


# ---

def evaluate(
        datadir = '/data/imagenetc/gaussian_blur/3',
        num_epochs = 5,
        batch_size = 96,
        learning_rate = 0.75e-3,
        gce_q = 0.8
    ):
    
    model = models.resnet50(pretrained = True).cuda()
    parameters = adapt_batchnorm(model)
    val_loader = get_dataset_loader(
        datadir,
        batch_size = batch_size,
        shuffle = True
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr = learning_rate
    )
    
    num_correct, num_samples = 0., 0.
    for epoch in range(num_epochs):
        predictions = []
        for images, targets in val_loader:

            logits = model(images.cuda())
            predictions = logits.argmax(dim = 1)
            loss = gce(logits, predictions, q = gce_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (predictions.detach().cpu() == targets).float().sum()
            num_samples += len(targets)
            print(f"Correct: {num_correct:#5.0f}/{num_samples:#5.0f} ({100 * num_correct / num_samples:.2f} %)")
            
evaluate()
