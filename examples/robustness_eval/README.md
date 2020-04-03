# Robustness evaluation on ImageNet-C, -R, -A and -D

In this example we show how to use `robusta` for robustness evaluation on basis of the original [PyTorch ImageNet example](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
You can check the difference to the original ImageNet train/evaluation script by running:

```bash
diff main.py <(curl -s https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py) 
```

For standard robustness evaluation, you want to use

``` python
import robusta.datasets

robusta.datasets.ImageNet1k(datadir, split = "val")
robusta.datasets.ImageNet200(datadir)
robusta.datasets.ImageNetC(datadir, corruption = "gaussian_blur", severity = 3)
robusta.datasets.ImageNetR(datadir)
robusta.datasets.ImageNetA(datadir)
robusta.datasets.ImageNetD(datadir, domain = "sketch")
```

Each of the datasets is equipped with an accuracy metric that accepts ImageNet classes (1000 classes in PyTorch format).

``` python
dataset.accuracy_metric()
```