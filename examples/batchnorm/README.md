# Improving robustness against common corruptions by covariate shift adaptation

Steffen Schneider*, Evgenia Rusak*, Luisa Eck, Oliver Bringmann, Wieland Brendel, Matthias Bethge 

Website: [domainadaptation.org/batchnorm](https://domainadaptation.org/batchnorm)

This repository contains evaluation code for the paper *Improving robustness against common corruptions by covariate shift adaptation*.
The repository is updated frequently. To get notified, watch and/or star this repository!

Today's state-of-the-art machine vision models are vulnerable to image corruptions like blurring or compression artefacts, limiting their performance in many real-world applications. We here argue that popular benchmarks to measure model robustness against common corruptions (like ImageNet-C) underestimate model robustness in many (but not all) application scenarios. The key insight is that in many scenarios, multiple unlabeled examples of the corruptions are available and can be used for unsupervised online adaptation. Replacing the activation statistics estimated by batch normalization on the training set with the statistics of the corrupted images consistently improves the robustness across 25 different popular computer vision models. Using the corrected statistics, ResNet-50 reaches 62.2% mCE on ImageNet-C compared to 76.7% without adaptation. With the more robust AugMix model, we improve the state of the art from 56.5% mCE to 51.0% mCE. Even adapting to a single sample improves robustness for the ResNet-50 and AugMix models, and 32 samples are sufficient to improve the current state of the art for a ResNet-50 architecture. We argue that results with adapted statistics should be included whenever reporting scores in corruption benchmarks and other out-of-distribution generalization settings

## Main results

### Results for vanilla trained and robust models on ImageNet-C

With a simple recalculation of batch normalization statistics, we improve the mean Corruption Error (mCE) of all commonly tested robust models.
| Model    |  mCE, w/o adapt  [%] ↘ | mCE, partial adapt  [%]  ↘ | mCE,  full adapt [%]   ↘    | 
|-|-|-|-|
|  Vanilla ResNet50 | 76.7 | 65.0  | 62.2 |
| [SIN](https://github.com/rgeirhos/texture-vs-shape) | 69.3 | 61.5 | 59.5|
| [ANT](https://github.com/bethgelab/game-of-noise) | 63.4  | 56.1  |53.6 |
| [ANT+SIN](https://github.com/bethgelab/game-of-noise) | 60.7 |55.3 |53.6|
| [AugMix](https://github.com/google-research/augmix) | 65.3 | 55.4 | 51.0 |
| [AssembleNet](https://github.com/clovaai/assembled-cnn) | 52.3 | -- | 50.1 |
| [DeepAugment](https://github.com/hendrycks/imagenet-r) | 60.4 | 52.3 |49.4 |
| [DeepAugment+AugMix](https://github.com/hendrycks/imagenet-r) | 53.6 | 48.4 |45.4|
| [DeepAug+AM+RNXt101](https://github.com/hendrycks/imagenet-r) | **44.5** |**40.7** | **38.0** |


### Results for models trained with [Fixup](https://github.com/hongyi-zhang/Fixup) and [GroupNorm](https://github.com/ppwwyyxx/GroupNorm-reproduce) on ImageNet-C

Fixup and GN trained models perform better than non-adapted BN models but worse than adapted BN models.

| Model    |  [Fixup](https://github.com/hongyi-zhang/Fixup), mCE [%] ↘ | [GroupNorm](https://github.com/ppwwyyxx/GroupNorm-reproduce), mCE [%] ↘ | BatchNorm, mCE [%] ↘   | BatchNorm+adapt, mCE [%] ↘  |
|-|-|-|-|-|
|ResNet-50 | 72.0  |72.4 |76.7 |**62.2**|
|ResNet-101 |68.2 |67.6 |69.0 |**59.1**|
|ResNet-152 |67.6 |65.4 |69.3 |**58.0**|

### To reproduce the first table above

Run [`scripts/paper/table1.sh`](scripts/paper/table1.sh): 
```sh
row="2" # This is the row to compute from the table
docker run -v "$IMAGENET_C_PATH":/ImageNet-C:ro \
    -v "$CHECKPOINT_PATH":/checkpoints:ro \
    -v .:/batchnorm \
    -v ..:/deps \
    -it georgepachitariu/robustness:latest \
    bash /batchnorm/scripts/paper/table1.sh $row 2>&1
```
The script file requires 2 dependencies:
1. `IMANGENETC_PATH="/ImageNet-C"`
    This is the path where you store the ImageNet-C dataset. The dataset is described [here](https://github.com/hendrycks/robustness) and you can download it from [here](https://zenodo.org/record/2235448#.YJjcNyaxWcw).
    
2. `CHECKPOINT_PATH="/checkpoints"`
    This is the path where you store our checkpoints.
    You can download them from here: TODO.


## News

- The paper was accepted for poster presentation at NeurIPS 2020.
- A shorter workshop version of our paper was accepted for oral presentation at the [Uncertainty & Robustness in Deep Learning](https://sites.google.com/view/udlworkshop2020) Workshop at ICML '20.
