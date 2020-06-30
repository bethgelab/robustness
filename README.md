## Improving robustness against common corruptions by covariate shift adaptation
Steffen Schneider*, Evgenia Rusak*, Luisa Eck, Oliver Bringmann, Wieland Brendel, Matthias Bethge 

**Coming Soon** -- This repository contains evaluation code for the paper *Improving robustness against common corruptions by covariate shift adaptation*.
We will release the code in the upcoming weeks. To get notified, watch and/or star this repository to get notified of updates!

Today's state-of-the-art machine vision models are vulnerable to image corruptions like blurring or compression artefacts, limiting their performance in many real-world applications. We here argue that popular benchmarks to measure model robustness against common corruptions (like ImageNet-C) underestimate model robustness in many (but not all) application scenarios. The key insight is that in many scenarios, multiple unlabeled examples of the corruptions are available and can be used for unsupervised online adaptation. Replacing the activation statistics estimated by batch normalization on the training set with the statistics of the corrupted images consistently improves the robustness across 25 different popular computer vision models. Using the corrected statistics, ResNet-50 reaches 62.2% mCE on ImageNet-C compared to 76.7% without adaptation. With the more robust AugMix model, we improve the state of the art from 56.5% mCE to 51.0% mCE. Even adapting to a single sample improves robustness for the ResNet-50 and AugMix models, and 32 samples are sufficient to improve the current state of the art for a ResNet-50 architecture. We argue that results with adapted statistics should be included whenever reporting scores in corruption benchmarks and other out-of-distribution generalization settings

## News

- A shorter workshop version of our paper was accepted for oral presentation at the [Uncertainty & Robustness in Deep Learning](https://sites.google.com/view/udlworkshop2020) Workshop at ICML '20.

## Contact

- Website: [domainadaptation.org/batchnorm](https://domainadaptation.org/batchnorm)
- Maintainers: [Evgenia Rusak](https://github.com/EvgeniaAR) & [Steffen Schneider](https://github.com/stes)
