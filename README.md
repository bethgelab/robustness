# Robustness and Adaptation on ImageNet scale

## News

- April 2021: The pre-print for "Adapting ImageNet-scale models to complex distribution shifts with self-learning" is now available on arXiv: arxiv.org/abs/2104.12928
- September 2020: The BatchNorm adaptation paper was accepted for poster presentation at NeurIPS 2020.
- June 2020: The pre-print for "Improving robustness against common corruptions by covariate shift adaptation" is available on arXiv: arxiv.org/abs/2006.16971.pdf

## Papers

### Batchnorm Adaptation

We propose to go beyond the assumption of a single sample from the target domain when evaluating robustness. Re-computing BatchNorm statistics is a simple baseline algorithm for improving the corruption error up to 14% points over a wide range of models, when access to more than a single sample is possible.

- Code release: `batchnorm/`
- Web: https://domainadaptation.org/batchnorm/

### Robust Pseudo-Labeling

Test-time adaptation with self-learning improves robustness of large-scale computer vision models on ImageNet-C, -R, and -A.

- Code release: `selflearning/`
- Web: http://domainadaptation.org/selflearning/


## Contact

- [Evgenia Rusak](https://github.com/EvgeniaAR)
- [Steffen Schneider](https://stes.io)
- [George Pachitariu](https://github.com/georgepachitariu)