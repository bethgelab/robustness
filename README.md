# Robustness and Adaptation on ImageNet scale

<table style="width:100%">
  <tr>
    <td><img src="https://user-images.githubusercontent.com/727984/116836295-4128cf80-abc6-11eb-8457-8502a3c59427.png"/></td>
    <td><img src="https://domainadaptation.org/selflearning/img/overview.svg"/></td>
  </tr>
  <tr>
    <td>Batch Norm adaptation improves corruption robustness on ImageNet-C.</td>
    <td>Self-learning during test time improves robustness across ImageNet-C,-A,-R.</td>
  </tr>
</table>

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
