# Calibrating Multimodal Learning (CML)

This repository contains the code of Calibrating Multimodal Learning (CML). Here we provide a demo and detailed instructions for constructing CML on several dataset.

## Requirment

Pytorch 1.3.0

Python 3

scikit-learn

numpy

scipy

## Example Experiments

This repository contains experimental code mentioned in the paper.

## Training and Testing

- train miwae without mcca:

  python miwae/train.py 

- train miwae with mcca:

  python miwae/train.py --beta 10

- test miwae without mcca:

  python miwae/test_metrics.py 

- test miwae with mcca:

  python miwae/test_metrics.py --beta 10
