# Pre+TTN_VQC (Pretrained Tensor-Train Network + Variational Quantum Circuit)

The package provides an implementation of Pre+TTN-VQC to corroborate our theoretical work on VQC in Refs. [1] and [2].
```
git clone https://github.com/uwjunqi/PreTrained-TTN_VQC
cd PreTrained-TTN_VQC
```

## Installation

The main depencies include *pytorch* and *torchquantum*. Moreover, we need the following packages:

### Our implemented package of TTN:
```
git clone https://github.com/uwjunqi/Pytorch-Tensor-Train-Network.git
cd Pytorch-Tensor-Train-Network
python setup.py install
```

### Torch Quantum 
```
pip3 install torchquantum
```

## Experimental simulations

### Running TTN-VQC
```
python vqc_classifier.py
```

### Running Pre+TTN-VQC
```
python vqc_finetune.py
```

### Running PCA-VQC
```
python pca_vqc_classifier.py
```

## Paper Citation

If you use the codes for your research work, please consider citing the following papers:

[1] Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, Min-Hsiu Hsieh, "Pre-Training Tensor-Train Networks Facilitate Machine Learning with Variational Quantum Circuits," arXiv:2306.03741v1, in Submission.

[2] Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, Min-Hsiu Hsieh, "Theoretical Error Performance Analysis for Variational Quantum Circuit Based Functional Regression," Nature Publishing Group, npj Quantum Information, Vol. 9, no. 4, 2023

[3] Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, "QTN-VQC: An End-to-End Learning Framework for Quantum Neural Networks," arXiv:2110.03861v3, in Submission.
