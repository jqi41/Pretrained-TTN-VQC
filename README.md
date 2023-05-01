# Pre+TTN_VQC (Pretrained Tensor-Train Network + Variational Quantum Circuit)

The package provides an implementation of Pre+TTN-VQC to corroborate our theoretical work.
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
