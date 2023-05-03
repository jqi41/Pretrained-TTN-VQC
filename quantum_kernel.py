#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:08:39 2023

@author: junqi
"""

import numpy as np
import torch

from torchquantum.functional import func_name_dict 
import torchquantum as tq

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Build an Ansatz, consist of a unitary and its transpose conjugation
class KernalAnsatz(tq.QuantumModule):
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, source, target):
        self.q_device = q_device
        self.q_device.reset_states(source.shape[0])
        for info in self.func_list:
            if tq.op_name_dict[info['func']].num_params > 0:
                params = source[:, info['input_idx']]
            else:
                params = None
            func_name_dict[info['func']](
                self.q_device,
                wires=info['wires'],
                params=params,
            )
        for info in reversed(self.func_list):
            if tq.op_name_dict[info['func']].num_params > 0:
                params = -target[:, info['input_idx']]
            else:
                params = None
            func_name_dict[info['func']](
                self.q_device,
                wires=info['wires'],
                params=params,
            )
     
class Quantum_Kernel(tq.QuantumModule):
    def __init__(self, 
                 n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.ansatz = KernalAnsatz(enc_cnt)
        
    def forward(self, source, target, use_qiskit=False):
        # bsz = 1
        source = source.reshape(1, -1)
        target = target.reshape(1, -1)
        self.ansatz(self.q_device, source, target)
        result = torch.abs(self.q_device.states.view(-1)[0])
        
        return result


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    X = X[:100]
    y = y[:100]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y_scaled = 2 * (y - 0.5)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

    kernel_function = Quantum_Kernel()
    def kernel_matrix(A, B):
        return np.array([[kernel_function(a, b) for b in B] for a in A])
    
    svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
    predictions = svm.predict(X_test)
    
    print(accuracy_score(predictions, y_test))
