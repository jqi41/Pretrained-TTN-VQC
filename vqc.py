#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 09:20:50 2023

@author: junqi
"""
from typing import Tuple, Callable

# PyTorch
import torch

# Torch Quantum
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict 


class VQC(tq.QuantumModule):
    """
    A variational quantum circuit (VQC, Variational Ansatz) consists of three parts: 
    (1) tensor product encoder; (2) variational ansatz; (3) measurement. We can 
    create an encoder by passing a list of gates to tq.GeneralEncoder. Each entry 
    in the list contains input_idx, func, and wires. Here, each qubit has a Pauli-Y 
    gate, which can convert the classical input data into the quantum states. Then, 
    we choose the variational ansatz such that each quantum channel is mutually entangled 
    and Pauli-X,Y,Z gates rotated by arbitrary angles. Finally, we perform Pauli-Z 
    measurements on each qubit on each qubit by creating a tq.MeasureAll module 
    and passing tq.PauliZ to it. The measure function will return four expectation 
    values from the qubits. 
    """
    def __init__(self, 
                 n_wires: int = 8,
                 n_qlayers: int = 1):
        super().__init__()
        self.n_wires = n_wires 
        self.n_qlayers = n_qlayers
            
        # Setting up tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)
        
        # We create trainable model parameters, which are stored in dict 
        self.params_rx_dct = {}
        self.params_ry_dct = {}
        self.params_rz_dct = {}
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[i + k*self.n_wires] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[i + k*self.n_wires] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[i + k*self.n_wires] = tq.RY(has_params=True, trainable=True)
        # The observables are Hermitian operator based on Pauli-Z 
        self.measure = tq.MeasureAll(tq.PauliZ)
        
    @tq.static_support 
    def forward(self, 
                x: torch.Tensor, 
                q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static model,
        we need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and 
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[i + k*self.n_wires](self.q_device, wires=i)
                self.params_ry_dct[i + k*self.n_wires](self.q_device, wires=i)
                self.params_rz_dct[i + k*self.n_wires](self.q_device, wires=i)
            
            for i in range(self.n_wires):
                if i == self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[i, 0], static=self.static_mode,
                             parent_graph=self.graph)
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode,
                             parent_graph=self.graph)
            
        return (self.measure(self.q_device))
    
    
class FF_VQC(tq.QuantumModule):
    """Training a VQC ansatz by using Forward-Forward algorithm
    """
    def __init__(self,
                 n_wires: int,
                 n_qlayers: int,
                 optimizer: torch.optim,
                 layer_optim_learning_rate: float,
                 threshold: float,
                 loss_fn: Callable):
        super().__init__()
        self.n_wires = n_wires 
        self.n_qlayers = n_qlayers
        self.optimizer = optimizer(self.parameters(), lr=layer_optim_learning_rate)
        self.threshold = threshold 
        self.loss_fn = loss_fn
            
        # Setting up tensor product encoder
        enc_cnt = list()
        for i in range(self.n_wires):
            cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            enc_cnt.append(cnt)
        self.encoder = tq.GeneralEncoder(enc_cnt)
            
        # We create trainable model parameters, which are stored in dict 
        self.params_rx_dct = {}
        self.params_ry_dct = {}
        self.params_rz_dct = {}
                
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[i + k*self.n_wires] = tq.RX(has_params=True, trainable=True)
                self.params_ry_dct[i + k*self.n_wires] = tq.RY(has_params=True, trainable=True)
                self.params_rz_dct[i + k*self.n_wires] = tq.RY(has_params=True, trainable=True)
        # The observables are Hermitian operator based on Pauli-Z 
        self.measure = tq.MeasureAll(tq.PauliZ)
            
    @tq.static_support
    def forward(self, 
                x: torch.Tensor,
                q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static model,
        we need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
            static=self.static_mode and 
            parent_graph=self.graph
            to all the tqf functions, such as tqf.hadamard below
        """
        # normalize the input
        x = x / (x.norm(2, 1, keepdim=True) + 1e-8) 
            
        self.q_device = q_device 
        self.encoder(self.q_device, x)
            
        for k in range(self.n_qlayers):
            for i in range(self.n_wires):
                self.params_rx_dct[i + k*self.n_wires](self.q_device, wires=i)
                self.params_ry_dct[i + k*self.n_wires](self.q_device, wires=i)
                self.params_rz_dct[i + k*self.n_wires](self.q_device, wires=i)
                
            for i in range(self.n_wires):
                if i == self.n_wires-1:
                    tqf.cnot(self.q_device, wires=[i, 0], static=self.static_mode,
                             parent_graph=self.graph)
                else:
                    tqf.cnot(self.q_device, wires=[i, i+1], static=self.static_mode,
                             parent_graph=self.graph)
                
        return (self.measure(self.q_device))
            
            
    def train_layer(self,
                    X_pos: torch.Tensor,
                    X_neg: torch.Tensor,
                    before: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Train a VQC layer with FF algorithm
        Args:
            X_pos (torch.Tensor): batch of positive examples
            X_neg (torch.Tensor): batch of negative examples
            before (bool): if True, successive layers get previous and negative predictions and loss value
        returns:
            Tuple[torch.Tensor, torch.Tensor, int]: batch of positive and negative predictions and loss value
        """
        X_pos_out = self.forward(X_pos)
        X_neg_out = self.forward(X_neg)
            
        loss = self.loss_fn(X_pos_out, X_neg_out, self.threshold)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        if before:
            return X_pos_out.detach(), X_neg_out.detach(), loss.item()
        else:
            return self.forward(X_pos).detach(), self.forward()
