#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:24:40 2023

@author: junqi
"""
import os
import argparse
import random
import numpy as np 

# Sklearn
from sklearn.decomposition import PCA

# PyTorch
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import CosineAnnealingLR

# Torch Quantum
import torchquantum as tq
from torchquantum.datasets import MNIST

# Variational Quantum Circuit
from vqc import VQC
        

class PCA_VQC(nn.Module):
    def __init__(self, 
                 input_dims: int, 
                 n_qubits: int = 8,
                 n_class: int = 3,
                 n_qlayers: int = 1):
        super(PCA_VQC, self).__init__()
        self.dev = tq.QuantumDevice(n_wires=n_qubits)
        self.vqc = VQC(n_wires=n_qubits, n_qlayers=n_qlayers)
        self.post_net = nn.Linear(n_qubits, n_class)
        
    def forward(self, input_features):
        q_in = torch.sigmoid(input_features) * np.pi / 2.0
        q_out = self.vqc(q_in, self.dev)
        q_class = self.post_net(q_out)
        
        return F.log_softmax(q_class, dim=1)
    

# The console of training a TTN-VQC model
def train(data_flow: dict, 
          model: VQC,
          device: tq.QuantumDevice,
          optimizer: torch.optim,
          pca_vqc: PCA):
    batch_idx = 1
    for feed_dct in dataflow['train']:
        inputs = feed_dct['image'].reshape((-1, args.input_dims)).to(device)
        targets = feed_dct['digit'].to(device)
        train_inputs = torch.tensor(pca_vqc.fit_transform(inputs))
        
        outputs = model(train_inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"batch: {batch_idx}, loss: {loss.item()}\n", end='\r')
        batch_idx += 1


def valid_test(dataflow: dict, 
               split, 
               model: VQC, 
               device: tq.QuantumDevice,
               pca_vqc: PCA):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].reshape((-1, args.input_dims)).to(device)
            targets = feed_dict['digit'].to(device)
            valid_inputs = torch.tensor(pca_vqc.fit_transform(inputs))
            outputs = model(valid_inputs)
            
            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)
        
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    
    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")   
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12,
                    help='number of training epochs')
    parser.add_argument('--static', action='store_true', 
                    help='compute with static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                    help='wires per block int static mode')
    parser.add_argument('--input_dims', type=int, default=784,
                    help='input dimensions')
    parser.add_argument('--digits_of_interest', type=list[int], default=[1, 3],
                    help='digits of interest')
    parser.add_argument('--n_class', type=int, default=2,
                    help='number of classes')
    parser.add_argument('--n_qubits', type=int, default=8,
                    help='number of qubits')
    parser.add_argument('--n_qlayers', type=int, default=1,
                    help='number of PQC layers')


    args = parser.parse_args()
    
    seed = 4321
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=args.digits_of_interest,
        n_test_samples=1000,
    )
    dataflow = dict()
    
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    pca = PCA(n_components=args.n_qubits)
    
    model = PCA_VQC(args.input_dims, 
                n_qubits=args.n_qubits, 
                n_class=args.n_class, 
                n_qlayers=args.n_qlayers).to(device)
    
    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    if args.static:
    # optionally to switch to the static mode, which can bring speedup on training
        model.VQC.static_on(wires_per_block=args.wires_per_block)
    
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer, pca)
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        
        # valid
        valid_test(dataflow, 'valid', model, device, pca)
        scheduler.step()
        
    # test
    valid_test(dataflow, 'test', model, device, pca)
    
    
    # Saving the TTN-VQC model as a whole
    path = os.path.join(os.path.abspath(''), 'models/model_ttn-vqc.pth')
    torch.save(model.state_dict(), path) 
    
