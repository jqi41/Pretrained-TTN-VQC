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

# PyTorch
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import CosineAnnealingLR

# Torch Quantum
import torchquantum as tq
from torchquantum.datasets import MNIST

# Tensor Train Network
from tc.tc_fc import TTLinear 

# Variational Quantum Circuit
from vqc import VQC
        
from utils import add_normal_noise, add_laplace_noise

class TTN_VQC(nn.Module):
    def __init__(self, 
                 input_dims: int, 
                 n_qubits: int = 8,
                 n_class: int = 3,
                 n_qlayers: int = 1,
                 use_tt: bool = True, 
                 tt_shape = [[7, 16, 7], [2, 2, 2]],
                 tt_rank = [1, 2, 2, 1]):
        super(TTN_VQC, self).__init__()
        self.dev = tq.QuantumDevice(n_wires=n_qubits)
        if use_tt:
            # Using a tensor-train layer
            assert np.prod(tt_shape[1]) == n_qubits
            assert len(tt_shape[0]) == len(tt_rank) - 1
            self.pre_net = TTLinear(inp_modes=tt_shape[0], 
                                    out_modes=tt_shape[1], 
                                    tt_rank=tt_rank)
        else:
            # Using a classical feed-forward layer
            self.pre_net = nn.Linear(input_dims, n_qubits)
        self.vqc = VQC(n_wires=n_qubits, n_qlayers=n_qlayers)
        self.post_net = nn.Linear(n_qubits, n_class)
        
    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.sigmoid(pre_out) * np.pi / 2.0
        q_out = self.vqc(q_in, self.dev)
        q_class = self.post_net(q_out)
        
        return F.log_softmax(q_class, dim=1)


# The console of training a TTN-VQC model
def train(data_flow: dict, 
          model: TTN_VQC,
          device: tq.QuantumDevice,
          optimizer: torch.optim):
    batch_idx = 1
    for feed_dct in dataflow['train']:
        inputs = feed_dct['image'].reshape((-1, args.input_dims)).to(device)           
        targets = feed_dct['digit'].to(device)
        
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"batch: {batch_idx}, loss: {loss.item()}\n", end='\r')
        batch_idx += 1


def valid_test(dataflow: dict, 
               split, 
               model: TTN_VQC, 
               device: tq.QuantumDevice,
               add_noise: bool = False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            if not add_noise:
                inputs = feed_dict['image'].reshape((-1, args.input_dims)).to(device)
            else:
                inputs = feed_dict['image'].reshape((-1, args.input_dims))
                for idx, _input in enumerate(inputs):
                    if args.noise_kind == 'normal':
                        inputs[idx, :] = add_normal_noise(_input, args.noise_snr)
                    else:
                        inputs[idx, :] = add_laplace_noise(_input, args.noise_snr)
                inputs = inputs.to(device)
            targets = feed_dict['digit'].to(device)
            outputs = model(inputs)
            
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
    parser.add_argument('--use_tt', type=bool, default=True, 
                    help='whether we use tensor-train network?')
    parser.add_argument('--tt_shape', type=list[list[int]], default=[[7, 16, 7], [2, 2, 2]],
                    help='the tensor-train shape')
    parser.add_argument('--tt_ranks', type=list[int], default=[1, 2, 2, 1], 
                    help='the ranks of tensor-train')
    parser.add_argument('--add_test_noise', type=bool, default=True, 
                    help='whether we add noise to test data')
    parser.add_argument('--noise_snr', type=int, default=6, 
                    help='noise snr level')
    parser.add_argument('--noise_kind', metavar='DIR', default='laplace', 
                    help='noise kind')

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
    
    model = TTN_VQC(args.input_dims, 
                    n_qubits=args.n_qubits, 
                    n_class=args.n_class, 
                    n_qlayers=args.n_qlayers,
                    use_tt=args.use_tt,
                    tt_shape=args.tt_shape, 
                    tt_rank=args.tt_ranks).to(device)
    
    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    if args.static:
    # optionally to switch to the static mode, which can bring speedup on training
        model.VQC.static_on(wires_per_block=args.wires_per_block)
    
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        
        # valid
        valid_test(dataflow, 'valid', model, device, args.add_test_noise)
        scheduler.step()
        
    # test
    valid_test(dataflow, 'test', model, device, args.add_test_noise)
    
    
    # Saving the TTN-VQC model as a whole
    path = os.path.join(os.path.abspath(''), 'models/model_ttn-vqc.pth')
    torch.save(model.state_dict(), path) 
    