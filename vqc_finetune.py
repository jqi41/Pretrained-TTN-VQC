#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:53:40 2023

@author: junqi
"""
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np 

# Tensor Train Network
from tc.tc_fc import TTLinear 

# Variational Quantum Circuit
from vqc import VQC  

# Torch Quantum
import torchquantum as tq
from torchquantum.dataset import MNIST  

from utils import add_normal_noise, add_laplace_noise


class VQC_FINETUNE(nn.Module):
    def __init__(self, 
                 input_dims: int, 
                 n_qubits: int = 8,
                 n_class: int = 2,
                 n_qlayers: int = 1,
                 tt_shape = [[7, 16, 7], [2, 2, 2]],
                 tt_rank = [1, 2, 2, 1],
                 pre_net: TTLinear = None):
        super(VQC_FINETUNE, self).__init__()
        self.dev = tq.QuantumDevice(n_wires=n_qubits)
        self.vqc = VQC(n_wires=n_qubits, n_qlayers=n_qlayers)
        self.post_net = nn.Linear(n_qubits, n_class)

        if pre_net:
            self.pre_net = TTLinear(inp_modes=tt_shape[0], 
                                    out_modes=tt_shape[1], 
                                    tt_rank=tt_rank)
            self.pre_net.load_state_dict(pre_net.state_dict())
        else:
            self.pre_net = None
        
        
    def forward(self, input_features, pre_net: TTLinear = None):
        if self.pre_net:
            pre_out = self.pre_net(input_features)
        else:
            pre_out = pre_net(input_features)
        q_in = torch.sigmoid(pre_out) * np.pi / 2.0
        q_out = self.vqc(q_in, self.dev)
        q_class = self.post_net(q_out)
        
        return F.log_softmax(q_class, dim=1)
    

# The console of training a TTN-VQC model
def train(data_flow: dict, 
          model: VQC_FINETUNE,
          device: tq.QuantumDevice,
          optimizer: torch.optim,
          pre_net: TTLinear = None):
    batch_idx = 1
    for feed_dct in dataflow['train']:
        inputs = feed_dct['image'].reshape((-1, args.input_dims)).to(device)
        targets = feed_dct['digit'].to(device)
        
        outputs = model(inputs, pre_net)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"batch: {batch_idx}, loss: {loss.item()}\n", end='\r')
        batch_idx += 1


def valid_test(dataflow: dict, 
               split, 
               model: VQC_FINETUNE, 
               device: tq.QuantumDevice,
               pre_net: TTLinear = None,
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
            outputs = model(inputs, pre_net)
            
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
    parser.add_argument('--tt_shape', type=list[list[int]], default=[[7, 16, 7], [2, 2, 2]])
    parser.add_argument('--tt_ranks', type=list[int], default=[1, 2, 2, 1])
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
    
    # Loading the TTN model
    pre_net = TTLinear(inp_modes=[7, 16, 7], 
                       out_modes=[2, 2, 2], 
                       tt_rank=[1, 2, 2, 1])
    pre_net_dict = pre_net.state_dict()
    
    save_ttn_vqc = torch.load('models/model_ttn-vqc.pth')
    for k, v in save_ttn_vqc.items():
        s = k.split('.')
        if len(s) == 3:
            k = s[1] + '.' + s[2]
        else:
            k = s[-1]
        if k in pre_net_dict.keys():
            pre_net_dict[k] = v
            
    pre_net.load_state_dict(pre_net_dict)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = VQC_FINETUNE(args.input_dims, 
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
        train(dataflow, model, device, optimizer, pre_net)
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        
        # valid
        valid_test(dataflow, 'valid', model, device, pre_net, args.add_test_noise)
        scheduler.step()
        
    # test
    valid_test(dataflow, 'test', model, device, pre_net, args.add_test_noise)
    
    
    
