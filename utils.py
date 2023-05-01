#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 06:11:14 2023

@author: junqi
"""
import numpy as np

def add_normal_noise(sig, target_snr_db=10):
    sig_watts = sig ** 2
    sig_avg_watts = sig_watts.mean()
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db 
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig_watts))
    
    return sig + noise_volts


def add_laplace_noise(sig, target_snr_db=10):
    sig_watts = sig ** 2
    sig_avg_watts = sig_watts.mean()
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db 
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_volts = np.random.laplace(mean_noise, np.sqrt(noise_avg_watts), len(sig_watts))
    
    return sig + noise_volts

