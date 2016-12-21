# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:56:00 2016

@author: wenjing
"""
import numpy as np

#generate validation index
ratio = 0.01
length = 39209
nbr_validation = int(round(length * ratio))# len(trainLabels) = 39209, nbr_v = 11763, nbr_t = 27446
idx_validation = np.random.choice(length, nbr_validation, replace=False)  # np.random.permutation(np.arange(5))[:3]                
np.savez('index'+format(ratio), idx_validation = idx_validation)