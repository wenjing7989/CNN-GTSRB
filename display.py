# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:15:16 2016

@author: wenjing
"""

import numpy as np
import matplotlib.pyplot as plt

with np.load('accd.npz') as data:
    time = data['time']
    train_acc = data['trainAcc']
    val_acc = data['valAcc']
    train_err = data['trainErr']
    val_err = data['valErr']
    test_acc = data['testAcc']

epoch = range(len(train_acc))
plt.plot(epoch,train_acc, color="blue", linewidth=2.5, linestyle="-",label="TrainAcc")
plt.plot(epoch,val_acc, color="red", linewidth=2.5, linestyle="-",label="ValAcc")
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right', frameon=False)

plt.show()

plt.plot(epoch,train_err, color="blue", linewidth=2.5, linestyle="-",label="TrainErr")
plt.plot(epoch,val_err, color="red", linewidth=2.5, linestyle="-",label="ValErr")
plt.xlabel('epoch')
plt.ylabel('Error')
plt.legend(loc='upper right', frameon=False)

plt.show()

print time.mean(), train_acc[-1:], val_acc[-1:], test_acc
print train_acc[-10:].mean(),val_acc[-10:].mean()

