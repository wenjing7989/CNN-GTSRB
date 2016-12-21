# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:28:07 2016

@author: wenjing
"""

import time
import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.misc import imresize

num_epochs= 2
learning_rate = 0.0001
batch_size = 1
roi = False
box_size = 48

def readTrafficSigns(rootpath, roi= False):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            temp=plt.imread(prefix + row[0])# the 1th column is the filename            
            if roi == True:
                images.append(temp[int(row[4]):int(row[6]),int(row[3]):int(row[5]),:])
            else:
                images.append(temp) 
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels
    
def readTestData(rootpath, roi = False):
    images = [] # images
    labels = [] # corresponding labels
    prefix = rootpath + '/'
    gtFile = open(prefix + '/GT-final_test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        temp = plt.imread(prefix + row[0]) # the 1th column is the filename        
        if roi == True:
            images.append(temp[int(row[4]):int(row[6]),int(row[3]):int(row[5]),:])        
        else:
            images.append(temp) # the 1th column is the filename
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return images, labels
    
def network(input_var, box_size=48,num_channel=3 ):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, num_channel, box_size, box_size),
                                        input_var=input_var)
                                        
    # Convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #Another convolution
    network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(3, 3),
          nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    #Another convolution
    network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(3, 3),
          nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # final
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=43,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

print("Loading data...")
trainImages, trainLabels = readTrafficSigns('./GTSRB/Final_Training/Images',roi)
testImages,testLabels = readTestData('./GTSRB/Final_Test/Images/',roi)

for idx in range(0, len(testImages)):
    testImages[idx] = imresize(testImages[idx], (box_size, box_size))/ np.float32(256)
    
# split the data   
with np.load('index0.3.npz') as data:
    idx_validation = data['idx_validation']

v_img = []  # images
v_labels = []  # corresponding labels
t_img = []
t_labels = []
for idx in range(0, len(trainLabels)):
    trainImages[idx] = imresize(trainImages[idx], (box_size, box_size))/ np.float32(256)
    if idx in idx_validation:
        v_img.append(trainImages[idx])                        
        v_labels.append(trainLabels[idx])                        
    else:                            
        t_img.append(trainImages[idx])                            
        t_labels.append(trainLabels[idx])



print("Building model and compiling functions...")


# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = network(input_var,box_size)

# Training
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates,allow_input_downcast=True)

# Validation
val_prediction = lasagne.layers.get_output(network, deterministic=True)
val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var).mean()
val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var), dtype=theano.config.floatX)
val_fn = theano.function([input_var, target_var], [val_loss, val_acc],allow_input_downcast=True)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield np.transpose(inputs[excerpt],(0,3,1,2)), targets[excerpt]

# Finally, launch the training loop.
print("Starting training...")

s_train_acc = []
s_val_acc = []
s_train_err = []
s_val_err = []
s_time = []
# We iterate over epochs:
for epoch in range(num_epochs):
    start_time = time.time()
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    train_acc = 0
    for batch in iterate_minibatches(t_img, t_labels, batch_size, shuffle=True):
        inputs, targets = batch
        err, acc = train_fn(inputs, targets)
        train_err += err
        train_acc += acc
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(v_img, v_labels, batch_size, shuffle=True):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
    
    s_train_acc.append(train_acc / train_batches * 100)
    s_val_acc.append(val_acc / val_batches * 100)
    s_train_err.append(train_err / train_batches)
    s_val_err.append(val_err / val_batches)
    s_time.append(time.time()-start_time)

test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(testImages, testLabels, batch_size, shuffle=True):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
    

# save the model
np.savez('model', *lasagne.layers.get_all_param_values(network))
np.savez('acc', testAcc = test_acc / test_batches * 100, testErr = test_err / test_batches, trainErr = s_train_err, valErr = s_val_err, time = s_time, trainAcc = s_train_acc, valAcc = s_val_acc)
