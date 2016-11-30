import time
import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.misc import imresize

split_ratio = 0.3

num_epochs= 2
learning_rate = 0.0001
box_size = 48
batch_size = 10

def readTrafficSigns(rootpath):
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
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels
    
def setup_network(input_var, box_size=48, channel=3):
    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, channel, box_size, box_size),
                                        input_var=input_var)
    #print lasagne.layers.get_output_shape(network)                                    
    # Convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    #print lasagne.layers.get_output_shape(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #Another convolution
    network = lasagne.layers.Conv2DLayer(
           network, num_filters=32, filter_size=(3, 3),
          nonlinearity=lasagne.nonlinearities.rectify)
    #print lasagne.layers.get_output_shape(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))    

    # A fully-connected layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
        
    # final
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=43,
        nonlinearity=lasagne.nonlinearities.softmax)
    
    return network

print("Loading data...")
trainImages, trainLabels = readTrafficSigns('./GTSRB/Final_Training/Images')
nbr_validation = int(round(len(trainLabels) * split_ratio))  # split 30% of the whole data for validation
idx_validation = np.random.choice(len(trainLabels), nbr_validation,
                                  replace=False)  # np.random.permutation(np.arange(5))[:3]
# plt.imshow(trainImages[0])
# print nbr_validation, len(trainImages)
v_img = []  # images for validation
v_labels = []  # corresponding labels for validation
t_img = []  # images for training
t_labels = []  # corresponding labels for training
for idx in range(0, len(trainLabels)):
    trainImages[idx] = imresize(trainImages[idx], (box_size, box_size))
    trainImages[idx] = trainImages[idx]/ np.float32(256)
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

network = setup_network(input_var,box_size)

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
s_epoch = []
s_time = []
s_train_acc = []
s_val_acc = []
# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    train_acc = 0
    start_time = time.time()
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

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  training accuracy:\t\t{:.2f} %".format(
        train_acc / train_batches * 100))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))
    
    s_epoch.append(epoch+1)
    s_time.append(time.time() - start_time)
    s_train_acc.append(train_acc / train_batches * 100)
    s_val_acc.append(val_acc / val_batches * 100)
    
plt.plot(s_epoch,s_train_acc, color="blue", linewidth=2.5, linestyle="-",label="TrainAcc")
plt.plot(s_epoch,s_val_acc, color="red", linewidth=2.5, linestyle="-",label="ValAcc")
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right', frameon=False)

print('Average cost time per epoch:', sum(s_time) / float(len(s_time)))

np.savez('modelparams.npz', *lasagne.layers.get_all_param_values(network))



