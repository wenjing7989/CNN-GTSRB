import theano
import theano.tensor as T
import lasagne
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.misc import imresize


box_size = 48

def readTestData(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    prefix = rootpath + '/'
    gtFile = open(prefix + '/GT-final_test.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return images, labels

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


print("Load data...")
testImages,testLabels = readTestData('./GTSRB/Final_Test/Images/')
for idx in range(0, len(testImages)):
    testImages[idx] = imresize(testImages[idx], (box_size, box_size))/ np.float32(256)

# Prediction
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
network = setup_network(input_var, box_size)
# Validation
val_prediction = lasagne.layers.get_output(network, deterministic=True)
val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_var).mean()
val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var), dtype=theano.config.floatX)
val_fn = theano.function([input_var, target_var], [val_loss, val_acc],allow_input_downcast=True)

# load model
with np.load('modelparams.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

# start testing
print("Start testing...")
# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(testImages, testLabels, 1, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

