
import VAE
import numpy as np
from scipy.io import loadmat
import tensorflow as tf


def train(input_data, network_architecture, learning_rate=0.001,
          batch_size=50, training_epochs=10, display_step=10, kind='frey'):
    if kind == 'frey':
        vae = VAE.VariationalAutoencoder(network_architecture,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size)
    if kind == 'mnist':
        vae = VAE.VariationalAutoencoder_Mnist(network_architecture,
                                         learning_rate=learning_rate,
                                         batch_size=batch_size)

    n_samples = np.shape(input_data)[0]
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for batch_xs in iterate_minibatches(input_data, batch_size):

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)

            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae


def load_frey_faces(test_prop=0.2):

    data = np.transpose(loadmat('../Data/frey_rawface.mat')['ff'])
    data = np.array([n / float(max(n) + 1) for n in data])

    N = data.shape[0]

    train = data[int(N*test_prop):]
    test = data[: int(N*test_prop)]

    return train, test


def iterate_minibatches(X, batch_size, y=None, shuffle=True):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start_idx in range(0, n - batch_size + 1, batch_size):
        idx_slice = idx[start_idx:start_idx+batch_size]
        mini_batch = X[idx_slice]
        if y is not None:
            mini_batch = (mini_batch, y[idx_slice])
        yield mini_batch


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.Variable(tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32))
