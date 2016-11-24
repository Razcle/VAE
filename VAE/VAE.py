"""
Defines a variatonal autoencoder
"""
import tensorflow as tf
import numpy as np
import utilities as util


class VariationalAutoencoder(object):

    def __init__(self, network_architecture, learning_rate=0.01,
                 transfer_fnc=tf.nn.tanh, batch_size=100, num_samp=1):
        self.network_architecture = network_architecture
        self.learning_rate = learning_rate
        self.transfer_fnc = transfer_fnc
        self.batch_size = batch_size
        self.num_samp = num_samp

        # Graph input
        self.input_ = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.network_architecture[
                                            'input_dim']],
                                     name="X")
        # Build the Graph
        self._create_network()

        # Add a loss optimizer
        self.opt_op = self._create_loss_opt()

        # Add initialisation opps to all the variables
        init_op = tf.initialize_all_variables()

        # Instantiate Session and initialize
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

    def _create_network(self):

        # Initialise all the weights
        self.weights = self._initialize_all_weights(**self.network_architecture)

        # Creat the recognition model Q(z|x)
        self.rec_mean, self.rec_log_sigma_sq = \
            self._build_model(self.weights['rec_wts'],
                              self.weights['rec_biases'], 'rec')

        # We need a sample from the recognition model to feed the gen model
        self.ltnt_samp = self._sample_posterior()

        # Create the generative model and return the mean and variance
        self.gen_mean, self.gen_log_sigma_sq = \
            self._build_model(self.weights['gen_wts'],
                              self.weights['gen_biases'], 'gen')

    def _sample_posterior(self):
        """ Draw a sample from the distribution Q(latent|oberved)
        """

        latent_dim = self.network_architecture['latent_dim']

        # Sample eps from standard Normal
        eps = tf.random_normal([self.batch_size, latent_dim], 0, 1,
                               dtype=tf.float32)

        # Transform using Z = mean + root_cov*eps
        samp = self.rec_mean + tf.mul(tf.sqrt(tf.exp(self.rec_log_sigma_sq)),
                                      eps)
        return samp

    def _initialize_all_weights(self, input_dim, latent_dim, hdn_dim):

        weights = {'rec_wts': {}, 'rec_biases': {}, 'gen_wts': {},
                   'gen_biases': {}}
        weights['rec_wts'] = {
            'h1': self._init_weight(input_dim, hdn_dim),
            'out_mean': self._init_weight(hdn_dim, latent_dim),
            'out_log_sigma_sq': self._init_weight(hdn_dim, latent_dim)
        }
        weights['rec_biases'] = {
            'h1': self._init_weight(1, hdn_dim),
            'out_mean': self._init_weight(1, latent_dim),
            'out_log_sigma_sq': self._init_weight(1, latent_dim)
        }
        weights['gen_wts'] = {
            'h1': self._init_weight(latent_dim, hdn_dim),
            'out_mean': self._init_weight(hdn_dim, input_dim),
            'out_log_sigma_sq': self._init_weight(hdn_dim, input_dim)
        }
        weights['gen_biases'] = {
            'h1': self._init_weight(1, hdn_dim),
            'out_mean': self._init_weight(1, input_dim),
            'out_log_sigma_sq': self._init_weight(1, input_dim)
        }

        return weights

    def _build_model(self, weights, biases, kind):
        print(kind)

        if kind == 'gen':
            input_ = self.ltnt_samp
        elif kind == 'rec':
            input_ = self.input_

        hdn_layer = self.transfer_fnc(tf.matmul(input_,
                                      weights['h1']) + biases['h1'])

        mean = tf.matmul(hdn_layer, weights['out_mean']) + biases['out_mean']

        log_sigma_sq = (tf.matmul(hdn_layer, weights['out_log_sigma_sq']) +
                        biases['out_log_sigma_sq'])

        return mean, log_sigma_sq

    def _init_weight(self, incoming, out):
        return util.xavier_init(incoming, out)

    def _create_loss_opt(self):

        # Get N X data_dim matrix of covariances
        gen_covs = tf.exp(self.gen_log_sigma_sq)

        # Loss due to reconstruction error log P(data|latent))
        rec_loss = 0.5 * (tf.reduce_sum(
                          self.gen_log_sigma_sq +
                          tf.div(tf.square(self.input_ - self.gen_mean), gen_covs), 1))

        # Loss due to KL penalisation term
        gen_loss = -0.5 * tf.reduce_sum(1 + self.rec_log_sigma_sq
                                        - tf.square(self.rec_mean)
                                        - tf.exp(self.rec_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(rec_loss + gen_loss)

        return tf.train.AdamOptimizer(learning_rate=
                                      self.learning_rate).minimize(self.cost)

    def transform(self, data):
        return self.sess.run(self.rec_mean, feed_dict={self._input: data})

    def generate(self, latent_mean=None):
        if latent_mean is None:
            latent_mean = np.random.normal(size=[1, self.network_architecture['latent_dim']])
        return self.sess.run(self.gen_mean,
                             feed_dict={self.ltnt_samp: latent_mean})

    def reconstruct(self, data):
        self.sess.run(self.gen_mean, feed_dict={self.input_: data})

    def partial_fit(self, data):
        opt, cost = self.sess.run([self.opt_op, self.cost],
                                  feed_dict={self.input_: data})
        return cost


class VariationalAutoencoder_Mnist(VariationalAutoencoder):

    def _create_loss_opt(self):

        self.gen_mean = tf.nn.sigmoid(self.gen_mean)

        # Loss due to reconstruction error log P(data|latent))
        rec_loss = \
            -tf.reduce_sum(self.input_ * tf.log(1e-5 + self.gen_mean) +
                           (1-self.input_) * tf.log(1e-5 + 1 -
                           self.gen_mean),
                           1)

        # Loss due to KL penalisation term
        gen_loss = -0.5 * tf.reduce_sum(1 + self.rec_log_sigma_sq
                                        - tf.square(self.rec_mean)
                                        - tf.exp(self.rec_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(rec_loss + gen_loss)

        return tf.train.AdamOptimizer(learning_rate=
                                      self.learning_rate).minimize(self.cost)
