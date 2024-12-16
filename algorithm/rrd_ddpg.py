import numpy as np
import tensorflow as tf

from algorithm import basis_algorithm_collection

from utils.tf_utils import get_vars, get_reg_loss
def RRD_ddpg(args):
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    class RandomizedReturnDecomposition(basis_alg_class):
        def __init__(self, args):
            super().__init__(args)
        def create_inputs(self):
            super().create_inputs()

            self.rrd_raw_obs_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_raw_obs_next_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_acts_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.acts_dims)
            self.rrd_rews_ph = tf.placeholder(tf.float32, [None, 1])
            if self.args.rrd_bias_correction:
                self.rrd_var_coef_ph = tf.placeholder(tf.float32, [None, 1])

        def create_normalizer(self):
            super().create_normalizer()

            if self.args.obs_normalization:
                self.rrd_obs_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_ph)
                self.rrd_obs_next_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_next_ph)
            else:
                self.rrd_obs_ph = self.rrd_raw_obs_ph
                self.rrd_obs_next_ph = self.rrd_raw_obs_next_ph
        def create_network(self):
            def mlp_policy(obs_ph):
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    pi_dense1 = tf.layers.dense(obs_ph, 400, activation=tf.nn.relu, name='pi_dense1')
                    pi_dense2 = tf.layers.dense(pi_dense1, 300, activation=tf.nn.relu, name='pi_dense2')
                    pi = tf.layers.dense(pi_dense2, self.args.acts_dims[0], activation=tf.nn.tanh, name='pi')
                return pi

            def mlp_value(obs_ph, acts_ph):
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    q_dense1 = tf.layers.dense(obs_ph, 400, activation=tf.nn.relu, name='q_dense1')
                    q_dense2 = tf.layers.dense(tf.concat([q_dense1, acts_ph], axis=1), 300, activation=tf.nn.relu, name='q_dense2')
                    q = tf.layers.dense(q_dense2, 1, name='q')
                return q

            with tf.variable_scope('main'):
                with tf.variable_scope('policy'):
                    self.pi = mlp_policy(self.obs_ph)
                with tf.variable_scope('value', regularizer=tf.contrib.layers.l2_regularizer(self.args.q_reg)):
                    self.q = mlp_value(self.obs_ph, self.acts_ph)
                with tf.variable_scope('value', reuse=True):
                    self.q_pi = mlp_value(self.obs_ph, self.pi)

            with tf.variable_scope('target'):
                with tf.variable_scope('policy'):
                    self.pi_t = mlp_policy(self.obs_next_ph)
                with tf.variable_scope('value'):
                    self.q_t = mlp_value(self.obs_next_ph, self.pi_t)
        
        def create_operators(self):
            self.pi_loss = -tf.reduce_mean(self.q_pi)
            self.pi_optimizer = tf.train.AdamOptimizer(self.args.pi_lr)
            self.pi_train_op = self.pi_optimizer.minimize(self.pi_loss, var_list=get_vars('main/policy'))
            self.rews_ph=tf.reduce_mean(self.q-tf.stop_gradient((1.0-self.done_ph)*self.args.gamma*self.q_t))
            self.q_loss = tf.reduce_mean(tf.square(self.rrd_rews_ph-self.rews_ph))
            self.q_reg_loss = get_reg_loss('main/value')
            self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr)
            self.q_train_op = self.q_optimizer.minimize(self.q_loss + self.q_reg_loss, var_list=get_vars('main/value'))

            self.target_update_op = tf.group([
                v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.target_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

            self.init_op = tf.global_variables_initializer()

        def feed_dict(self, batch):
            batch_size = np.array(batch['obs']).shape[0]
            basis_feed_dict = super().feed_dict(batch)
            del basis_feed_dict[self.rews_ph]
            def one_hot(idx):
                idx = np.array(idx)
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx = np.reshape(idx, [batch_size*sample_size])
                res = np.zeros((batch_size*sample_size, self.acts_num), dtype=np.float32)
                res[np.arange(batch_size*sample_size),idx] = 1.0
                res = np.reshape(res, [batch_size, sample_size, self.acts_num])
                return res
            rrd_feed_dict = {
                **basis_feed_dict, **{
                    self.rrd_raw_obs_ph: batch['rrd_obs'],
                    self.rrd_raw_obs_next_ph: batch['rrd_obs_next'],
                    self.rrd_acts_ph: batch['rrd_acts'] if self.args.env_category!='atari' else one_hot(batch['rrd_acts']),
                    self.rrd_rews_ph: batch['rrd_rews'],
                }
            }
            if self.args.rrd_bias_correction:
                rrd_feed_dict[self.rrd_var_coef_ph] = batch['rrd_var_coef']
            return rrd_feed_dict

    return RandomizedReturnDecomposition(args)
