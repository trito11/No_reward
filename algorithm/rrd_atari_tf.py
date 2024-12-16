import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from algorithm import basis_algorithm_collection
from utils.tf_utils import Normalizer

import numpy as np
import tensorflow as tf
from utils.tf_utils import Normalizer

class Base:
    def __init__(self, args):
        self.args = args
        self.create_model()

    def create_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def create_inputs(self):
        self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.raw_obs_next_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
        self.acts_ph = tf.placeholder(tf.float32, [None]+self.args.acts_dims)
        self.rews_ph = tf.placeholder(tf.float32, [None, 1])
        self.done_ph = tf.placeholder(tf.float32, [None, 1])

    def create_normalizer(self):
        if self.args.obs_normalization:
            with tf.variable_scope('normalizer'):
                self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
            self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
            self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)
        else:
            self.obs_normalizer = None
            self.obs_ph = self.raw_obs_ph
            self.obs_next_ph = self.raw_obs_next_ph

    def create_network(self):
        raise NotImplementedError

    def create_operators(self):
        raise NotImplementedError

    def create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_session()
            self.create_inputs()
            self.create_normalizer()
            self.create_network()
            self.create_operators()
        self.init_network()

    def init_network(self):
        self.sess.run(self.init_op)
        self.sess.run(self.target_init_op)

    def normalizer_update(self, batch):
        if self.args.obs_normalization:
            self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def target_update(self):
        self.sess.run(self.target_update_op)

    def save_model(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path)

    def load_model(self, load_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, load_path)


def RRD_atari_tf(args):
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    class RandomizedReturnDecomposition(basis_alg_class):
        def __init__(self, args):
            super().__init__(args)

            self.train_info_r = {
                'Q_total_loss': self.q_total_loss
            }
            if args.rrd_bias_correction:
                self.train_info_r['Q_var'] = self.r_var
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}

        def create_inputs(self):
            
            
            self.rrd_raw_obs_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_raw_obs_next_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.obs_dims)
            self.rrd_acts_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size]+self.args.acts_dims)
            
            self.rrd_rews_ph = tf.placeholder(tf.float32, [None, 1])

            self.rrd_done_ph = tf.placeholder(tf.float32, [None, self.args.rrd_sample_size,1])

            self.raw_obs_ph = tf.placeholder(tf.float32, [None]+self.args.obs_dims)
            
            if self.args.rrd_bias_correction:
                self.rrd_var_coef_ph = tf.placeholder(tf.float32, [None, 1])

        def create_normalizer(self):
            if self.args.obs_normalization:
                with tf.variable_scope('normalizer'):
                    self.obs_normalizer = Normalizer(self.args.obs_dims, self.sess)
                self.obs_ph = self.obs_normalizer.normalize(self.raw_obs_ph)
                # self.obs_next_ph = self.obs_normalizer.normalize(self.raw_obs_next_ph)
            else:
                self.obs_normalizer = None
                self.obs_ph = self.raw_obs_ph
                # self.obs_next_ph = self.raw_obs_next_ph
    
            if self.args.obs_normalization:
                self.rrd_obs_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_ph)
                self.rrd_obs_next_ph = self.obs_normalizer.normalize(self.rrd_raw_obs_next_ph)
            else:
                self.rrd_obs_ph = self.rrd_raw_obs_ph
                self.rrd_obs_next_ph = self.rrd_raw_obs_next_ph

        def create_network(self):
            def mlp_value(obs_ph):
                flatten = (len(list(obs_ph.shape))==len(self.args.obs_dims)+2)
                if flatten:
                    obs_ph = tf.reshape(obs_ph, [-1]+self.args.obs_dims[:-1]+[self.args.obs_dims[-1]])
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    q_dense1 = tf.layers.dense(obs_ph, 256, activation=tf.nn.relu, name='q_dense1')
                    q_dense2 = tf.layers.dense(q_dense1, 256, activation=tf.nn.relu, name='q_dense2')
                    q = tf.layers.dense(q_dense2, self.acts_num, name='q')
                    if flatten:
                        q = tf.reshape(q, [-1, self.args.rrd_sample_size, self.acts_num])   
                return q

            def conv_value(obs_ph):
                flatten = (len(list(obs_ph.shape))==len(self.args.obs_dims)+2)
                if flatten:
                    obs_ph = tf.reshape(obs_ph, [-1]+self.args.obs_dims[:-1]+[self.args.obs_dims[-1]])
                with tf.variable_scope('net', initializer=tf.contrib.layers.xavier_initializer()):
                    q_conv1 = tf.layers.conv2d(obs_ph, 32, 8, 4, 'same', activation=tf.nn.relu, name='q_conv1')
                    q_conv2 = tf.layers.conv2d(q_conv1, 64, 4, 2, 'same', activation=tf.nn.relu, name='q_conv2')
                    q_conv3 = tf.layers.conv2d(q_conv2, 64, 3, 1, 'same', activation=tf.nn.relu, name='q_conv3')
                    q_conv3_flat = tf.layers.flatten(q_conv3)

                    q_dense_act = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_act')
                    q_act = tf.layers.dense(q_dense_act, self.acts_num, name='q_act')

                    if self.args.dueling:
                        q_dense_base = tf.layers.dense(q_conv3_flat, 512, activation=tf.nn.relu, name='q_dense_base')
                        q_base = tf.layers.dense(q_dense_base, 1, name='q_base')
                        q = q_base + q_act - tf.reduce_mean(q_act, axis=1, keepdims=True)
                    else:
                        q = q_act
                    if flatten:
                        q = tf.reshape(q, [-1, self.args.rrd_sample_size, self.acts_num])
                return q

            value_net = mlp_value if len(self.args.obs_dims)==1 else conv_value

            def mlp_policy(Q_values,axis=-1):
                Q_values_max = np.max(Q_values, axis=axis, keepdims=True)
                Q_values_shifted = Q_values - Q_values_max               
                exp_Q = tf.exp(Q_values_shifted)
                sum_exp_Q = tf.reduce_sum(exp_Q, axis=-1, keepdims=True)

                action_distribution = exp_Q / sum_exp_Q
                
                return action_distribution

            with tf.variable_scope('main'):
                with tf.variable_scope('value'):
                    self.q = value_net(self.rrd_obs_ph)
                    self.q_action=tf.reduce_sum(self.q*self.rrd_acts_ph,axis=-1,keepdims=True)
                    self.q_pi = tf.reduce_max(self.q, axis=-1, keepdims=True)
                with tf.variable_scope('value',reuse=True):
                        self.q2=value_net(self.obs_ph)
                        self.policy=mlp_policy(self.q2)
                        self.policy=tf.reshape(self.policy,[self.acts_num])
                    

            with tf.variable_scope('target'):
                with tf.variable_scope('value'):
                        self.q_t1= value_net(self.rrd_obs_next_ph)
                        self.policy1=tf.stop_gradient(mlp_policy(self.q_t1))
                        self.q_t=self.policy1*(self.q_t1-self.args.alpha*tf.log(self.policy1))
                        self.q_t=tf.reduce_sum(self.q_t,axis=-1,keepdims=True)
         

        def step(self, obs, explore=False, test_info=False):
            if (not test_info) and (self.args.buffer.step_counter<self.args.warmup):
                return np.random.randint(self.acts_num)

            # eps-greedy exploration
            if explore and np.random.uniform()<=self.args.eps_act:
                return np.random.randint(self.acts_num)

            feed_dict = {
                # the same processing as frame_stack_buffer
                self.raw_obs_ph: [obs/255.0]
            }
            policy, info = self.sess.run([self.policy, self.step_info], feed_dict)
            action = np.random.choice(len(policy), p=policy)

            if test_info: return action, info
            return action


        def create_operators(self):

            target = tf.stop_gradient((1.0-self.rrd_done_ph)*self.args.gamma*self.q_t)
            # print(self.q_action.shape)
            # print(target.shape)
            
            self.rrd_rews_pred=self.q_action-target
            # print(self.rrd_rews_pred.shape)
            self.rrd=tf.reduce_mean(self.rrd_rews_pred, axis=1)
            # print(self.rrd.shape)
            self.q_loss = tf.reduce_mean(tf.square(self.rrd_rews_ph-self.rrd))
            # print(self.q_loss.shape)
            if self.args.rrd_bias_correction:

                assert self.args.rrd_sample_size>1
                n = self.args.rrd_sample_size
                self.r_var_single = tf.reduce_sum(tf.square(self.rrd_rews_pred-tf.reduce_mean(self.rrd_rews_pred, axis=1, keepdims=True)), axis=1) / (n-1)
                self.r_var = tf.reduce_mean(self.r_var_single*self.rrd_var_coef_ph/n)
                print(self.r_var.shape)
                self.q_total_loss = self.q_loss - self.r_var
            else:
                self.q_total_loss = self.q_loss

            if self.args.optimizer=='adam':
                self.q_optimizer = tf.train.AdamOptimizer(self.args.q_lr, epsilon=self.args.Adam_eps)
            elif self.args.optimizer=='rmsprop':
                self.q_optimizer = tf.train.RMSPropOptimizer(self.args.q_lr, decay=self.args.RMSProp_decay, epsilon=self.args.RMSProp_eps)
            self.q_train_op = self.q_optimizer.minimize(self.q_total_loss, var_list=get_vars('main/value'))

            self.target_update_op = tf.group([
                # v_t.assign(v)
                v_t.assign(self.args.polyak*v_t + (1.0-self.args.polyak)*v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])

            self.saver=tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
            self.target_init_op = tf.group([
                v_t.assign(v)
                for v, v_t in zip(get_vars('main'), get_vars('target'))
            ])


        def feed_dict(self, batch):

            batch_size = np.array(batch['rrd_obs']).shape[0]
            print(f'batch_size:{batch_size}')
            # basis_feed_dict = super().feed_dict(batch)
            # del basis_feed_dict[self.rews_ph]
            def one_hot(idx):
                idx = np.array(idx)
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx = np.reshape(idx, [batch_size*sample_size])
                res = np.zeros((batch_size*sample_size, self.acts_num), dtype=np.float32)
                res[np.arange(batch_size*sample_size),idx] = 1.0
                res = np.reshape(res, [batch_size, sample_size, self.acts_num])
                return res
            rrd_feed_dict = {
                    self.rrd_raw_obs_ph: batch['rrd_obs'],
                    self.rrd_raw_obs_next_ph: batch['rrd_obs_next'],
                    self.rrd_acts_ph: batch['rrd_acts'] if self.args.env_category!='atari' else one_hot(batch['rrd_acts']),
                    self.rrd_rews_ph: batch['rrd_rews'],
                    self.rrd_done_ph: batch['rrd_done'],
                    self.raw_obs_ph : np.zeros((batch_size, 84, 84, 4))
            }
            
            if self.args.rrd_bias_correction:
                rrd_feed_dict[self.rrd_var_coef_ph] = batch['rrd_var_coef']
            return rrd_feed_dict


    return RandomizedReturnDecomposition(args)
