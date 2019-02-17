import numpy as np
import tensorflow as tf
from replaybuffer import PrioritizedReplayBuffer, ReplayBuffer
from baselines.common.schedules import LinearSchedule


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            Name,
            optimizer = 'rmsprop',
            momentum = None,
            learning_rate=0.01,
            opt_decay=0.99,
            reward_decay=0.99,
            e_greedy=0.9,
	        e_greedy_max=0.99,
            e_greedy_increment=None,
            e_greedy_iter=5e4,                                  		                   
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            output_graph=False,
            prioritized_replay=False,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta0=0.4,
            prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-6,  
            ):
        with tf.variable_scope(Name):
            self.n_actions = n_actions
            self.n_features = n_features
            self.Name=Name
            self.lr = learning_rate
            self.gamma = reward_decay
            self.epsilon_max = e_greedy_max
            self.replace_target_iter = replace_target_iter
            self.memory_size = memory_size
            self.batch_size = batch_size
            self.epsilon_increment = e_greedy_increment
            # self.epsilon = e_greedy if e_greedy_increment is not None else self.epsilon_max
            # new
            self.e_greedy_iter = e_greedy_iter
            self.epsilon = LinearSchedule(self.e_greedy_iter,
                                          initial_p=e_greedy,
                                          final_p=self.epsilon_max)
            self.prioritized_replay = prioritized_replay
            self.prioritized_replay_alpha = prioritized_replay_alpha
            self.prioritized_replay_beta0 = prioritized_replay_beta0
            self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
            self.prioritized_replay_eps = prioritized_replay_eps
            self.optimizer = optimizer
            self.momentum = momentum
            self.opt_decay = opt_decay
            if self.optimizer is 'momentum':
                assert self.momentum is not None
            if self.optimizer is 'rmpsprop':
                assert self.opt_decay is not None
            

            # total learning step
            self.learn_step_counter = 0

            # initialize zero memory [s, a, r, s_]
            # self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
            # new
            self.memory = self._build_replay_buffer()

            # consist of [target_net, evaluate_net]
            self._build_net()

            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            with tf.variable_scope('soft_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.sess = tf.Session()

            if output_graph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter("logs/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
            self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.weights = tf.placeholder(tf.float32, [None, ], name='weights')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(self.Name):
            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                     bias_initializer=b_initializer, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='t2')

            with tf.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
                self.q_target = tf.stop_gradient(q_target)
            with tf.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
            with tf.variable_scope('TD_error'):
                self.td_errors = tf.abs(self.q_target - self.q_eval_wrt_a, name='TD_error')
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(self.weights*tf.squared_difference(self.q_target, self.q_eval_wrt_a))
            with tf.variable_scope('train'):
                if self.optimizer is 'rmsprop':
                    self._train_op = tf.train.RMSPropOptimizer(self.lr, self.opt_decay).minimize(self.loss)
                elif self.optimizer is 'momentum':
                    self._train_op = tf.train.MomentumOptimizer(self.lr,self.momentum).minimize(self.loss)   
                elif self.optimizer is 'adam':
                    self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                else: print('!!!')       

    def _build_replay_buffer(self):
        # Create the replay buffer
        if self.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(self.memory_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                self.prioritized_replay_beta_iters = self.prioritized_replay_iter
            self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                                initial_p=self.prioritized_replay_beta0,
                                                final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(self.memory_size)
            self.beta_schedule = None

        return replay_buffer

    # def store_transition(self, s, a, r, s_):
    #     if not hasattr(self, 'memory_counter'):
    #         self.memory_counter = 0
    #     transition = np.hstack((s, [a, r], s_))
    #     # replace the old memory with new memory
    #     index = self.memory_counter % self.memory_size
    #     self.memory[index, :] = transition
    #     self.memory_counter += 1

    def choose_action(self, observation, test=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon.value(self.learn_step_counter) or test:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
        # print('\ntarget_params_replaced\n')
        if self.learn_step_counter == self.e_greedy_iter:
            print('\n',self.Name, 'achieved e_greedy_max!!!!!!!!!!!!!!!!!!!!!!')

        # sample batch memory from all memory
        # if self.memory_counter > self.memory_size:
        #     sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # else:
        #     sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # batch_memory = self.memory[sample_index, :]
        # new
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        if self.prioritized_replay:
            experience = self.memory.sample(self.batch_size, beta=self.beta_schedule.value(self.learn_step_counter))
            (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
        else:
            obses_t, actions, rewards, obses_tp1, dones = self.memory.sample(self.batch_size)
            weights, batch_idxes = np.ones_like(rewards), None


        # _, cost = self.sess.run(
        #     [self._train_op, self.loss],
        #     feed_dict={
        #         self.s: batch_memory[:, :self.n_features],
        #         self.a: batch_memory[:, self.n_features],
        #         self.r: batch_memory[:, self.n_features + 1],
        #         self.s_: batch_memory[:, -self.n_features:],
        #     })

        _, cost, td_errors = self.sess.run(
            [self._train_op, self.loss, self.td_errors],
            feed_dict={
                self.s: obses_t,
                self.a: actions,
                self.r: rewards,
                self.s_: obses_tp1,
                self.weights: weights,
            })
        
        self.cost_his.append(cost)
     
        # new
        # update memory
        if self.prioritized_replay:
            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
            self.memory.update_priorities(batch_idxes, new_priorities)

    # increasing epsilon
        #self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)
