import numpy as np
import tensorflow as tf
import argparse
import sys
import math
import time
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops



class BasicModel:
    def __init__(self, input_size, dim_output, seq_length, filter_num, meta_lr, update_batch_size,
                 clstm_bidden_num, temporal_type="", period_seq_length=4, period_loss_weight=0.5,
                 l_value=0.5):
        """ must call construct_model() after initializing MAML! """
        self.input_size = input_size
        # self.source_input_height = input_size[2]
        # self.source_input_width = input_size[3]
        self.channels = dim_output
        self.dim_output = dim_output
        self.seq_length = seq_length
        self.period_seq_length = period_seq_length
        self.filter_num = filter_num
        self.clstm_bidden_num = clstm_bidden_num
        self.update_batch_size = update_batch_size
        self.meta_lr = meta_lr
        self.l_value = l_value
        tf.set_random_seed(1234)
        self.inputa = tf.placeholder(tf.float32, [None, seq_length, input_size[0],
                                                  input_size[1], dim_output])
        self.labela = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32, [None, seq_length, input_size[0],
                                                  input_size[1], dim_output])
        self.labelb = tf.placeholder(tf.float32)
        if temporal_type == "period":
            self.p_inputa = tf.placeholder(tf.float32, [None, period_seq_length, input_size[0],
                                                             input_size[1], dim_output])
            self.p_inputb = tf.placeholder(tf.float32, [None, period_seq_length, input_size[0],
                                                             input_size[1], dim_output])
            self.p_seq_length = period_seq_length
        self.temporal_type = temporal_type
        self.period_loss_weight = period_loss_weight

    def construct_convlstm(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3
        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv1'] = tf.Variable(tf.zeros([self.filter_num]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv2'] = tf.Variable(tf.zeros([self.filter_num]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv3'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['d_conv1'] = tf.get_variable('d_conv_1', [k, k, self.channels, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_d_conv1'] = tf.Variable(tf.zeros([self.filter_num]))
        weights['d_conv2'] = tf.get_variable('d_conv_2', [k, k, self.filter_num, self.filter_num],
                                              initializer=conv_initializer, dtype=dtype)
        weights['b_d_conv2'] = tf.Variable(tf.zeros([self.filter_num]))
        weights['d_conv3'] = tf.get_variable('d_conv_3', [k, k, self.filter_num, self.filter_num],
                                              initializer=conv_initializer, dtype=dtype)
        weights['b_d_conv3'] = tf.Variable(tf.zeros([self.filter_num]))
        weights['clstm1'] = tf.get_variable('clstm1', [1, 1, self.filter_num * 2 + self.clstm_bidden_num,
                                                       4 * self.clstm_bidden_num],
                                            initializer=init_ops.truncated_normal_initializer(stddev=0.001),
                                            dtype=dtype)
        weights['b_clstm1'] = tf.Variable(tf.zeros([4 * self.clstm_bidden_num]))
        h_num = 2

        if self.temporal_type in ["period"]:
            h_num = h_num * 2
        # 输出层
        weights['h_conv1'] = tf.get_variable('h_conv1', [1, 1, self.filter_num * h_num, self.channels],
                                             initializer=conv_initializer, dtype=dtype)
        weights['b_h_conv1'] = tf.Variable(tf.zeros([self.channels]))
        return weights

    def p_construct_convlstm(self):
        p_weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        p_weights['period_clstm1'] = tf.get_variable('period_clstm1',
                                                     [1, 1, self.filter_num*2 + self.clstm_bidden_num,
                                                      4 * self.clstm_bidden_num],
                                                     initializer=init_ops.truncated_normal_initializer(
                                                         stddev=0.001),
                                                     dtype=dtype)
        h_num = 2
        p_weights['b_period_clstm1'] = tf.Variable(tf.zeros([4 * self.clstm_bidden_num]))
        p_weights["period_conv1"] = tf.get_variable("period_conv1",
                                                    [1, 1, self.channels * self.seq_length, self.filter_num * h_num],
                                                    initializer=conv_initializer, dtype=dtype)
        p_weights['b_period_conv1'] = tf.Variable(tf.zeros([self.filter_num * h_num]))
        p_weights["period_conv2"] = tf.get_variable("period_conv2",
                                                    [1, 1, self.filter_num * h_num, self.filter_num * h_num],
                                                    initializer=conv_initializer, dtype=dtype)
        p_weights['b_period_conv2'] = tf.Variable(tf.zeros([self.filter_num * h_num]))

        return p_weights

    def conv_block(self, cinp, cweight, bweight, activation):
        """ Perform, conv, batch norm, nonlinearity, and max pool """
        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
        conv_output = tf.nn.conv2d(cinp, cweight, no_stride, 'SAME') + bweight
        return activation(conv_output)

    def dilated_conv_block(self, cinp, cweight, bweight, rate, activation):
        """ Perform, conv, batch norm, nonlinearity, and max pool """
        conv_output = tf.nn.atrous_conv2d(cinp, cweight, rate, 'SAME') + bweight
        return activation(conv_output)

    def conv_lstm(self, inp, weights_c, weights_b, h_size, w_size, return_sequence=True):
        def convlstm_block(linp, pre_state, kweight, bweight, activation):
            sigmoid = math_ops.sigmoid

            c, h = array_ops.split(axis=3, num_or_size_splits=2, value=pre_state)
            res = tf.nn.conv2d(array_ops.concat([linp, h], axis=3), kweight, [1, 1, 1, 1], padding='SAME')
            concat = res + bweight
            i, j, f, o = array_ops.split(axis=3, num_or_size_splits=4, value=concat)
            forget_bias_tensor = constant_op.constant(1.0, dtype=f.dtype)
            new_c = (c * sigmoid(f + forget_bias_tensor) + sigmoid(i) *
                     activation(j))

            new_h = activation(new_c) * sigmoid(o)
            new_state = array_ops.concat([new_c, new_h], axis=3)
            return new_h, new_state
        height = h_size
        width = w_size
        # 矩阵分解shape=(8, ?, height, width, channel)
        inp = tf.unstack(tf.transpose(inp, perm=[1, 0, 2, 3, 4]))
        state = tf.zeros([self.update_batch_size, height, width, self.clstm_bidden_num*2])
        output = None
        output_list = []
        for t in range(len(inp)):
            output, state = convlstm_block(inp[t], state, weights_c, weights_b, tf.nn.tanh)
            output_list.append(output)
        if return_sequence == True:
            output_list = tf.transpose(tf.stack(output_list, axis=0), perm=[1, 0, 2, 3, 4])
            return output_list
        else:
            return output

    def spatial_network(self, inp, weights):
        inp = tf.unstack(tf.transpose(inp, perm=[1, 0, 2, 3, 4]))
        conv_list = []
        for i in range(len(inp)):
            conv_inp = inp[i]
            conv_1 = self.conv_block(conv_inp, weights['conv1'], weights['b_conv1'], tf.nn.relu)
            conv_2 = self.conv_block(conv_1, weights['conv2'], weights['b_conv2'], tf.nn.relu)
            conv_3 = self.conv_block(conv_2, weights['conv3'], weights['b_conv3'], tf.nn.relu)
            conv_list.append(conv_3)
        conv_near = tf.transpose(tf.stack(conv_list), perm=[1, 0, 2, 3, 4])


        conv_list_distant = []
        for i in range(len(inp)):
            conv_inp = inp[i]
            d_conv_1 = self.dilated_conv_block(conv_inp, weights['d_conv1'], weights['b_d_conv1'], 2, tf.nn.relu)
            d_conv_2 = self.dilated_conv_block(d_conv_1, weights['d_conv2'], weights['b_d_conv2'], 3, tf.nn.relu)
            d_conv = self.dilated_conv_block(d_conv_2, weights['d_conv3'], weights['b_d_conv3'], 5, tf.nn.relu)
            conv_list_distant.append(d_conv)
        conv_distant = tf.transpose(tf.stack(conv_list_distant), perm=[1, 0, 2, 3, 4])
        conv_all = tf.concat([conv_near, conv_distant], axis=-1)
        return conv_all


    def forward_convlstm(self, inp, weights, input_list):
        h_size, w_size = input_list[0], input_list[1]
        conv_all = self.spatial_network(inp, weights)
        conv_lstm = self.conv_lstm(conv_all, weights['clstm1'], weights['b_clstm1'], h_size, w_size,
                                   return_sequence=False)
        output = self.conv_block(conv_lstm, weights['h_conv1'], weights['b_h_conv1'], tf.nn.relu)

        output = tf.nn.tanh(output)
        return output

    def period_convlstm(self, inp, period_inp, weights, input_list, p_weights):
        h_size, w_size = input_list[0], input_list[1]
        conv_all = self.spatial_network(inp, weights)
        conv_lstm = self.conv_lstm(conv_all, weights['clstm1'], weights['b_clstm1'], h_size, w_size,
                                   return_sequence=False)
        period_conv_all = self.spatial_network(period_inp, weights)
        period_conv_lstm = self.conv_lstm(period_conv_all, p_weights['period_clstm1'],
                                          p_weights['b_period_clstm1'], h_size, w_size, return_sequence=False)
        n_conv_lstm = tf.concat([conv_lstm, period_conv_lstm], axis=-1)
        output = self.conv_block(n_conv_lstm, weights['h_conv1'], weights['b_h_conv1'], tf.nn.relu)
        inp_trans = tf.transpose(inp, perm=[0, 2, 3, 1, 4])
        inp_reshape = tf.reshape(inp_trans, [-1, input_list[0], input_list[1], self.seq_length*self.dim_output])
        inp_h = self.conv_block(inp_reshape, p_weights['period_conv1'], p_weights['b_period_conv1'], tf.nn.relu)
        period_rep = self.conv_block(inp_h, p_weights['period_conv2'], p_weights['b_period_conv2'], tf.nn.relu)
        p_conv_lstm = tf.concat([conv_lstm, period_rep], axis=-1)
        p_output = self.conv_block(p_conv_lstm, weights['h_conv1'], weights['b_h_conv1'], tf.nn.relu)
        output = tf.nn.tanh(output)
        p_output = tf.nn.tanh(p_output)
        return output, period_rep, period_conv_lstm, p_output

    def t_period_convlstm(self, inp, weights, input_list, p_weights):
        h_size, w_size = input_list[0], input_list[1]
        conv_all = self.spatial_network(inp, weights)
        conv_lstm = self.conv_lstm(conv_all, weights['clstm1'], weights['b_clstm1'], h_size, w_size,
                                   return_sequence=False)
        inp_trans = tf.transpose(inp, perm=[0, 2, 3, 1, 4])
        inp_reshape = tf.reshape(inp_trans, [-1, input_list[0], input_list[1], self.seq_length*self.dim_output])
        inp_h = self.conv_block(inp_reshape, p_weights['period_conv1'], p_weights['b_period_conv1'], tf.nn.relu)
        period_rep = self.conv_block(inp_h, p_weights['period_conv2'], p_weights['b_period_conv2'], tf.nn.relu)
        p_conv_lstm = tf.concat([conv_lstm, period_rep], axis=-1)
        p_output = self.conv_block(p_conv_lstm, weights['h_conv1'], weights['b_h_conv1'], tf.nn.relu)
        # print(output.shape)
        p_output = tf.nn.tanh(p_output)
        return p_output

    def combine_loss(self, loss, ploss):
        return (1-self.period_loss_weight) * loss + self.period_loss_weight * ploss

    def combine_loss_2(self, loss1, loss2, ploss):
        l_value = self.l_value
        return (1-self.period_loss_weight) * l_value * loss1 + \
               (1-self.period_loss_weight) * (1-l_value) * loss2 + self.period_loss_weight * ploss

    def loss_func(self, pred, label):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.reduce_mean(tf.square(pred - label))

    def loss_func_sum(self, pred, label):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.divide(tf.reduce_sum(tf.square(pred - label)), self.update_batch_size)

    def np_loss_func(self, pred, label):
        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        return np.mean(np.square(pred - label))

    def construct_model(self):
        tf.set_random_seed(1234)
        with tf.variable_scope('model', reuse=None):
            input_list = [self.input_size[0], self.input_size[1]]

            if self.temporal_type == "period":
                with tf.variable_scope('recent', reuse=None):
                    self.weights = weights = self.construct_convlstm()
                with tf.variable_scope('period', reuse=None):
                    self.p_weights = p_weights = self.p_construct_convlstm()
                outputa, period_repa, period_conv_lstma, p_outputa = \
                    self.period_convlstm(self.inputa, self.p_inputa, weights, input_list, p_weights)
                outputb, period_repb, period_conv_lstmb, p_outputb = \
                    self.period_convlstm(self.inputb, self.p_inputb, weights, input_list, p_weights)
            else:
                self.weights = weights = self.construct_convlstm()
                outputa = \
                    self.forward_convlstm(self.inputa, weights, input_list)
                outputb = \
                    self.forward_convlstm(self.inputb, weights, input_list)

        self.outputas, self.outputbs = outputa, outputb
        self.total_loss1 = self.loss_func(outputa, self.labela)
        self.total_loss2 = self.loss_func(outputb, self.labelb)
        self.total_rmse1 = tf.sqrt(self.total_loss1)
        self.total_rmse2 = tf.sqrt(self.total_loss2)
        # Performance & Optimization
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.total_loss1)
        if self.temporal_type == "period":
            self.p_outputas, self.p_outputbs = p_outputa, p_outputb
            self.p_total_loss1 = self.loss_func(p_outputa, self.labela)
            self.p_total_loss2 = self.loss_func(p_outputb, self.labelb)
            self.p_total_rmse1 = tf.sqrt(self.p_total_loss1)
            self.p_total_rmse2 = tf.sqrt(self.p_total_loss2)
            self.period_loss1 = self.loss_func(period_repa, period_conv_lstma)
            self.period_loss2 = self.loss_func(period_repb, period_conv_lstmb)
            self.period_rmse1 = tf.sqrt(self.period_loss1)
            self.period_rmse2 = tf.sqrt(self.period_loss2)
            self.pretrain_period_op = tf.train.AdamOptimizer(self.meta_lr).minimize(
                self.combine_loss(self.total_loss1, self.period_loss1))
            recent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/recent")
            self.p_train_op = tf.train.AdamOptimizer(self.meta_lr).minimize(self.p_total_loss1, var_list=recent_vars)

class Att_ReptileModel(BasicModel):
    def __init__(self, input_size, t_input_size, dim_output, seq_length, filter_num, meta_lr, update_lr, session,
                 meta_batch_size, test_num_updates, update_batch_size, clstm_bidden_num, l_value=0.5,
                 temporal_type='', period_seq_length=4, period_loss_weight=0.5):
        print("Initializing attention ReptileModel...")
        BasicModel.__init__(self, input_size, dim_output, seq_length, filter_num,
                            meta_lr, update_batch_size, clstm_bidden_num, l_value=l_value,
                            temporal_type=temporal_type, period_seq_length=period_seq_length,
                            period_loss_weight=period_loss_weight)
        tf.set_random_seed(1234)
        self.update_lr = update_lr
        self.test_num_updates = test_num_updates
        self.meta_batch_size = meta_batch_size
        self.session = session
        self.t_input_size = t_input_size
        self.input1_a = (tf.placeholder(tf.float32, [None, self.seq_length, input_size[0], input_size[1], self.dim_output]),
                         tf.placeholder(tf.float32, [None, self.seq_length, input_size[2], input_size[3], self.dim_output]))
        self.t_input = (tf.placeholder(tf.float32, [None, self.seq_length, t_input_size[0], t_input_size[1], self.dim_output]))
        self.label1_a = (tf.placeholder(tf.float32, [None, input_size[0], input_size[1], self.dim_output]),
                         tf.placeholder(tf.float32, [None, input_size[2], input_size[3], self.dim_output]))
        self.t_label = (tf.placeholder(tf.float32, [None, t_input_size[0], t_input_size[1], self.dim_output]))
        if temporal_type == "period":
            self.p_input1_a = (
                tf.placeholder(tf.float32, [None, period_seq_length, input_size[0], input_size[1], self.dim_output]),
                tf.placeholder(tf.float32, [None, period_seq_length, input_size[2], input_size[3], self.dim_output]))
            self.period_seq_length = period_seq_length

    def interpolate_vars(self, old_vars, new_vars, epsilon):
        return self.add_vars(old_vars, self.scale_vars(self.subtract_vars(new_vars, old_vars), epsilon))

    def average_vars(self, var_seqs):
        res = []
        for variables in zip(*var_seqs):
            res.append(np.mean(variables, axis=0))
        return res

    def att_vars(self, var_seqs, att_val):
        res = []
        for variables in zip(*var_seqs):
            res.append(att_val[0]*variables[0]+att_val[1]*variables[1])
        return res

    def subtract_vars(self, var_seq_1, var_seq_2):
        return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

    def add_vars(self, var_seq_1, var_seq_2):
        return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

    def scale_vars(self, var_seq, scale):
        return [v * scale for v in var_seq]

    def task_reptile(self, inputa, labela, input_list):
        weights = self.weights
        task_outputa = self.forward_convlstm(inputa, weights, input_list)
        task_lossa = self.loss_func(task_outputa, labela)
        return task_outputa, task_lossa

    def period_task_reptile(self, inputa, p_inputa, labela, input_list):
        weights = self.weights
        p_weights = self.p_weights
        outputa, period_repa, period_conv_lstma, p_outputa = \
            self.period_convlstm(inputa, p_inputa, weights, input_list, p_weights)
        task_lossa = self.loss_func(outputa, labela)
        return outputa, period_repa, period_conv_lstma, p_outputa, task_lossa

    def t_period_reptile(self, inputa, labela, input_list):
        weights = self.weights
        p_weights = self.p_weights
        p_outputa = self.t_period_convlstm(inputa, weights, input_list, p_weights)
        task_lossa = self.loss_func(p_outputa, labela)
        return p_outputa, task_lossa

    def model_state(self):
        self._model_state = VariableState(self.session, tf.trainable_variables())

    def target_loss(self, train_inputs, train_labels):
        train_batch_num = math.ceil(train_inputs.shape[0] / self.update_batch_size)
        train_data_size = train_inputs.shape[0]
        for i in range(train_batch_num):
            end_index = (i + 1) * self.update_batch_size
            if end_index <= train_inputs.shape[0]:
                t_inputa = train_inputs[i * self.update_batch_size: (i + 1) * self.update_batch_size, :, :, :, :]
                t_labela = train_labels[i * self.update_batch_size: (i + 1) * self.update_batch_size, :, :, :]
            else:
                t_inputa = train_inputs[i * self.update_batch_size: train_data_size, :, :, :, :]
                t_labela = train_labels[i * self.update_batch_size: train_data_size, :, :, :]
                t_diff = end_index - train_data_size
                t_diff_last = self.update_batch_size - t_diff
                t_add_dat = train_inputs[0: t_diff, :, :, :, :]
                t_add_label = train_labels[0: t_diff, :, :, :]
                t_inputa = np.concatenate([t_inputa, t_add_dat], axis=0)
                t_labela = np.concatenate([t_labela, t_add_label], axis=0)
            feed_dict = {self.t_input: t_inputa, self.t_label: t_labela
                         }
            t_outputa = self.session.run(self.t_task_outputa, feed_dict)
            if i == 0:
                t_total_outputa = t_outputa
            elif end_index <= train_data_size:
                t_total_outputa = np.concatenate([t_total_outputa, t_outputa], axis=0)
            else:
                t_total_outputa = np.concatenate([t_total_outputa, t_outputa[:t_diff_last]], axis=0)
        t_loss = self.np_loss_func(t_total_outputa, train_labels)
        return t_loss

    def train_step(self, data_generator, t_data_generator):
        num_updates = self.test_num_updates
        temporal_type = self.temporal_type
        old_vars = self._model_state.export_variables()
        new_vars = []
        loss_list = []
        t_train_inputs_list, t_train_labels_list = t_data_generator.get_all_data(purpose='train')
        t_train_inputs = t_train_inputs_list[0][0]
        t_train_labels = t_train_labels_list[0][0]
        for city_num in range(2):
            for _ in range(num_updates):
                input_list = {}
                if temporal_type == "period":
                    batch_x, p_batch_x, batch_y = data_generator.generate(purpose='train',
                                                                          update_batch_size=self.update_batch_size)
                    input_list['inputa'] = batch_x
                    input_list['p_inputa'] = p_batch_x
                    input_list['labela'] = batch_y
                    feed_dict = {self.input1_a: input_list['inputa'], self.p_input1_a: input_list['p_inputa'],
                                 self.label1_a: input_list['labela']}
                    self.session.run([self.pretrain_period_op_list[city_num]], feed_dict)
                else:
                    batch_x, batch_y = data_generator.generate(purpose='train',
                                                               update_batch_size=self.update_batch_size)
                    input_list['inputa'] = batch_x
                    input_list['labela'] = batch_y
                    feed_dict = {self.input1_a: input_list['inputa'], self.label1_a: input_list['labela']}
                    self.session.run([self.pretrain_op_list[city_num]], feed_dict)
            losses_target = self.target_loss(t_train_inputs, t_train_labels)
            loss_list.append(np.sqrt(losses_target))
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        initial_losses_target = np.sqrt(self.target_loss(t_train_inputs, t_train_labels))
        d_losses_list = [(initial_losses_target-loss_list[i])/(self.update_lr*5) for i in range(len(loss_list))]
        att_val = np.exp(d_losses_list) / np.sum(np.exp(d_losses_list), axis=-1)
        new_vars = self.att_vars(new_vars, att_val)
        self._model_state.import_variables(self.interpolate_vars(old_vars, new_vars, self.meta_lr))

    def construct_model(self):
        tf.set_random_seed(1234)
        with tf.variable_scope('model', reuse=None):
            if self.temporal_type == "period":
                with tf.variable_scope('recent', reuse=None):
                    self.weights = self.construct_convlstm()
                with tf.variable_scope('period', reuse=None):
                    self.p_weights = self.p_construct_convlstm()
                outputas, lossesa, p_outputas, task_period_repas, task_period_conv_lstmas = [], [], [], [], []
                for i in range(self.meta_batch_size):
                    input_list = [self.input_size[i * 2], self.input_size[i * 2 + 1]]
                    task_outputa, task_period_repa, task_period_conv_lstma, task_p_outputa, task_lossa = \
                        self.period_task_reptile(self.input1_a[i], self.p_input1_a[i], self.label1_a[i], input_list)
                    outputas.append(task_outputa)
                    lossesa.append(task_lossa)
                    p_outputas.append(task_p_outputa)
                    task_period_repas.append(task_period_repa)
                    task_period_conv_lstmas.append(task_period_conv_lstma)
                t_input_list = [self.t_input_size[0], self.t_input_size[1]]
                t_task_outputa, t_task_lossa = \
                    self.t_period_reptile(self.t_input, self.t_label, t_input_list)
            else:
                self.weights = self.construct_convlstm()
                outputas, lossesa = [], []
                for i in range(self.meta_batch_size):
                    input_list = [self.input_size[i * 2], self.input_size[i * 2 + 1]]
                    task_outputa, task_lossa = \
                        self.task_reptile(self.input1_a[i], self.label1_a[i], input_list)
                    outputas.append(task_outputa)
                    lossesa.append(task_lossa)
                t_input_list = [self.t_input_size[0], self.t_input_size[1]]
                t_task_outputa, t_task_lossa = \
                    self.task_reptile(self.t_input, self.t_label, t_input_list)
        self.t_task_outputa = t_task_outputa
        self.pretrain_op_list = [tf.train.AdamOptimizer(self.update_lr).minimize(lossesa[i])
                                 for i in range(len(lossesa))]
        self.total_rmse_list = [tf.sqrt(lossesa[i]) for i in range(len(lossesa))]
        if self.temporal_type == "period":
            self.p_outputas = p_outputas
            self.p_total_loss_list = [self.loss_func(p_outputas[i], self.label1_a[i]) for i in range(len(p_outputas))]
            self.p_total_loss_rmse_list = [tf.sqrt(self.p_total_loss_list[i]) for i in range(len(p_outputas))]
            self.period_loss_list = [self.loss_func(task_period_repas[i], task_period_conv_lstmas[i])
                                     for i in range(len(p_outputas))]
            self.period_rmse_list = [tf.sqrt(self.period_loss_list[i]) for i in range(len(p_outputas))]
            self.pretrain_period_op_list = [tf.train.AdamOptimizer(self.update_lr).minimize(
                self.combine_loss_2(lossesa[i], self.p_total_loss_list[i], self.period_loss_list[i])) for i in
                range(len(p_outputas))]

class VariableState:
    def __init__(self, session, variables):
        self._session = session
        self._variables = variables
        self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in variables]
        assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

    def export_variables(self):
        return self._session.run(self._variables)

    def import_variables(self, values):
        self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))
