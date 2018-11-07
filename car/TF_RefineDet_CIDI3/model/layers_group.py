#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'xie wei'

import tensorflow as tf
from model.candidate_box_process import *
import numpy as np
import os, sys

def switch_norm(x, scope='switch_norm'):
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x

def _norm(x, norm_type, is_train, G=32, esp=1e-5):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.layers.batch_normalization(conv, training=is_train)

        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [N, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [N, C, H, W]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output

def get_conv_filter(shape,name):
    init = tf.truncated_normal(shape, stddev=0.01)
    var = tf.get_variable(name="filter", initializer=init, shape=shape)
    return var

def get_bias(shape,name):
    bias_wights = tf.constant(0.0, shape)
    init = tf.constant_initializer(value=bias_wights,dtype=tf.float32)
    var = tf.get_variable(name="biases", initializer=init, shape=shape)
    return var

def add_relu(input1,input2,name='add_relu'):
    with tf.variable_scope(name):
        add = tf.add(input1,input2,name=name)
        return tf.nn.relu(add)

def conv(inputs, filters, kernel_size=1, strides=1, pad='SAME',is_training = False,w_summary=True, bias = True,name='conv'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
        tf.contrib.layers.xavier_initializer(uniform=False), trainable=is_training)

        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        if bias:
            bia = tf.Variable(tf.constant(0.0, shape=[filters]),name='biases')
            return tf.nn.bias_add(conv, bia)
        else:
            return conv

def conv_(inputs, filters, kernel_size=[9,1], strides=1, pad='SAME',w_summary=True, name='conv'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        return conv
def deconv2d(inputs, filters, kernel_size=1, strides=1, pad='SAME',is_training = False,w_summary=True, bias = True,name='deconv'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, filters, inputs.get_shape().as_list()[3]]),initializer=
        tf.contrib.layers.xavier_initializer(uniform=False), trainable=is_training)
        # x_shape = tf.shape(inputs)
        # output_shape = tf.stack([x_shape[0], x_shape[1]*strides, x_shape[2]*strides, filters])
        x_shape = inputs.shape.as_list()
        output_shape = [x_shape[0],x_shape[1]*strides, x_shape[2]*strides, filters]
        deconv = tf.nn.conv2d_transpose(inputs, kernel, output_shape, strides=[1, strides, strides, 1], padding=pad,
                               data_format='NHWC')
        if bias:
            bia = tf.Variable(tf.constant(0.0, shape=[filters]),name='biases')
            return tf.nn.bias_add(deconv, bia)
        else:
            return deconv
def conv_bn_relu(inputs, filters, kernel_size=1, strides=1, pad='SAME',is_training=False,w_summary=True, name='conv_bn_relu'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad,name=name)
    if (name[:3] == 'res'):
        with tf.variable_scope('bn'+name[3:]):
            # norm = _norm(conv, 'group', is_training)
            norm = tf.layers.batch_normalization(conv,training=is_training)
    else:
        with tf.variable_scope('bn_' + name):
            # norm = _norm(conv, 'group', is_training)
            norm = tf.layers.batch_normalization(conv,training=is_training)
    # if w_summary:
        with tf.device('/cpu:0'):
            tf.summary.histogram('weights_summary', kernel, collections=['weights'])
    return tf.nn.relu(norm)

def conv_bn(inputs, filters, kernel_size=1, strides=1, pad='SAME', is_training=False, w_summary=True,name='conv_bn'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, name=name)
    with tf.variable_scope('bn'+name[3:]):
        # norm = _norm(conv, 'group', is_training)
        norm = tf.layers.batch_normalization(conv,training=is_training)
    if w_summary:
        with tf.device('/cpu:0'):
            tf.summary.histogram('weights_summary', kernel, collections=['weights'])
    return norm


def conv_relu(inputs, filters, kernel_size=1, strides=1, pad='SAME',is_training=False,w_summary=True, bias=True,name='conv_bn_relu'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        if bias:
            bia = tf.get_variable('biases',([filters]),initializer=tf.constant_initializer(0.0))
            return tf.nn.relu(tf.nn.bias_add(conv, bia))
        else:
            return tf.nn.relu(conv)

def astrous_conv(inputs,filters,kernel_size,rate=1,pad='SAME',is_training = False,bias = True,w_summary=True,name='astrous_conv'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate,padding=pad)
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        if bias:
            bia = tf.get_variable('biases',initializer = tf.zeros_initializer(filters),shape=[filters])
            return tf.nn.bias_add(conv, bia)
        else:
            return conv


def astrous_conv_relu(inputs,filters,kernel_size,rate=1,pad='SAME',w_summary=True,is_training = False,bias = True,name='astrous_conv'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate,padding=pad)
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        if bias:
            bia = tf.get_variable('biases',initializer=tf.constant_initializer(0.0),shape=[filters])
            return tf.nn.relu(tf.nn.bias_add(conv, bia))
        else:
            return tf.nn.relu(conv)

def astrous_conv_bn_relu(inputs,filters,kernel_size,rate=1,pad='SAME',w_summary=True,is_training = False,bias = False,name='astrous_conv_bn'):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate,padding=pad)
    with tf.variable_scope('bn'+name[3:]):
        # norm = _norm(conv, 'group', is_training)
        norm = tf.layers.batch_normalization(conv,training=is_training)
    if w_summary:
        with tf.device('/cpu:0'):
            tf.summary.histogram('weights_summary', kernel, collections=['weights'])
    if bias:
        bia = tf.Variable(tf.constant(0.0, shape=[filters]),name='biases')
        return tf.nn.relu(tf.nn.bias_add(norm, bia))
    else:
        return tf.nn.relu(norm)

def separable_conv_bn_relu(inputs,filters,kernel_size,strides=1,pad='same',w_summary=True,is_training = False,bias = False,rate=None,name='separable_conv_bn'):
    with tf.variable_scope(name):
        if not rate is None:
            rates=(rate,rate)
        else:
            rates=(1,1)
        conv = tf.layers.separable_conv2d(inputs, filters, (kernel_size, kernel_size),
                                            strides=(strides, strides), padding=pad,
                                            activation=None, use_bias=bias,
                                            depthwise_initializer=tf.truncated_normal_initializer(stddev=0.33),
                                            pointwise_initializer=tf.truncated_normal_initializer(stddev=0.06),
                                            bias_initializer=tf.zeros_initializer(),
                                            dilation_rate=rates,
                                            name=name, reuse=None)
    with tf.variable_scope('bn'+name[3:]):
        # norm = _norm(conv, 'group', is_training)
        norm = tf.layers.batch_normalization(conv,training=is_training)
    return tf.nn.relu(norm)

def pool(inputs,kernel_size,stride,pooltpype='max',padding='SAME',name='pool'):
    with tf.variable_scope(name):
        if pooltpype=='max':
            return tf.contrib.layers.max_pool2d(inputs,[kernel_size,kernel_size],[stride,stride],padding=padding)
        else:
            return tf.contrib.layers.avg_pool2d(inputs,[kernel_size,kernel_size],[stride,stride],padding=padding)

def Unsampling_bilinear(inputs,filters,kernel_size,outshape=[None,800,288,5],stride=1,padding='SAME',is_training =False,name='pool'):
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights',([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),initializer=
                                 tf.contrib.layers.xavier_initializer(uniform=False),trainable=is_training)
        conv = tf.nn.conv2d_transpose(inputs, kernel, outshape, strides=[1, stride, stride, 1], padding=padding)
        return conv

def spatial_conv(inputs,filters,kernel_size,strides,padding,spatial='up',w_summary=True,is_training= False,name='spatial_conv'):
    # feature,NHWC
    # filters,W H C_in C_out
    N, H, W, C = inputs.get_shape().as_list()
    R=0
    dim = 0
    if spatial=='up':
        bottom_slice = tf.split(inputs, H, 1)
        R = H
        dim = 1
    elif spatial=='left':
        bottom_slice = tf.split(inputs, W, 2)
        R = W
        dim = 2
    for i in range(R-1):
        with tf.variable_scope(name+"_D"+str(i)):
            kernel = tf.get_variable('weights', ([kernel_size[0], kernel_size[1], C, filters]), initializer=
            tf.truncated_normal_initializer(mean=0.0, stddev=0.01863, seed=None, dtype=tf.float32),trainable=is_training)
        if i==0:
            conv = tf.nn.conv2d(bottom_slice[i], kernel,[1, strides, strides, 1], padding=padding,data_format='NHWC')
            sum = tf.add(tf.nn.relu(conv),bottom_slice[i+1])
            out = bottom_slice[i]
            out = tf.concat([out,sum],dim)
        else:
            conv = tf.nn.conv2d(sum, kernel, [1, strides, strides, 1],padding=padding,data_format='NHWC')
            sum = tf.add(tf.nn.relu(conv),bottom_slice[i+1])
            out = tf.concat([out, sum], dim)

    up_slice = tf.split(out,R,dim)
    for j in reversed(range(1,R,1)):
        with tf.variable_scope(name+str(j)):
            kernel = tf.get_variable('weights', ([kernel_size[0], kernel_size[1], C, filters]), initializer=
            tf.truncated_normal_initializer(mean=0.0, stddev=0.01863, seed=None, dtype=tf.float32),trainable=is_training)
        if j==R-1:
            conv = tf.nn.conv2d(up_slice[j], kernel, [1, strides, strides, 1], padding=padding,data_format='NHWC')
            sum = tf.add(tf.nn.relu(conv), up_slice[j - 1])
            out = up_slice[j - 1]
            out = tf.concat([out, sum], dim)
        else:
            conv = tf.nn.conv2d(sum, kernel, [1, strides, strides, 1],padding=padding,data_format='NHWC')
            sum = tf.add(tf.nn.relu(conv),up_slice[j - 1])
            out = tf.concat([out,sum],dim)

    if w_summary:
        with tf.device('/cpu:0'):
            tf.summary.histogram('weights_summary', kernel, collections=['weights'])

    return out


def fc(inputs,filters,bias=True,name='fc'):
    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()
        if len(shape)>2:
            inputs = tf.reshape(inputs, [-1, shape[1] * shape[2] * shape[3]])
            weights = tf.get_variable('weights',shape=[shape[1] * shape[2] * shape[3], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        else:
            weights = tf.get_variable('weights',shape=[shape[1], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if bias:
            biases = tf.get_variable('biases',shape=[filters], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.matmul(inputs, weights),biases)
        else:
            return tf.matmul(inputs, weights)

def fc_relu(inputs,filters,bias=True,name='fc_relu'):
    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()
        if len(shape)>2:
            inputs = tf.reshape(inputs, [-1, shape[1] * shape[2] * shape[3]])
            weights = tf.get_variable(shape=[shape[1] * shape[2] * shape[3], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), name=name+'_weights')
        else:
            weights = tf.get_variable(shape=[shape[1], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), name=name+'_weights')

        biases = tf.get_variable('biases',([filters]),initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights),biases))

def fc_sigmoid(inputs,filters,bias=True,name='fc_sigmoid'):
    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            inputs = tf.reshape(inputs, [-1, shape[1] * shape[2] * shape[3]])
            weights = tf.get_variable(shape=[shape[1] * shape[2] * shape[3], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), name=name+'_weights')
        else:
            weights = tf.get_variable(shape=[shape[1], filters], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), name=name+'_weights')

        biases = tf.get_variable(shape=[filters], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0), name=name+'_biases')
        return tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(inputs, weights),biases))

def l2norm(x, scale, trainable=True, scope="L2Normalization"):
    n_channels = x.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(x, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
        return l2_norm * gamma





