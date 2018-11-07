#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'xie wei'

import tensorflow as tf
import numpy as np
import os, sys
import math

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


def conv(inputs, filters, kernel_size=[1,1], strides=[1,1], pad='SAME',is_training=False,bias=False,name='conv',**args_dict):
    with tf.variable_scope(name):
        if args_dict['init']=='gauss':
            kernel = tf.get_variable('weights',([kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[3], filters]),
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                              dtype=tf.float32), trainable=is_training)
        else:
            kernel = tf.get_variable('weights', ([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=is_training)
        conv = tf.nn.conv2d(inputs, kernel, [1, strides[0], strides[1], 1], padding=pad, data_format='NHWC')
        if bias:
            bia = tf.Variable(tf.constant(0.0, shape=[filters]),name='biases')
            conv = tf.nn.bias_add(conv, bia)
        if args_dict['norm']=='bn':
            conv = tf.layers.batch_normalization(conv, training=is_training)
        elif args_dict['norm']=='gn':
            conv = _norm(conv, 'group', is_training)
        elif args_dict['norm']=='sn':
            conv = switch_norm(conv)
        if args_dict['activity']=='relu':
            conv = tf.nn.relu(conv)
        elif args_dict['activity']=='leaky':
            conv = tf.nn.leaky_relu(conv)
        if args_dict['w_summary']:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        else:
            return conv


def astrous_conv(inputs,filters,kernel_size=[1,1],rate=1,pad='SAME',is_training = False,bias = False,name='astrous_conv',**args_dict):
    with tf.variable_scope(name):
        if args_dict['init'] == 'gauss':
            kernel = tf.get_variable('weights',
                                     ([kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[3], filters]),
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                                 dtype=tf.float32),
                                     trainable=is_training)
        else:
            kernel = tf.get_variable('weights', ([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     trainable=is_training)
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate, padding=pad)
        if bias:
            bia = tf.Variable(tf.constant(0.0, shape=[filters]), name='biases')
            conv = tf.nn.bias_add(conv, bia)
        if args_dict['norm'] == 'bn':
            conv = tf.layers.batch_normalization(conv, training=is_training)
        elif args_dict['norm'] == 'gn':
            conv = _norm(conv, 'group', is_training)
        elif args_dict['norm'] == 'sn':
            conv = switch_norm(conv)
        if args_dict['activity'] == 'relu':
            conv = tf.nn.relu(conv)
        elif args_dict['activity'] == 'leaky':
            conv = tf.nn.leaky_relu(conv)
        if args_dict['w_summary']:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weights'])
        else:
            return conv

def deconv(inputs, filters,kernel_size=[1,1],strides=[1,1], pad='SAME',is_training=False,bias = False, name='deconv_bn',**args_dict):
    with tf.variable_scope(name):
        with tf.variable_scope(name):
            if args_dict['init'] == 'gauss':
                kernel = tf.get_variable('weights',
                                         ([kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[3], filters]),
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                                                                     dtype=tf.float32),
                                         trainable=is_training)
            else:
                kernel = tf.get_variable('weights',
                                         ([kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]),
                                         initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                         trainable=is_training)
            conv = tf.nn.conv2d_transpose(inputs, kernel,
                                            [inputs.get_shape().as_list()[0], 2 * inputs.get_shape().as_list()[1],
                                             2 * inputs.get_shape().as_list()[2], filters],
                                            [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if bias:
                bia = tf.Variable(tf.constant(0.0, shape=[filters]), name='biases')
                conv = tf.nn.bias_add(conv, bia)
            if args_dict['norm'] == 'bn':
                conv = tf.layers.batch_normalization(conv, training=is_training)
            elif args_dict['norm'] == 'gn':
                conv = _norm(conv, 'group', is_training)
            elif args_dict['norm'] == 'sn':
                conv = switch_norm(conv)
            if args_dict['activity'] == 'relu':
                conv = tf.nn.relu(conv)
            elif args_dict['activity'] == 'leaky':
                conv = tf.nn.leaky_relu(conv)
            if args_dict['w_summary']:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections=['weights'])
            else:
                return conv


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

        biases = tf.get_variable('biases',shape=[filters], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(inputs, weights),biases)

def prior_box(inputs,imgShape,min_sizes,aspect_ratios,flip,clip,variance,offset,step,name):
    with tf.variable_scope(name):
        anchors = []
        num_priors = len(aspect_ratios) * len(min_sizes)
        layer_width = inputs.get_shape().as_list()[2]
        layer_height = inputs.get_shape().as_list()[1]
        img_width = imgShape[0]
        img_height = imgShape[1]
        step_w = float(img_width)/layer_width
        step_h = float(img_height)/layer_height
        dim = layer_height*layer_width*num_priors*4

        for h in range(layer_height):
            for w in range(layer_width):
                center_x = (w + offset) * step_w
                center_y = (h + offset) * step_h
                for s,min_size in min_sizes:
                    box_width = box_height = min_size
                    anchors.append((center_x - box_width / 2.) / img_width)
                    anchors.append((center_y - box_height / 2.) / img_height)
                    anchors.append((center_x + box_width / 2.) / img_width)
                    anchors.append((center_y + box_height / 2.) / img_height)

                    for aspect_ratio in aspect_ratios:
                        box_w = min_size * np.sqrt(aspect_ratio)
                        box_h = min_size / np.sqrt(aspect_ratio)
                        anchors.append((center_x - box_w / 2.) / img_width)
                        anchors.append((center_y - box_h / 2.) / img_height)
                        anchors.append((center_x + box_w / 2.) / img_width)
                        anchors.append((center_y + box_h / 2.) / img_height)

        if (clip):
            for d in range(dim):
                anchors[d] = min(max(anchors[d], 0.), 1.)
        return anchors

def prior_one_box(img_shape,
                 feat_shape,
                 sizes,
                 ratios,
                 step,
                 offset=0.5,
                 dtype=np.float32):
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def prior_one_layer_all_box(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = prior_one_box(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors
























