# -*- coding: UTF-8 -*-
"""
vgg(base network)->RefineDet_tf for vechile detection.
@author: xie wei
"""
from model.layers_group import *
slim = tf.contrib.slim
def VGG(inputs,name,training=True,w_summary=True,keep_prob = 1.0,use_bn = False, reuse = False):
    with tf.variable_scope(name,reuse=reuse):
        end_points_collection = name + '_end_logits'
        if (use_bn):
            conv1_1 = conv_bn_relu(inputs, 64, 3, 1,'SAME', training, w_summary, name='conv1_1')
            conv1_2 = conv_bn_relu(conv1_1, 64, 3, 1, 'SAME', training, w_summary, name='conv1_2')
            pool1 = pool(conv1_2,2, 2, 'max', name='pool1')

            conv2_1 = conv_bn_relu(pool1, 128, 3, 1, 'SAME', training, w_summary, name='conv2_1')
            conv2_2 = conv_bn_relu(conv2_1, 128, 3, 1, 'SAME', training, w_summary, name='conv2_2')
            pool2 = pool(conv2_2, 2, 2, 'max', name='pool2')

            conv3_1 = conv_bn_relu(pool2, 256, 3, 1, 'SAME', training, w_summary, name='conv3_1')
            conv3_2 = conv_bn_relu(conv3_1, 256, 3, 1, 'SAME', training, w_summary, name='conv3_2')
            conv3_3 = conv_bn_relu(conv3_2, 256, 3, 1, 'SAME', training, w_summary, name='conv3_3')
            pool3 = pool(conv3_3, 2, 2, 'max', name='pool3')

            conv4_1 = conv_bn_relu(pool3, 512, 3, 1, 'SAME', training, w_summary, name='conv4_1')
            conv4_2 = conv_bn_relu(conv4_1, 512, 3, 1, 'SAME', training, w_summary, name='conv4_2')
            conv4_3 = conv_bn_relu(conv4_2, 512, 3, 1, 'SAME', training, w_summary, name='conv4_3')
            pool4 = pool(conv4_3, 2, 2, 'max', name='pool4')

            conv5_1 = conv_bn_relu(pool4, 512, 3, 1, 'SAME', training, w_summary, name='conv5_1')
            conv5_2 = conv_bn_relu(conv5_1, 512, 3, 1, 'SAME', training, w_summary, name='conv5_2')
            conv5_3 = conv_bn_relu(conv5_2, 512, 3, 1, 'SAME', training, w_summary, name='conv5_3')
            pool5 = pool(conv5_3, 2, 2, 'max', name='pool5')
        else:
            conv1_1 = conv_relu(inputs, 64, 3, 1, 'SAME', training, w_summary, bias=False,name='conv1_1')
            conv1_2 = conv_relu(conv1_1, 64, 3, 1, 'SAME', training, w_summary, bias=False,name='conv1_2')
            pool1 = pool(conv1_2, 2, 2, 'max', name='pool1')

            conv2_1 = conv_relu(pool1, 128, 3, 1, 'SAME', training, w_summary, bias=False,name='conv2_1')
            conv2_2 = conv_relu(conv2_1, 128, 3, 1, 'SAME', training, w_summary, bias=False,name='conv2_2')
            pool2 = pool(conv2_2, 2, 2, 'max', name='pool2')

            conv3_1 = conv_relu(pool2, 256, 3, 1, 'SAME', training, w_summary, bias=False,name='conv3_1')
            conv3_2 = conv_relu(conv3_1, 256, 3, 1, 'SAME', training, w_summary, bias=False,name='conv3_2')
            conv3_3 = conv_relu(conv3_2, 256, 3, 1, 'SAME', training, w_summary, bias=False,name='conv3_3')
            pool3 = pool(conv3_3, 2, 2, 'max', name='pool3')

            conv4_1 = conv_relu(pool3, 512, 3, 1, 'SAME', training, w_summary, bias=False,name='conv4_1')
            conv4_2 = conv_relu(conv4_1, 512, 3, 1, 'SAME', training, w_summary, bias=False,name='conv4_2')
            conv4_3 = conv_relu(conv4_2, 512, 3, 1, 'SAME', training, w_summary, bias=False,name='conv4_3')
            pool4 = pool(conv4_3, 2, 2, 'max', name='pool4')

            conv5_1 = conv_relu(pool4, 512, 3, 1, 'SAME', training, w_summary, name='conv5_1')
            conv5_2 = conv_relu(conv5_1, 512, 3, 1, 'SAME', training, w_summary, name='conv5_2')
            conv5_3 = conv_relu(conv5_2, 512, 3, 1, 'SAME', training, w_summary, name='conv5_3')
            pool5 = pool(conv5_3, 2, 2, 'max', name='pool5')
        fc6 = astrous_conv_relu(pool5,1024,3,3,'SAME',w_summary,training,True,name='fc6')
        fc7 = conv_relu(fc6, 1024, 1, 1, 'SAME', w_summary, training, True, name='fc7')
        end_logits = slim.utils.convert_collection_to_dict(end_points_collection)
        end_logits['conv4_3'] = conv4_3
        end_logits['conv5_3'] = conv5_3
        end_logits['fc7'] = fc7

        return end_logits






