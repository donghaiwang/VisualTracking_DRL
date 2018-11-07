# -*- coding: UTF-8 -*-
"""
apollo multitask for tensorflow
resNet(base network)+refineDet+SCNN for lane seg and detect.
@author: xie wei

"""
from model.VGG import VGG
from model.layers_group import *
from model.l2_normalization import l2_normalization

class RefineDet_Model(object):
    def __init__(self,anchors,anchor_ratios,anchor_steps,
                 img_size, det_num_class=2,num_anchors=9,
                training=True,keep_prob=0.5,w_summary=True,
                 name='RefineDet',reuse=False):
        self.img_size = img_size
        self.training = training
        self.name = name
        self.w_summary = w_summary
        self.keep_prob = keep_prob
        self.det_num_class = det_num_class
        self.num_anchors = num_anchors
        self.reuse = reuse
        self.use_bn = False
        self.anchors = anchors
        self.anchor_ratios=anchor_ratios
        self.anchor_steps = anchor_steps
        self.prior_scaling = [0.1, 0.1, 0.2, 0.2]
        self.arm_source_layers = ['conv4_3_norm', 'conv5_3_norm', 'fc7', 'conv6_2']
        self.odm_source_layers = ['P3', 'P4', 'P5', 'P6']
        self.normalizations = [10, 8, -1, -1]

    def ResBody(self,inputs,block_name, out2a, out2b, out2c, stride, use_branch1, dilation=1,name=None):
        # with tf.variable_scope(name):
            conv_prefix = 'res{}_'.format(block_name)
            if use_branch1:
                branch_name = 'branch1'
                branch1 = conv_bn(inputs, out2c, 1, stride, 'SAME', self.training, self.w_summary,
                                  '{}{}'.format(conv_prefix, branch_name))
            else:
                branch1 = inputs
            branch_name = 'branch2a'
            branch2a = conv_bn_relu(inputs, out2a, 1, stride, 'SAME', self.training, self.w_summary,
                                    '{}{}'.format(conv_prefix, '{}{}'.format(conv_prefix, branch_name)))
            branch_name = 'branch2b'
            if dilation == 1:
                branch2b = conv_bn_relu(branch2a, out2b, 3, 1, 'SAME', self.training, self.w_summary,
                                    '{}{}'.format(conv_prefix, branch_name))
            else:
                branch2b = astrous_conv_bn_relu(branch2a, out2b, 3, rate=dilation, pad='SAME', w_summary=False,
                                     is_training=self.training,bias=False, name='astrous_conv_bn')
            branch_name = 'branch2c'
            branch2c = conv_bn(branch2b, out2c, 1, 1, 'SAME', self.training, self.w_summary,
                              '{}{}'.format(conv_prefix, branch_name))
            res_name = 'res{}'.format(block_name)
            net = add_relu(branch1, branch2c, name=res_name)
            return net


    def AddExtraLayers(self,name):
        with tf.variable_scope(name):
            self.arm_source_layers.reverse()
            if(self.use_bn):
                self.end_logits['conv6_1'] = conv_bn_relu(self.end_logits['fc7'], 256, 1, 1, 'SAME', self.training, self.w_summary, name='conv6_1')
                self.end_logits['conv6_2'] = conv_bn_relu(self.end_logits['conv6_1'], 512, 3, 2, 'SAME', self.training,self.w_summary, name='conv6_2')
            else:
                self.end_logits['conv6_1'] = conv_relu(self.end_logits['fc7'], 256, 1, 1, 'SAME', self.training, self.w_summary, name='conv6_1')
                self.end_logits['conv6_2'] = conv_relu(self.end_logits['conv6_1'], 512, 3, 2, 'SAME', self.training,self.w_summary, name='conv6_2')

            TL6_1 = conv_relu(self.end_logits['conv6_2'], 256, 3, 1, 'SAME',self.training, self.w_summary, name='TL6_1')
            TL6_2 = conv_relu(TL6_1, 256, 3, 1, 'SAME', self.training, self.w_summary, name='TL6_2')
            self.end_logits['P6'] = conv_relu(TL6_2, 256, 3, 1, 'SAME', self.training, self.w_summary, name='P6')
            P6_up = deconv2d(self.end_logits['P6'], 256, 2,2, 'SAME',self.training,False,bias=False, name='P6_up')

            TL5_1 = conv_relu(self.end_logits['fc7'], 256, 3, 1, 'SAME',self.training, self.w_summary, name='TL5_1')
            TL5_2 = conv(TL5_1, 256, 3, 1, 'SAME', self.training, self.w_summary, name='TL5_2')
            Elt5 = add_relu(P6_up,TL5_2,'Elt5')
            self.end_logits['P5'] = conv_relu(Elt5, 256, 3, 1, 'SAME',self.training, self.w_summary, name='P5')
            P5_up = deconv2d(self.end_logits['P5'], 256, 2, 2, 'SAME', self.training, False, bias=False, name='P5_up')

            self.end_logits['conv5_3_norm'] = l2_normalization(self.end_logits['conv5_3'],'conv5_3_norm',scaling=True,scale_initializer=tf.constant_initializer(8))
            TL4_1 = conv_relu(self.end_logits['conv5_3_norm'], 256, 3, 1, 'SAME',self.training, self.w_summary, name='TL4_1')
            TL4_2 = conv(TL4_1, 256, 3, 1, 'SAME', self.training, self.w_summary, name='TL4_2')
            Elt4 = add_relu(P5_up,TL4_2,'Elt4')
            self.end_logits['P4'] = conv_relu(Elt4, 256, 3, 1, 'SAME',self.training, self.w_summary, name='P4')
            P4_up = deconv2d(self.end_logits['P4'], 256, 2, 2, 'SAME', self.training, False, bias=False, name='P4_up')

            self.end_logits['conv4_3_norm'] = l2_normalization(self.end_logits['conv4_3'],'conv4_3_norm',scaling=True,scale_initializer=tf.constant_initializer(10))
            TL3_1 = conv_relu(self.end_logits['conv4_3_norm'], 256, 3, 1, 'SAME',self.training, self.w_summary, name='TL3_1')
            TL3_2 = conv(TL3_1, 256, 3, 1, 'SAME', self.training, self.w_summary, name='TL3_2')
            Elt3 = add_relu(P4_up,TL3_2,'Elt3')
            self.end_logits['P3'] = conv_relu(Elt3, 256, 3, 1, 'SAME',self.training, self.w_summary, name='P3')

    # def AddExtraLayers(self,name):
    #     with tf.variable_scope(name):
    #         if(self.use_bn):
    #             self.end_logits['conv6_1'] = conv_bn_relu(self.end_logits['fc7'], 256, 1, 1, 'SAME', self.training, self.w_summary, name='conv6_1')
    #             self.end_logits['conv6_2'] = conv_bn_relu(self.end_logits['conv6_1'], 512, 3, 2, 'SAME', self.training,self.w_summary, name='conv6_2')
    #         else:
    #             self.end_logits['conv6_1'] = conv_relu(self.end_logits['fc7'], 256, 1, 1, 'SAME', self.training, self.w_summary, name='conv6_1')
    #             self.end_logits['conv6_2'] = conv_relu(self.end_logits['conv6_1'], 512, 3, 2, 'SAME', self.training,self.w_summary, name='conv6_2')
    #
    #         self.arm_source_layers.reverse()
    #         self.normalizations.reverse()
    #
    #         num_p = 6
    #         for index, layer in enumerate(self.arm_source_layers):
    #             from_layer = layer
    #             net = self.end_logits[from_layer]
    #
    #             '''l2_norm'''
    #             if self.normalizations:
    #                 if self.normalizations[index] != -1:
    #                     norm_name = "{}_norm".format(layer)
    #
    #                     self.end_logits[norm_name] = l2_normalization(net,norm_name,scaling=True)
    #                     net = self.end_logits[norm_name]
    #                     self.arm_source_layers[index] = norm_name
    #
    #             out_layer = "TL{}_{}".format(num_p, 1)
    #             net = conv_relu(net, 256, 3, 1, 'SAME',self.training, self.w_summary, name=out_layer)
    #
    #             if num_p == 6:
    #                 out_layer = "TL{}_{}".format(num_p, 2)
    #                 net = conv_relu(net, 256, 3, 1, 'SAME', self.training, self.w_summary,name=out_layer)
    #                 out_layer = "P{}".format(num_p)
    #                 pnet = conv_relu(net, 256, 3, 1, 'SAME', self.training, self.w_summary,name=out_layer)
    #             else:
    #                 out_layer = "TL{}_{}".format(num_p, 2)
    #                 net = conv(net, 256, 3, 1, 'SAME', self.training, self.w_summary, name=out_layer)
    #                 out_layer = "P{}_up".format(num_p + 1)
    #                 upnet = deconv2d(pnet, 256, 2,2, 'SAME',self.training,False,bias=False, name=out_layer)
    #                 net = add_relu(net,upnet,'Elt'+str(num_p))
    #                 out_layer = "P{}".format(num_p)
    #                 pnet = conv_relu(net, 256, 3, 1, 'SAME', self.training, self.w_summary, name=out_layer)
    #             self.end_logits["P" + str(num_p)] = pnet
    #             num_p = num_p - 1


    def CreateRefineDetHead(self,
                            from_layers,
                            from_layers2,
                            sizes,
                            ratios,
                            steps,
                            num_classes,
                            kernel_size):
        logist = {}
        clssfications = []
        localisations = []
        for i,layer in enumerate(from_layers):
            # Number of anchors.
            num_anchors = len(ratios[i])
            # Location.
            num_loc_pred = num_anchors * 4
            loc_pred = conv(self.end_logits[layer],num_loc_pred,kernel_size,1,'SAME',self.training,name='conv_loc_arm_'+layer)

            # Class prediction.
            num_cls_pred = num_anchors * 2
            cls_pred = conv(self.end_logits[layer], num_cls_pred, kernel_size, 1, 'SAME', self.training, name='conv_cls_arm_'+layer)

            clssfications.append(cls_pred)
            localisations.append(loc_pred)
        clssfications2 = []
        localisations2 = []
        for i,layer in enumerate(from_layers2):
            # Number of anchors.
            num_anchors = len(ratios[i])
            # Location.
            num_loc_pred = num_anchors * 4
            loc_pred = conv(self.end_logits[layer],num_loc_pred,kernel_size,1,'SAME',self.training,name='conv_loc_odm_'+layer)

            # Class prediction.
            num_cls_pred = num_anchors * num_classes
            cls_pred = conv(self.end_logits[layer], num_cls_pred, kernel_size, 1, 'SAME', self.training, name='conv_cls_odm_'+layer)

            clssfications2.append(cls_pred)
            localisations2.append(loc_pred)

        logist['arm'] = [clssfications,localisations]
        logist['odm'] = [clssfications2,localisations2]

        return logist


    def refine_det(self,name='refineDet'):
        with tf.variable_scope(name):
            share_location = True
            flip = True
            clip = False
            self.AddExtraLayers('AddExtLayers')
            self.arm_source_layers.reverse()
            self.normalizations.reverse()
            loc_conf_layers = self.CreateRefineDetHead( self.arm_source_layers,
                                                        self.odm_source_layers,
                                                        self.anchors,
                                                        self.anchor_ratios,
                                                        self.anchor_steps,
                                                        self.det_num_class,3)
            return loc_conf_layers

    def buildModel(self,inputs):
        self.end_logits = VGG(inputs,"VGG",self.training,self.w_summary,self.keep_prob,self.reuse)
        det_logits = self.refine_det('refineDet')

        return det_logits








