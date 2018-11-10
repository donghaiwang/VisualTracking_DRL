# -*- coding: UTF-8 -*-
"""
vgg(base network)->RefineDet_tf for vechile detection.
@author: xie wei
"""
import time
import cv2
import tensorflow as tf
import numpy as np
import datetime
import os
from model.RefineNet import RefineDet_Model
from model.candidate_box_process import *
from model.anchors_layer import *
from model.layers_group import *
import scipy.misc as scm
import tensorflow.contrib.slim as slim


class Test_Moedel():
    def __init__(self,baseNet='VGG',batch_size = 2,img_size=[512,512,3],learn_rate=0.001,decay = 0.96,
                 decay_step = 200, training = True,keep_prob=0.5,w_summary = True,num_of_classes=5,
                 savepath='log/save',use_premodel=True,pretrain_modelpath='convert_model/save/model.ckpt',
                 name = 'RefineDet'):
        self.testing = True
        self.baseNet=baseNet
        self.img_size = img_size
        self.name = name
        self.num_of_classes = num_of_classes
        self.thresh = 0.5
        self.keep_prob = keep_prob
        self.IMG_MEAN = np.array((71, 75, 74), dtype=np.float32)
        self.GPU_GROUPS = ["/gpu:0"]
        self.model_name='RefineDet'
        self.anchor_sizes =[(32, 32),
                            (64, 64),
                            (128, 128),
                            (256, 256)]
        self.anchor_ratios=[[1, 2, .5],
                            [1, 2, .5],
                            [1, 2, .5],
                            [1, 2, .5]]
        self.anchor_steps = [8, 16, 32, 64]
        self.prior_scaling = [0.1, 0.1, 0.2, 0.2]
        self.feat_shapes = [(img_size[1] // self.anchor_steps[0],
                             img_size[0] // self.anchor_steps[0]),
                            (img_size[1] // self.anchor_steps[1],
                             img_size[0] // self.anchor_steps[1]),
                            (img_size[1] // self.anchor_steps[2],
                             img_size[0] // self.anchor_steps[2]),
                            (img_size[1] // self.anchor_steps[3],
                             img_size[0] // self.anchor_steps[3])]
        self.anchors_layers = anchors_all_layers(self.img_size,
                                            self.feat_shapes,
                                            self.anchor_sizes,
                                            self.anchor_ratios,
                                            self.anchor_steps,
                                            offset=0.5,
                                            dtype=np.float32)

    def build_model(self):
        tf.reset_default_graph()
        if (self.testing):
            self.image = tf.placeholder(dtype=tf.float32, shape=(1, self.img_size[1],
                                    self.img_size[0],self.img_size[2]), name='input_img')
            self.RefineDet = RefineDet_Model(self.anchor_sizes,self.anchor_ratios,
                                             self.anchor_steps,
                                             img_size=self.img_size,
                                             det_num_class=self.num_of_classes,
                                             training=False,
                                             keep_prob=self.keep_prob,
                                             name='RefineDet',
                                             reuse=False)
            self.pred = self.RefineDet.buildModel(self.image)

            self.pred['arm'][0] = [tf.nn.softmax(tf.reshape(layer,[layer.shape.as_list()[0],layer.shape.as_list()[1],
                                                   layer.shape.as_list()[2],layer.shape.as_list()[3]//2,2]))
                                                    for layer in self.pred['arm'][0]]
            self.pred['odm'][0] = [tf.nn.softmax(tf.reshape(tf.nn.softmax(layer),[layer.shape.as_list()[0],layer.shape.as_list()[1],
                                                   layer.shape.as_list()[2],layer.shape.as_list()[3]//self.num_of_classes,
                                                   self.num_of_classes])) for layer in self.pred['odm'][0]]

            delete_nmask = [nsocre[:,:,:,:,0] <= 0.99 for nsocre in self.pred['arm'][0]]
            self.pred['odm'][0] = [score * tf.cast(tf.expand_dims(delete_nmask[s],axis=-1),tf.float32) for s,score in enumerate(self.pred['odm'][0])]

            self.pred['arm'][1] = [tf.reshape(layer,[layer.shape.as_list()[0],layer.shape.as_list()[1],
                                                   layer.shape.as_list()[2],layer.shape.as_list()[3]//4,
                                                   4]) for layer in self.pred['arm'][1]]
            self.pred['odm'][1] = [tf.reshape(layer,[layer.shape.as_list()[0],layer.shape.as_list()[1],
                                                    layer.shape.as_list()[2],layer.shape.as_list()[3]//4,
                                                   4]) for layer in self.pred['odm'][1]]

            self.pred['arm'][1] = tf_bboxes_decode(self.pred['arm'][1], self.anchors_layers)
            self.pred['odm'][1] = tf_bboxes_decode(self.pred['odm'][1], self.pred['arm'][1],is_refine=True)

    def load_model(self, model_path):
        with tf.name_scope('Session'):
            self.init = tf.global_variables_initializer()
            print('Session initialization')
            conf = tf.ConfigProto(allow_soft_placement=True)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            conf.gpu_options.allow_growth = True
            self.Session = tf.Session(config=conf)
            self.Session.run(self.init)
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                t = time.time()
                self.saver.restore(self.Session, ckpt.model_checkpoint_path)
                print('Model Loaded (', time.time() - t, ' sec.)')
            else:
                print('Please give a Model in args ...')


    def close(self):
        self.Session.close()


    def detect(self, image_path):
        # name = name.split('\n')[0]
        # img_name = name.split('/')[-1]
        img_src = scm.imread(image_path)
        img_src_w = img_src.shape[1]
        img_src_h = img_src.shape[0]
        w, h, _ = self.img_size
        img = scm.imresize(img_src, (h, w))
        img = img.astype(np.float32)
        img -= self.IMG_MEAN
        img = np.expand_dims(img, axis=0)
        start = time.time()
        scores, locations = self.Session.run([self.pred['odm'][0], self.pred['odm'][1]], feed_dict={self.image: img})
        # self.Session.close()

        rclasses, rscores, rbboxes = bboxes_select(scores,
                                                   locations,
                                                   self.anchors_layers,
                                                   select_threshold=0.5,
                                                   img_shape=(self.img_size[0], self.img_size[1]),
                                                   num_classes=self.num_of_classes,
                                                   decode=False)
        rbbox_img = [0., 0., 1., 1.]
        rbboxes = bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = bboxes_sort(rclasses, rscores, rbboxes, top_k=1000)
        rclasses, rscores, rbboxes = bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.2)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = bboxes_resize(rbbox_img, rbboxes)

        index = 0

        # bboxes = np.zeros((rbboxes.size, 4))
        bboxes = []
        for bb in rbboxes:
            bbox = []
            p1 = [int(bb[0] * self.img_size[1]), int(bb[1] * self.img_size[0])]
            p2 = [int(bb[2] * self.img_size[1]), int(bb[3] * self.img_size[0])]

            '''对应到原图'''
            p1[0] = round(p1[0] * (img_src_w / self.img_size[0]))
            p1[1] = round(p1[1] * (img_src_h / self.img_size[1]))
            p2[0] = round(p2[0] * (img_src_w / self.img_size[0]))
            p2[1] = round(p2[1] * (img_src_h / self.img_size[1]))

            # print(p1[0], p1[1], p2[0], p2[1])
            # bboxes(bb) =  [p1[0], p1[1], p2[0], p2[1]];
            bbox.append(p1[0])
            bbox.append(p1[1])
            bbox.append(p2[0])
            bbox.append(p2[1])
            bboxes.append(bbox)
            index += 1
        return bboxes




