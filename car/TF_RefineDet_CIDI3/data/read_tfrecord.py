# # -*- coding: utf-8 -*-
# """
# read tfrecord.
# @author: xie wei
# """
import tensorflow as tf
import os
import numpy as np
from model.candidate_box_process import *
from data.dataAugement import Augement
import cv2

# VOC 70 (104, 117, 123)
# CIDI_data 30 (64, 68, 66)
# BDD100k 70 (74,75,71)
max_object_num = 30
IMG_MEAN = np.array((71,75,74), dtype=np.float32)
# imshape = [1024,1024,3]

def reshape_list(l, shape=None):
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

def train_parse_fn(example,batch_size,anchors_layers,num_of_classes,imshape):
    """
    :param example: 序列化的输入
    :return:
    """
    features = tf.parse_single_example(
        serialized=example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string),
            'img_shape': tf.FixedLenFeature([], tf.string),
            'xmin': tf.VarLenFeature(dtype=tf.float32),
            'xmax': tf.VarLenFeature(dtype=tf.float32),
            'ymin': tf.VarLenFeature(dtype=tf.float32),
            'ymax': tf.VarLenFeature(dtype=tf.float32),
            'cls_label': tf.VarLenFeature(dtype=tf.int64),
        }
    )
    img_name = features['img_name']
    img = tf.decode_raw(features['img'], tf.uint8)
    img_shape = tf.decode_raw(features['img_shape'], tf.int32)
    xmin = tf.sparse_tensor_to_dense(features['xmin'])
    ymin = tf.sparse_tensor_to_dense(features['ymin'])
    xmax = tf.sparse_tensor_to_dense(features['xmax'])
    ymax = tf.sparse_tensor_to_dense(features['ymax'])
    cls_label = tf.sparse_tensor_to_dense(features['cls_label'])

    reg_label = tf.stack(values=[ymin, xmin, ymax, xmax], axis=1)
    reg_label.set_shape([max_object_num, 4])
    cls_label.set_shape([max_object_num,])

    cls_label = tf.reshape(cls_label, [-1, 1])
    # cls_label = tf.reshape(cls_label, [-1, ])
    reg_label = tf.reshape(reg_label, [-1, 4])
    mask = cls_label > 0
    imask = tf.cast(mask, tf.int64)
    num_box = tf.reduce_sum(imask)

    img_shape = tf.reshape(img_shape, (3,))
    img = tf.reshape(img, [img_shape[0], img_shape[1], img_shape[2]])
    cls_label_real = cls_label[:num_box]
    reg_label_real = reg_label[:num_box]

    augement = Augement(img, reg_label_real, cls_label_real, img_shape)
    img, reg_label_real, cls_label_real, img_shape = augement.execute()



    # return img, reg_label_real, cls_label_real, img_shape
    ratio_h = tf.cast(tf.cast(img_shape[0],tf.float32) / tf.cast(imshape[0],tf.float32),tf.float32)
    ratio_w = tf.cast(tf.cast(img_shape[1],tf.float32) / tf.cast(imshape[1],tf.float32),tf.float32)
    img = tf.image.resize_images(img, (imshape[0], imshape[1]), method=0)
    img.set_shape([imshape[0], imshape[1],3])

    # reg_label = tf.cast(tf.reshape(reg_label,(-1, 4)),tf.float32)

    reg_label_x1 = tf.reshape(reg_label_real[:, 1]*tf.cast(img_shape[1],tf.float32) / ratio_w / tf.cast(imshape[1],tf.float32),[-1,1])
    reg_label_x2 = tf.reshape(reg_label_real[:, 3]*tf.cast(img_shape[1],tf.float32) / ratio_w / tf.cast(imshape[1],tf.float32),[-1,1])
    reg_label_y1 = tf.reshape(reg_label_real[:, 0]*tf.cast(img_shape[0],tf.float32) / ratio_h / tf.cast(imshape[0],tf.float32),[-1,1])
    reg_label_y2 = tf.reshape(reg_label_real[:, 2]*tf.cast(img_shape[0],tf.float32) / ratio_h / tf.cast(imshape[0],tf.float32),[-1,1])
    reg_label_real = tf.concat([reg_label_x1,reg_label_y1,reg_label_x2,reg_label_y2],axis=1)


    num_box_a = tf.reduce_sum(tf.cast(cls_label_real>0,tf.int64))
    reg_no_obj = reg_label[num_box_a:]*0 + -1
    cls_no_obj = cls_label[num_box_a:]*0 + -1
    reg_label = tf.concat(values=[reg_label_real, reg_no_obj], axis=0)
    cls_label = tf.concat(values=[cls_label_real, cls_no_obj], axis=0)

    reg_label.set_shape([max_object_num, 4])
    cls_label.set_shape([max_object_num, 1])
    reg_label = tf.reshape(reg_label, [-1, 4])
    cls_label = tf.reshape(cls_label, [-1, ])
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img,0,255)
    img -= IMG_MEAN

    gclasses, glocalisations, gscores = bboxes_encode(
                                        cls_label_real,
                                        reg_label_real,
                                        anchors_layers,
                                        num_of_classes)


    r = tf.train.shuffle_batch(reshape_list([img, gclasses, glocalisations, gscores,img_name,reg_label,cls_label,num_box]),
                                                    batch_size= batch_size,
                                                    num_threads=4,
                                                    capacity=1000,
                                                    min_after_dequeue=500)
    img_batch, cls_batch, reg_batch, score_batch, name_batch, bbox_batch,cls_batch2,num_box_batch = \
        reshape_list(r, [1,len(gclasses),len(glocalisations),len(gscores),1,1,1,1])

    return img_batch, cls_batch, reg_batch, score_batch, name_batch, bbox_batch ,cls_batch2,num_box_batch

def train_input_fn(data_dir, batch_size, anchors_layers,num_of_classes,img_size):
    data_files = tf.gfile.Glob("/home/xj/桌面/tf_refineDet_dug/data/tfrecord/train.tfrecord")
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_batch, cls_batch, reg_batch, score_batch, name_batch, bbox_batch, cls_batch2,num_box_batch \
        = train_parse_fn(serialized_example,batch_size,anchors_layers,num_of_classes,img_size)

    # coord = tf.train.Coordinator()
    # init = tf.initialize_all_variables()
    # Session = tf.Session()
    # Session.run(init)
    # threads = tf.train.start_queue_runners(coord=coord, sess=Session)
    # for i in range(10000):
    #     imgs, bbox, cls, score, names, gt_box, num_bbox = Session.run(
    #         [img_batch, reg_batch, cls_batch, score_batch, name_batch, bbox_batch, num_box_batch])
    #
    #     scor = []
    #     s_ = []
    #     for s, sco in enumerate(score):
    #         scor.append(np.max(sco[0, :, :, :]))
    #         s_.append(s)
    #     id = np.argmax(scor)
    #     print(names[0], [s_[id], scor[id]])
    #
    #     import cv2
    #     imgs = (imgs[0, :, :, :] + IMG_MEAN).astype(np.uint8)
    #     imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    #     for i in range(gt_box[0].shape[0]):
    #         bbox = gt_box[0][i, :] * img_size[0]
    #         # if (bbox[0] == -1):
    #         #     continue
    #         imgs = cv2.rectangle(imgs.astype(np.uint8), (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
    #                              (0, 0, 255), 2)
    #     cv2.imshow('seg', imgs)
    #     cv2.waitKey()
    #
    # coord.request_stop()
    # coord.join(threads)

    return img_batch,reg_batch,cls_batch,score_batch,bbox_batch, cls_batch2


