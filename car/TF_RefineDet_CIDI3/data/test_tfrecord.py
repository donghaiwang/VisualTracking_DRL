#coding=utf-8
import tensorflow as tf
import os
import numpy as np
import cv2
from model.candidate_box_process import *
from model.anchors_layer import *
IMG_MEAN = np.array((104, 116, 122), dtype=np.float32)
imshape = [640,640,3]
# h = 360
# w = 640

anchor_sizes =[
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256)]
anchor_ratios=[[1, 2, .5],
                [1, 2, .5],
                [1, 2, .5],
                [1, 2, .5]]
anchor_steps = [8, 16, 32, 64]
prior_scaling = [0.1, 0.1, 0.2, 0.2]
feat_shapes = [(imshape[0] / anchor_steps[0],
                imshape[1] / anchor_steps[0]),
                (imshape[0] / anchor_steps[1],
                 imshape[1] / anchor_steps[1]),
                (imshape[0] / anchor_steps[2],
                 imshape[1] / anchor_steps[2]),
                (imshape[0] / anchor_steps[3],
                 imshape[1] / anchor_steps[3])]

anchors_layers = anchors_all_layers(imshape,
                                     feat_shapes,
                                     anchor_sizes,
                                     anchor_ratios,
                                     anchor_steps,
                                     offset=0.5,
                                     dtype=np.float32)
print()

def train_parse_fn(example,random_scale=True,batch_size=2):
    """
    :param example: 序列化的输入
    :return:
    """
    features = tf.parse_single_example(
        serialized=example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_shape': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string),
            'reg_label': tf.FixedLenFeature([], tf.string),
            'cls_label': tf.FixedLenFeature([], tf.string)
        }
    )
    img_name = features['img_name']
    img = tf.decode_raw(features['img'], tf.uint8)
    img_shape = tf.decode_raw(features['img_shape'], tf.int32)
    reg_label = tf.decode_raw(features['reg_label'],tf.float32)
    cls_label = tf.decode_raw(features['cls_label'], tf.int32)

    img_shape = tf.reshape(img_shape,(3,))
    img = tf.reshape(img, [img_shape[0],img_shape[1],img_shape[2]])
    ratio_h = tf.cast(tf.cast(img_shape[0],tf.float32) / tf.cast(imshape[0],tf.float32),tf.float32)
    ratio_w = tf.cast(tf.cast(img_shape[1],tf.float32) / tf.cast(imshape[1],tf.float32),tf.float32)
    img = tf.image.resize_images(img, (imshape[0], imshape[1]), method=0)
    reg_label = tf.cast(tf.reshape(reg_label,(-1, 4)),tf.float32)
    reg_label_x1 = tf.reshape(reg_label[:, 0]*tf.cast(img_shape[1],tf.float32) / ratio_w / tf.cast(imshape[1],tf.float32),[-1,1])
    reg_label_x2 = tf.reshape(reg_label[:, 2]*tf.cast(img_shape[1],tf.float32) / ratio_w / tf.cast(imshape[1],tf.float32),[-1,1])
    reg_label_y1 = tf.reshape(reg_label[:, 1]*tf.cast(img_shape[0],tf.float32) / ratio_h / tf.cast(imshape[0],tf.float32),[-1,1])
    reg_label_y2 = tf.reshape(reg_label[:, 3]*tf.cast(img_shape[0],tf.float32) / ratio_h / tf.cast(imshape[0],tf.float32),[-1,1])

    reg_label = tf.concat([reg_label_x1,reg_label_y1,reg_label_x2,reg_label_y2],axis=1)
    reg_label = tf.reshape(reg_label,[-1,4])
    cls_label = tf.reshape(cls_label,[-1,])
    img = tf.cast(img, tf.float32)
    img = img - IMG_MEAN

    gclasses, glocalisations, gscores = bboxes_encode(
                                        batch_size,
                                        cls_label,
                                        reg_label,
                                        anchors_layers,
                                        2)

    decode_box = tf_bboxes_decode([tf.expand_dims(g,axis=0) for g in glocalisations], anchors_layers)

    return img, reg_label,cls_label,img_shape,gclasses,glocalisations,gscores,decode_box

def train_input_fn(data_dir, batch_size=2, epochs=1):
    data_files = tf.gfile.Glob(os.path.join(data_dir, "/home/xj/桌面/tf_refineDet_dug/data/tfrecord/train.tfrecord"))
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_batch, reg_batch, cls_batch,shape_batch,gclass,glocalisations, gscores,decode_box\
        = train_parse_fn(serialized_example,random_scale=True,batch_size=batch_size)

    return img_batch, reg_batch, cls_batch,shape_batch,gclass,glocalisations,gscores,decode_box


def test():
    image_batch, seg_label_batch, cls_label_batch,shape_batch,\
    gclasses_batch,glocalisations_batch, gscores_batch,decode_box_batch = train_input_fn("tfrecord",1)
    coord = tf.train.Coordinator()
    init = tf.initialize_all_variables()
    Session = tf.Session()
    Session.run(init)
    threads = tf.train.start_queue_runners(coord=coord, sess=Session)
    for i in range(5000):

        imgs,bbox,cls,shape,gclasses,glocalisations,gscores,decode_box, \
                    = Session.run([image_batch, seg_label_batch,
                    cls_label_batch,shape_batch,gclasses_batch,glocalisations_batch, gscores_batch,decode_box_batch])

        imgs = (imgs+IMG_MEAN).astype(np.uint8)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)


        for i2,g in enumerate(gscores):
            if (i2 == 0):
                color = (200, 0, 0)
            elif (i2 == 1):
                color = (0, 200, 0)
            elif (i2 == 2):
                color = (200, 0, 200)
            else:
                color = (200, 200, 0)

            anchors_positions=[]
            max_index = np.where(g==np.max(g))
            for j in range(max_index[0].shape[0]):
                pos = [max_index[0][j],max_index[1][j],max_index[2][j]]
                anchor_center = [anchors_layers[i2][0][pos[0],pos[1],0],anchors_layers[i2][1][pos[0],pos[1],0]]
                anchor_size = [anchors_layers[i2][2][pos[2]],anchors_layers[i2][3][pos[2]]]
                anchor_pos = [anchor_center[0]-anchor_size[0]/2.,
                              anchor_center[1] - anchor_size[1] / 2.,
                              anchor_center[0] + anchor_size[0] / 2.,
                              anchor_center[1] + anchor_size[1] / 2.]
                anchors_positions.append(anchor_pos)

            for b,bb in enumerate(anchors_positions):
                imgs = cv2.rectangle(imgs.astype(np.uint8),
                                     (int(bb[0] * imshape[1]), int(bb[1] * imshape[0])),
                                     (int(bb[2] * imshape[1]), int(bb[3] * imshape[0])), color, 2)

        for i in range(bbox.shape[0]):
           imgs = cv2.rectangle(imgs.astype(np.uint8),(int(bbox[i,0]*imshape[1]),int(bbox[i,1]*imshape[0])),
                                (int(bbox[i,2]*imshape[1]),int(bbox[i,3]*imshape[0])),(0,0,255),2)

           scale = np.sqrt((bbox[i,2] - bbox[i,0]) * (bbox[i,3] - bbox[i,1]))
           print("scale...: " + str(scale))

        cv2.imshow('seg',imgs)
        cv2.waitKey()

    coord.request_stop()
    coord.join(threads)
print('starting ....')
test()



