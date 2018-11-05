# -*- coding: utf-8 -*-
"""
@author: yangxuefeng
"""
import numpy as np
import tensorflow as tf

IMG_MEAN = np.array((74,75,71), dtype=np.float32)

class Augement():
    def __init__(self,image,reg_label_real,cls_label,shape):

        self.images = image
        self.reg_label_real = reg_label_real
        self.cls_label = cls_label
        self.shape = shape

    def execute(self):
        flag = tf.random_uniform(shape=[],minval=3,maxval=4,dtype=tf.int32)
        images, reg_label_real, cls_label = tf.case({tf.equal(flag, 0): self.order1,
                         tf.equal(flag, 1): self.order2,
                         tf.equal(flag, 2): self.order3,
                         tf.equal(flag, 3): self.order4
                         }, exclusive=True)
        img_shape = tf.shape(images)
        return images, reg_label_real, tf.reshape(cls_label,[-1,1]),img_shape
    def order1(self):
        images0, reg_label_real0, cls_label0 = self.crop(self.images, self.reg_label_real, self.cls_label)
        images1, reg_label_real1, cls_label1 = self.color(images0, reg_label_real0, cls_label0)
        images2, reg_label_real2, cls_label2 = self.flip(images1, reg_label_real1, cls_label1)
        return images2, reg_label_real2, cls_label2
    def order2(self):
        images0, reg_label_real0, cls_label0 = self.padding(self.images,self.reg_label_real,self.cls_label,4,self.shape)
        images1, reg_label_real1, cls_label1 = self.color(images0, reg_label_real0, cls_label0)
        images2, reg_label_real2, cls_label2 = self.flip(images1, reg_label_real1, cls_label1 )
        return images2, reg_label_real2, cls_label2
    def order3(self):
        return self.images,self.reg_label_real,self.cls_label

    def order4(self):
        is_do = tf.random_uniform(shape=[],minval=0,maxval=2,dtype=tf.int32)
        images0, reg_label_real0, cls_label0 = tf.cond(tf.equal(is_do,0),lambda:self.color(self.images, self.reg_label_real, self.cls_label),lambda:self.returnsrc(self.images, self.reg_label_real, self.cls_label))

        is_do = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        images1, reg_label_real1, cls_label1 = tf.cond(tf.equal(is_do, 0),
                                                       lambda: self.padding(images0, reg_label_real0, cls_label0,2,self.shape),lambda:self.returnsrc(images0, reg_label_real0, cls_label0))

        is_do = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        images2, reg_label_real2, cls_label2 = tf.cond(tf.equal(is_do, 0),
                                                       lambda: self.crop(images1, reg_label_real1, cls_label1),lambda:self.returnsrc(images1, reg_label_real1, cls_label1))

        is_do = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        images3, reg_label_real3, cls_label3 = tf.cond(tf.equal(is_do, 0),
                                                       lambda: self.flip(images2, reg_label_real2, cls_label2),lambda:self.returnsrc(images2, reg_label_real2, cls_label2))
        return images3, reg_label_real3, cls_label3

    def returnsrc(self,images,reg_label_real,cls_label):
        return images,reg_label_real,cls_label


    def color(self,images,reg_label_real,cls_label):
        def f1():
            image = tf.image.random_brightness(images, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image
        def f2():
            image = tf.image.random_saturation(images, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            return image
        def f3():
            image = tf.image.random_contrast(images, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            return image
        def f4():
            image = tf.image.random_hue(images, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            return image
        color_ordering = tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.case({tf.equal(color_ordering, 0): f1,
                      tf.equal(color_ordering, 1): f2,
                      tf.equal(color_ordering, 2): f3,
                      tf.equal(color_ordering, 3): f4},exclusive=True)
        return image, reg_label_real, cls_label

    def flip(self,images,reg_label_real,cls_label):
        image = tf.image.flip_left_right(images)
        ymin = reg_label_real[:,0]
        xmin = 1.0 - reg_label_real[:,3]
        ymax = reg_label_real[:,2]
        xmax = 1.0 - reg_label_real[:,1]
        reg_label_realNew = tf.stack(values=[ymin, xmin, ymax, xmax], axis=1)
        reg_label_realNew = tf.reshape(reg_label_realNew,[-1,4])
        return image, reg_label_realNew, cls_label

    def padding(self, images,reg_label_real,cls_label,ratio,shape):
        ratios = tf.random_uniform(shape=[], minval=1.0, maxval=ratio, dtype=tf.float32)
        shapesize = tf.cast(shape,tf.float32)
        width = shapesize[1] * ratios
        hight = shapesize[0] * ratios
        offset_h = tf.random_uniform(shape=[],minval=0,dtype=tf.float32,maxval=hight-shapesize[0])
        offset_w = tf.random_uniform(shape=[],minval=0,dtype=tf.float32,maxval=width-shapesize[1])
        offset_h = tf.cast(offset_h,tf.int32)
        offset_w = tf.cast(offset_w, tf.int32)
        width = tf.cast(width,tf.int32)
        hight = tf.cast(hight, tf.int32)
        padding = [[offset_h,hight-tf.cast(shapesize[0],tf.int32)-tf.cast(offset_h,tf.int32)],[offset_w,width-tf.cast(shapesize[1],tf.int32)-tf.cast(offset_w,tf.int32)]]
        image_0 = tf.pad(tensor=images[:,:,0],paddings=padding,constant_values=IMG_MEAN[0])
        image_1 = tf.pad(tensor=images[:, :, 1], paddings=padding, constant_values=IMG_MEAN[1])
        image_2 = tf.pad(tensor=images[:, :, 2], paddings=padding, constant_values=IMG_MEAN[2])
        image = tf.stack(values=[image_0,image_1,image_2],axis=-1)
        offset_h = tf.cast(offset_h, tf.float32)
        offset_w = tf.cast(offset_w, tf.float32)
        width = tf.cast(width, tf.float32)
        hight = tf.cast(hight, tf.float32)
        ymin = (reg_label_real[:,0]*shapesize[0]+offset_h)/hight
        xmin = (reg_label_real[:,1]*shapesize[1]+offset_w)/width
        ymax = (reg_label_real[:,2]*shapesize[0]+offset_h)/hight
        xmax = (reg_label_real[:,3]*shapesize[1]+offset_w)/width
        reg_label_realNew = tf.stack(values=[ymin, xmin, ymax, xmax], axis=1)
        return image, reg_label_realNew, cls_label

    def crop(self, images,reg_label_real,cls_label):
        reg_label_real0 = tf.transpose(reg_label_real)
        ymin,xmin,ymax,xmax = tf.split(reg_label_real0,4,0)
        reg_label_real_withLab = tf.stack(values=[ymin,xmin,ymax,xmax,tf.cast(tf.transpose(cls_label),tf.float32)], axis=1)
        reg_label_real_withLab = tf.reshape(tf.transpose(reg_label_real_withLab),[-1,5])
        tf_image = tf.cast(images, dtype=tf.float32)
        bounding_boxes = tf.expand_dims(reg_label_real, 0)
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(tf_image),
            bounding_boxes=bounding_boxes,
            min_object_covered=0.3,
            aspect_ratio_range=(0.5, 2),
            area_range=(0.3, 1.0),
            max_attempts=None,
            use_image_if_no_bounding_boxes=True,
            name=None
        )
        image_with_box = tf.squeeze(tf.cast(tf.image.draw_bounding_boxes(tf.expand_dims(tf_image, 0), bbox_for_draw), tf.uint8))
        distorted_image = tf.cast(tf.slice(tf_image, begin, size), tf.uint8)
        distort_bbox = bbox_for_draw[0, 0]
        filter_box = self.bboxes_intersection_filter(distort_bbox, reg_label_real_withLab)
        filter_box = tf.reshape(filter_box,[-1,5])
        return distorted_image,filter_box[:,0:4],tf.cast(filter_box[:,4],tf.int64)
    def bboxes_intersection_filter(self,bbox_ref, bboxes, threshold=0.3):
        # thresholds = tf.random_uniform(shape=[], minval=0, maxval=6, dtype=tf.int32)
        # threshold = tf.case({tf.equal(thresholds, 0): lambda :0.1,
        #                                              tf.equal(thresholds, 1):  lambda :0.3,
        #                                              tf.equal(thresholds, 2):  lambda :0.5,
        #                                              tf.equal(thresholds, 3):  lambda :0.7,
        #                                              tf.equal(thresholds, 4): lambda: 0.9,
        #                                              tf.equal(thresholds, 5): lambda: 1.0
        #                                              }, exclusive=True)


        int_ymin = tf.maximum(bboxes[:,0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[:,1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[:,2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[:,3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        bboxes_vol = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
        scores =tf.divide(inter_vol, bboxes_vol)
        clip_ymin = (tf.clip_by_value(bboxes[:,0], bbox_ref[0], bbox_ref[2])-bbox_ref[0])/(bbox_ref[2] - bbox_ref[0])
        clip_xmin = (tf.clip_by_value(bboxes[:, 1], bbox_ref[1], bbox_ref[3])-bbox_ref[1])/(bbox_ref[3] - bbox_ref[1])
        clip_ymax = (tf.clip_by_value(bboxes[:, 2], bbox_ref[0], bbox_ref[2])-bbox_ref[0])/(bbox_ref[2] - bbox_ref[0])
        clip_xmax = (tf.clip_by_value(bboxes[:, 3], bbox_ref[1], bbox_ref[3])-bbox_ref[1])/(bbox_ref[3] - bbox_ref[1])
        clip_cls = bboxes[:, 4]
        bboxes = tf.stack(values=[clip_ymin, clip_xmin, clip_ymax, clip_xmax,clip_cls], axis=1)
        filter_score = tf.gather(bboxes, tf.squeeze(tf.where(tf.greater_equal(scores,threshold))))
        return filter_score