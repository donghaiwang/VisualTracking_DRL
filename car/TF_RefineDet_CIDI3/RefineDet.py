#coding:utf-8
from test import *

import scipy.misc as scm
import os
# import matplotlib.pyplot as plt
import cv2

# -*- coding: UTF-8 -*-
"""
vgg(base network)->RefineDet_tf for vechile detection.
@author: xie wei
"""


def init():
    mode = 'test'
    imshape = [768, 768, 3]
    batch_size = 12
    learning_rate = 0.001
    decay = 0.95
    decay_step = 8000
    random_scale = True
    use_premodel = True
    model = Test_Moedel(baseNet='VGG', batch_size=batch_size,
                        img_size=imshape, learn_rate=learning_rate,
                        decay=decay, decay_step=decay_step,
                        training=mode, keep_prob=1.0, w_summary=False,
                        num_of_classes=2, savepath='log/save',
                        use_premodel=True, pretrain_modelpath=
                        'convert_model/save/model.ckpt', name='RefineDet')
    # model_path = os.getcwd() + os.sep + 'log' + os.sep + 'save_bbd'
    # must full path, otherwise detect many boxes
    # model_path = '/home/laoli/rl/VisualTracking_DRL/car/TF_RefineDet_CIDI3/log/save_bbd'
    model_path = '/home/laoli/rl/VisualTracking_DRL/car/TF_RefineDet_CIDI3/log/save_cidi'
    model.build_model()
    model.load_model(model_path)
    return model
    # detectronRes = model.detection(image_path)
    # return detectronRes


if __name__ == "__main__":
    image_name_1 = '/home/laoli/rl/VisualTracking_DRL/car/tmp/1.jpg'
    test_image_1 = scm.imread(image_name_1)
    image_name_2 = '/home/laoli/rl/VisualTracking_DRL/car/tmp/2.jpg'
    # test_image_2 = scm.imread(image_name_2)


    model = init()
    detect_res_1 = model.detect(image_name_1)
    print(detect_res_1)
    print(len(detect_res_1))

    for i in range(1, len(detect_res_1)):
        x1 = int(detect_res_1[i][0])
        y1 = int(detect_res_1[i][1])
        x2 = int(detect_res_1[i][2])
        y2 = int(detect_res_1[i][3])
        cv2.rectangle(test_image_1, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # plt.imshow(test_image_1)
    # plt.axis('off')
    # plt.show()

    # detect_res_2 = model.detection(image_name_2)
    # print(detect_res_2)












