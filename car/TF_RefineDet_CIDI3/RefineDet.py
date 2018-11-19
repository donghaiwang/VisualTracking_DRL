#coding:utf-8
from test import *

import scipy.misc as scm

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
    model_path = 'D:/workspace/rl/DRLT/VisualTracking_DRL/car/TF_RefineDet_CIDI3/log/save_cidi'
    model.build_model()
    model.load_model(model_path)
    return model
    # detectronRes = model.detection(image_path)
    # return detectronRes


if __name__ == "__main__":
    # detect image
    image_name_1 = 'test/20180505102607869_000001.jpg'
    test_image_1 = scm.imread(image_name_1)

    model = init()
    detect_res_1 = model.detect(image_name_1)
    print(detect_res_1)
    print(len(detect_res_1))

    for i in range(1, len(detect_res_1)):
        x1 = int(detect_res_1[i][0])
        y1 = int(detect_res_1[i][1])
        x2 = int(detect_res_1[i][2])
        y2 = int(detect_res_1[i][3])
    #     cv2.rectangle(test_image_1, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #
    # img_src = cv2.cvtColor(test_image_1, cv2.COLOR_BGR2RGB)
    # cv2.imshow('result', img_src)
    # cv2.waitKey()


    # detect video
    # cv2.namedWindow("Detect vehicle")
    # videoPath = '/home/laoli/rl/VisualTracking_DRL/car/tmp/cidi.wmv'
    # cap = cv2.VideoCapture(videoPath)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     model = init()
    #     cv2.imwrite('/tmp/buffer.jpg', frame);
    #     detect_res_1 = model.detect('/tmp/buffer.jpg')
    #     print(detect_res_1)
    #     print(len(detect_res_1))
    #
    #     for i in range(1, len(detect_res_1)):
    #         x1 = int(detect_res_1[i][0])
    #         y1 = int(detect_res_1[i][1])
    #         x2 = int(detect_res_1[i][2])
    #         y2 = int(detect_res_1[i][3])
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #
    #     img_src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     cv2.imshow('image', img_src)
    #     cv2.waitKey(1)
    #     # k = cv2.waitKey()
    #     # if k & 0xff == ord('q'):
    #     #     break
    # cap.release()
    # cv2.destroyAllWindows()













