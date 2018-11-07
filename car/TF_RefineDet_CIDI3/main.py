#coding:utf-8
from train import *
from test import *

# build graph
mode = 'test'

'''BDD100k_val'''
# path = '/home/cidi/ZSH/dataset/VOC_BDD100k_vehicle/VOCdevkit/VOC2007/ImageSets/Main/test_.txt'
# pathdir = '/home/cidi/ZSH/dataset/VOC_BDD100k_vehicle/VOCdevkit/VOC2007/JPEGImages/'

'''CIDI_data'''
# path = '/home/cidi/ZSH/dataset/cidi_data/test_.txt'
# pathdir = '/home/cidi/ZSH/dataset/cidi_data/image_test/'

'''演示'''
path = '/home/dong/rl/VisualTracking_DRL/car/tmp/imageList.txt'
pathdir = '/home/dong/rl/VisualTracking_DRL/car/tmp/'

# path = '/home/cidi/ZSH/result_compare/vehicle/list.txt'
# pathdir = '/home/cidi/ZSH/result_compare/vehicle/rename_004/'


imshape=[768,768,3]
batch_size = 12
learning_rate = 0.001
decay = 0.95
decay_step = 8000
random_scale = True
use_premodel = True

if mode=='test':
    Model = Test_Moedel(baseNet='VGG',batch_size = batch_size,
                   img_size=imshape,learn_rate=learning_rate,
                   decay = decay,decay_step = decay_step,
                   training = mode,keep_prob=1.0,w_summary=False,
                   num_of_classes=2,savepath='log/save',
                   use_premodel=True,pretrain_modelpath=
                   'convert_model/save/model.ckpt',name = 'RefineDet')
    lines = [pathdir+line+'.jpg' for line in open(path,'r').readlines()]
    Model.build_model()
    Model.test(lines,'log/save_bbd')

else:
    Model = Moedel(baseNet='VGG',batch_size = batch_size,
                   img_size=imshape,learn_rate=learning_rate,
                   decay = decay,decay_step = decay_step,
                   training = mode,keep_prob=0.5,w_summary=False,
                   num_of_classes=2,savepath='log/save',
                   use_premodel=use_premodel,pretrain_modelpath=
                   'convert_model/save/VGG.ckpt',name='RefineDet')
    Model.build_model()
    Model.train()












