# -*- coding: UTF-8 -*-
"""
vgg(base network)->RefineDet_tf for vechile detection.
@author: xie wei
"""
from model.RefineNet import RefineDet_Model
from model.anchors_layer import *
from model.layers_group import *
from model.multi_loss3 import *
from data.read_tfrecord_no import train_input_fn

class Moedel():
    def __init__(self,baseNet='VGG',batch_size = 2,img_size=[512,512,3],learn_rate=0.001,decay = 0.96,
                 decay_step = 200, training = True,keep_prob=0.5,w_summary = True,num_of_classes=5,
                 savepath='log/save',use_premodel=True,pretrain_modelpath='convert_model/save/model.ckpt',
                 name = 'RefineDet'):
        self.baseNet=baseNet
        self.data_dir = '/home/xw/workspace/deepcode/Tf_RefineDet_cidi/data/tfrecord'
        self.batchSize = batch_size
        self.img_size = img_size
        self.training = training
        self.w_summary = w_summary
        self.learning_rate = learn_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.num_of_classes = num_of_classes
        self.thresh = 0.5
        self.keep_prob = keep_prob
        self.savepath = savepath
        self.IMG_MEAN = np.array((71, 75, 74), dtype=np.float32)
        self.weight_decay = 0.0005
        self.pretrain_modelpath = pretrain_modelpath
        self.use_pretrained=use_premodel
        self.is_val = False
        self.GPU_GROUPS = ["/gpu:0"]
        self.checkpoint_dir=savepath
        self.epoch=100000
        self.log_dir = 'log'
        self.model_name='RefineDet'
        self.save_step=1000
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
        if(self.training):

            # load reader
            self.image_batch, self.glocalisations, self.gclasses, self.gscores, self.feat_maxMask, self.bbox, self.cls,self.name_batch= \
                                                train_input_fn('data/tfrecord',self.batchSize,
                                              self.anchors_layers,self.num_of_classes,self.img_size)
            self.coord = tf.train.Coordinator()

            self.RefineDet = RefineDet_Model(self.anchor_sizes,self.anchor_ratios,
                                             self.anchor_steps,
                                             img_size=self.img_size,
                                             det_num_class=self.num_of_classes,
                                             training=self.training,
                                             keep_prob=self.keep_prob,
                                             name='RefineDet',
                                             reuse=False)
            self.output = self.RefineDet.buildModel(self.image_batch)

            with tf.device(self.cpu):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)
                # self.train_step = tf.Variable(0, trainable=False)
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step,
                                                             self.decay, staircase=True, name='learning_rate')
            with tf.name_scope('loss') as scope:
                l2_losses = [self.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                             'weights' in v.name]

                self.loss,self.arm_loss,self.odm_loss,self.arm_pmask,self.odm_pmask,self.arm_decode = \
                    RefineDet_losses(self.output['arm'][0], self.output['arm'][1],
                                             self.output['odm'][0], self.output['odm'][1],
                                             self.gclasses, self.glocalisations, self.gscores,self.feat_maxMask,
                                             self.batchSize,self.num_of_classes,self.anchors_layers,
                                             self.bbox, self.cls,
                                             match_threshold=0.5,negative_ratio=3.,scope=scope)
                self.loss = self.loss + tf.add_n(l2_losses)

            self.all_trainable =  [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
            self.VGG_trainable = [v for v in self.all_trainable if 'VGG' in v.name]
            self.Extra_trainable = [v for v in self.all_trainable if 'VGG' not in v.name]
            assert (len(self.all_trainable) == len(self.VGG_trainable) + len(self.Extra_trainable))

            with tf.name_scope('opt'):
                # self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
                self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.9)
                # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

                self.opt_vgg = tf.train.MomentumOptimizer(learning_rate=self.lr*1e-1,momentum=0.9)
                self.opt_extra = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.9)

            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(self.update_ops):
                # grads = self.opt.compute_gradients(self.loss)
                # for i, (g, v) in enumerate(grads):
                #     if g is not None:
                #         grads[i] = (tf.clip_by_norm(g, 5.0), v)  # clip gradients
                # self.train_op = self.opt.apply_gradients(grads, self.train_step)

                grads = tf.gradients(self.loss,self.VGG_trainable + self.Extra_trainable)
                grads_vgg = grads[:len(self.VGG_trainable)]
                grads_extra = grads[len(self.VGG_trainable):]

                self.train_op_vgg = self.opt_vgg.apply_gradients(zip(grads_vgg, self.VGG_trainable),
                                                                   global_step=self.train_step)
                self.train_op_extra= self.opt_vgg.apply_gradients(zip(grads_extra, self.Extra_trainable),
                                                                 global_step=self.train_step)
                self.train_op = tf.group(self.train_op_vgg,self.train_op_extra)

                Refine_loss_summ = tf.summary.scalar('total_loss', self.loss)
                arm_loss_summ = tf.summary.scalar('arm_loss', self.arm_loss)
                odm_loss_summ = tf.summary.scalar('odm_loss', self.odm_loss)
                l2_loss_summ = tf.summary.scalar('l2_loss',tf.add_n(l2_losses))
                lr_summ = tf.summary.scalar('learning_rate', self.lr)
            self.summary_sum = tf.summary.merge([Refine_loss_summ, arm_loss_summ,odm_loss_summ,l2_loss_summ,lr_summ])

    def train(self):
        self.init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.Session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.Session.run(self.init)
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.Session, ckpt.model_checkpoint_path)
            print('Loading Trained Model succed ...')
        else:
            if self.use_pretrained:
                self.pre_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                                      and 'refineDet' not in v.name]
                load_saver = tf.train.Saver(var_list=self.pre_trainable)
                load_saver.restore(self.Session, self.pretrain_modelpath)
            print('Please give a Model in args ...')

        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.Session)

        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.Session.graph)
        for epoch in range(self.epoch):
            _, lr,summary_write,loss,arm_loss,odm_loss,image_batch,arm_pmask,odm_pmask,arm_decode,name_batch = self.Session.run([self.train_op,
                                                                           self.lr,self.summary_sum,self.loss,
                                                                                self.arm_loss,self.odm_loss,self.image_batch,
                                                                           self.arm_pmask, self.odm_pmask,
                                                                           self.arm_decode,self.name_batch])
            self.writer.add_summary(summary_write, epoch)
            if(epoch % self.save_step == 0):
                # save model
                self.saver.save(self.Session, os.path.join(self.checkpoint_dir, str('RefineDet' + '_' + str(epoch + 1))))
            print('-epoch: '+str(epoch)+' -lr: '+str(lr)+' -loss: ' + str(loss)+' -arm_loss: ' + str(arm_loss)+' -odm_loss: ' + str(odm_loss))



        self.coord.request_stop()
        self.coord.join(threads)























