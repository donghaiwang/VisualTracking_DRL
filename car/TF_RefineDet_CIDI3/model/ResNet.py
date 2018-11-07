# -*- coding: UTF-8 -*-
"""
SCNN(torch) for tensorflow
vgg(base network)->SCNN_UP->SCNN_DOWN for car lane seg.
@author: xie wei
"""
from model.layers_group import *
slim = tf.contrib.slim
def ResNet(inputs,name,training=True,w_summary=True,keep_prob = 1.0,reuse = False):
	with tf.variable_scope(name,reuse=reuse):
		end_points_collection = name + '_end_logits'
		conv1 = conv_bn_relu(inputs, 64, 7, 2,'SAME',training,w_summary,'conv1')
		pool1 = pool(conv1,3, 2, 'max', name='pool1')

		res2a_branch1 = conv_bn(pool1, 256, 1, 1,'SAME',training,w_summary,'res2a_branch1')

		res2a_branch2a = conv_bn_relu(pool1,64,1,1,'SAME',training,w_summary,'res2a_branch2a')
		res2a_branch2b = conv_bn_relu(res2a_branch2a, 64, 3, 1, 'SAME', training, w_summary, 'res2a_branch2b')
		res2a_branch2c = conv_bn(res2a_branch2b, 256, 1, 1, 'SAME', training, w_summary, 'res2a_branch2c')
		res2a = add_relu(res2a_branch1,res2a_branch2c,name='res2a')

		res2b_branch2a = conv_bn_relu(res2a,64,1,1,'SAME',training,w_summary,'res2b_branch2a')
		res2b_branch2b = conv_bn_relu(res2b_branch2a, 64, 3, 1, 'SAME', training, w_summary, 'res2b_branch2b')
		res2b_branch2c = conv_bn(res2b_branch2b, 256, 1, 1, 'SAME', training, w_summary, 'res2b_branch2c')
		res2b = add_relu(res2a,res2b_branch2c,name='res2b')

		res2c_branch2a = conv_bn_relu(res2b,64,1,1,'SAME',training,w_summary,'res2c_branch2a')
		res2c_branch2b = conv_bn_relu(res2c_branch2a, 64, 3, 1, 'SAME', training, w_summary, 'res2c_branch2b')
		res2c_branch2c = conv_bn(res2c_branch2b, 256, 1, 1, 'SAME', training, w_summary, 'res2c_branch2c')
		res2c = add_relu(res2a,res2c_branch2c,name='res2c')

		res3a_branch1 = conv_bn(res2c, 512, 1, 2,'SAME',training,w_summary,'res3a_branch1')

		res3a_branch2a = conv_bn_relu(res2c,128,1,2,'SAME',training,w_summary,'res3a_branch2a')
		res3a_branch2b = conv_bn_relu(res3a_branch2a, 128, 3, 1, 'SAME', training, w_summary, 'res3a_branch2b')
		res3a_branch2c = conv_bn(res3a_branch2b, 512, 1, 1, 'SAME', training, w_summary, 'res3a_branch2c')
		res3a = add_relu(res3a_branch1,res3a_branch2c,name='res3a')

		res3b1_branch2a = conv_bn_relu(res3a,128,1,1,'SAME',training,w_summary,'res3b1_branch2a')
		res3b1_branch2b = conv_bn_relu(res3b1_branch2a, 128, 3, 1, 'SAME', training, w_summary, 'res3b1_branch2b')
		res3b1_branch2c = conv_bn(res3b1_branch2b, 512, 1, 1, 'SAME', training, w_summary, 'res3b1_branch2c')
		res3b1 = add_relu(res3a,res3b1_branch2c,name='res3b1')

		res3b2_branch2a = conv_bn_relu(res3b1,128,1,1,'SAME',training,w_summary,'res3b2_branch2a')
		res3b2_branch2b = conv_bn_relu(res3b2_branch2a, 128, 3, 1, 'SAME', training, w_summary, 'res3b2_branch2b')
		res3b2_branch2c = conv_bn(res3b2_branch2b, 512, 1, 1, 'SAME', training, w_summary, 'res3b2_branch2c')
		res3b2 = add_relu(res3b1,res3b2_branch2c,name='res3b2')

		res3b3_branch2a = conv_bn_relu(res3b2,128,1,1,'SAME',training,w_summary,'res3b3_branch2a')
		res3b3_branch2b = conv_bn_relu(res3b3_branch2a, 128, 3, 1, 'SAME', training, w_summary, 'res3b3_branch2b')
		res3b3_branch2c = conv_bn(res3b3_branch2b, 512, 1, 1, 'SAME', training, w_summary, 'res3b3_branch2c')
		res3b3 = add_relu(res3b2,res3b3_branch2c,name='res3b3')

		res4a_branch1 = conv_bn(res3b3, 1024, 1, 2, 'SAME', training, w_summary, 'res4a_branch1')

		res4a_branch2a = conv_bn_relu(res3b3, 256, 1, 2,'SAME',training,w_summary,'res4a_branch2a')
		res4a_branch2b = astrous_conv_bn_relu(res4a_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4a_branch2b')
		res4a_branch2c = conv_bn(res4a_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4a_branch2c')
		res4a = add_relu(res4a_branch1,res4a_branch2c,name='res4a')

		res4b1_branch2a = conv_bn_relu(res4a, 256, 1, 1,'SAME',training,w_summary,'res4b1_branch2a')
		res4b1_branch2b = astrous_conv_bn_relu(res4b1_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b1_branch2b')
		res4b1_branch2c = conv_bn(res4b1_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b1_branch2c')
		res4b1 = add_relu(res4a_branch1,res4b1_branch2c,name='res4b1')

		res4b2_branch2a = conv_bn_relu(res4b1, 256, 1, 1,'SAME',training,w_summary,'res4b2_branch2a')
		res4b2_branch2b = astrous_conv_bn_relu(res4b2_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b2_branch2b')
		res4b2_branch2c = conv_bn(res4b2_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b2_branch2c')
		res4b2 = add_relu(res4b1,res4b2_branch2c,name='res4b2')

		res4b3_branch2a = conv_bn_relu(res4b2, 256, 1, 1,'SAME',training,w_summary,'res4b3_branch2a')
		res4b3_branch2b = astrous_conv_bn_relu(res4b3_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b3_branch2b')
		res4b3_branch2c = conv_bn(res4b3_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b3_branch2c')
		res4b3 = add_relu(res4b2,res4b3_branch2c,name='res4b3')

		res4b4_branch2a = conv_bn_relu(res4b3, 256, 1, 1,'SAME',training,w_summary,'res4b4_branch2a')
		res4b4_branch2b = astrous_conv_bn_relu(res4b4_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b4_branch2b')
		res4b4_branch2c = conv_bn(res4b4_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b4_branch2c')
		res4b4 = add_relu(res4b3,res4b4_branch2c,name='res4b4')

		res4b5_branch2a = conv_bn_relu(res4b4, 256, 1, 1,'SAME',training,w_summary,'res4b5_branch2a')
		res4b5_branch2b = astrous_conv_bn_relu(res4b5_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b5_branch2b')
		res4b5_branch2c = conv_bn(res4b5_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b5_branch2c')
		res4b5 = add_relu(res4b4,res4b5_branch2c,name='res4b5')

		res4b6_branch2a = conv_bn_relu(res4b5, 256, 1, 1,'SAME',training,w_summary,'res4b6_branch2a')
		res4b6_branch2b = astrous_conv_bn_relu(res4b6_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b6_branch2b')
		res4b6_branch2c = conv_bn(res4b6_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b6_branch2c')
		res4b6 = add_relu(res4b5,res4b6_branch2c,name='res4b6')

		res4b7_branch2a = conv_bn_relu(res4b6, 256, 1, 1,'SAME',training,w_summary,'res4b7_branch2a')
		res4b7_branch2b = astrous_conv_bn_relu(res4b7_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b7_branch2b')
		res4b7_branch2c = conv_bn(res4b7_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b7_branch2c')
		res4b7 = add_relu(res4b6,res4b7_branch2c,name='res4b7')

		res4b8_branch2a = conv_bn_relu(res4b7, 256, 1, 1,'SAME',training,w_summary,'res4b8_branch2a')
		res4b8_branch2b = astrous_conv_bn_relu(res4b8_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b8_branch2b')
		res4b8_branch2c = conv_bn(res4b8_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b8_branch2c')
		res4b8 = add_relu(res4b7,res4b8_branch2c,name='res4b8')

		res4b9_branch2a = conv_bn_relu(res4b8, 256, 1, 1,'SAME',training,w_summary,'res4b9_branch2a')
		res4b9_branch2b = astrous_conv_bn_relu(res4b9_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b9_branch2b')
		res4b9_branch2c = conv_bn(res4b9_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b9_branch2c')
		res4b9 = add_relu(res4b8,res4b9_branch2c,name='res4b9')

		res4b10_branch2a = conv_bn_relu(res4b9, 256, 1, 1,'SAME',training,w_summary,'res4b10_branch2a')
		res4b10_branch2b = astrous_conv_bn_relu(res4b10_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b10_branch2b')
		res4b10_branch2c = conv_bn(res4b10_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b10_branch2c')
		res4b10 = add_relu(res4b9,res4b10_branch2c,name='res4b10')

		res4b11_branch2a = conv_bn_relu(res4b10, 256, 1, 1,'SAME',training,w_summary,'res4b11_branch2a')
		res4b11_branch2b = astrous_conv_bn_relu(res4b11_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b11_branch2b')
		res4b11_branch2c = conv_bn(res4b11_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b11_branch2c')
		res4b11 = add_relu(res4b10,res4b11_branch2c,name='res4b11')

		res4b12_branch2a = conv_bn_relu(res4b11, 256, 1, 1,'SAME',training,w_summary,'res4b12_branch2a')
		res4b12_branch2b = astrous_conv_bn_relu(res4b12_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b12_branch2b')
		res4b12_branch2c = conv_bn(res4b12_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b12_branch2c')
		res4b12 = add_relu(res4b11,res4b12_branch2c,name='res4b12')

		res4b13_branch2a = conv_bn_relu(res4b12, 256, 1, 1,'SAME',training,w_summary,'res4b13_branch2a')
		res4b13_branch2b = astrous_conv_bn_relu(res4b13_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b13_branch2b')
		res4b13_branch2c = conv_bn(res4b13_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b13_branch2c')
		res4b13 = add_relu(res4b12,res4b13_branch2c,name='res4b13')

		res4b14_branch2a = conv_bn_relu(res4b13, 256, 1, 1,'SAME',training,w_summary,'res4b14_branch2a')
		res4b14_branch2b = astrous_conv_bn_relu(res4b14_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b14_branch2b')
		res4b14_branch2c = conv_bn(res4b14_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b14_branch2c')
		res4b14 = add_relu(res4b13,res4b14_branch2c,name='res4b14')

		res4b15_branch2a = conv_bn_relu(res4b14, 256, 1, 1,'SAME',training,w_summary,'res4b15_branch2a')
		res4b15_branch2b = astrous_conv_bn_relu(res4b15_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b15_branch2b')
		res4b15_branch2c = conv_bn(res4b15_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b15_branch2c')
		res4b15 = add_relu(res4b14,res4b15_branch2c,name='res4b15')

		res4b16_branch2a = conv_bn_relu(res4b15, 256, 1, 1,'SAME',training,w_summary,'res4b16_branch2a')
		res4b16_branch2b = astrous_conv_bn_relu(res4b16_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b16_branch2b')
		res4b16_branch2c = conv_bn(res4b16_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b16_branch2c')
		res4b16 = add_relu(res4b15,res4b16_branch2c,name='res4b16')

		res4b17_branch2a = conv_bn_relu(res4b16, 256, 1, 1,'SAME',training,w_summary,'res4b17_branch2a')
		res4b17_branch2b = astrous_conv_bn_relu(res4b17_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b17_branch2b')
		res4b17_branch2c = conv_bn(res4b17_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b17_branch2c')
		res4b17 = add_relu(res4b16,res4b17_branch2c,name='res4b17')

		res4b18_branch2a = conv_bn_relu(res4b17, 256, 1, 1,'SAME',training,w_summary,'res4b18_branch2a')
		res4b18_branch2b = astrous_conv_bn_relu(res4b18_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b18_branch2b')
		res4b18_branch2c = conv_bn(res4b18_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b18_branch2c')
		res4b18 = add_relu(res4b17,res4b18_branch2c,name='res4b18')

		res4b19_branch2a = conv_bn_relu(res4b18, 256, 1, 1,'SAME',training,w_summary,'res4b19_branch2a')
		res4b19_branch2b = astrous_conv_bn_relu(res4b19_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b19_branch2b')
		res4b19_branch2c = conv_bn(res4b19_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b19_branch2c')
		res4b19 = add_relu(res4b18,res4b19_branch2c,name='res4b19')

		res4b20_branch2a = conv_bn_relu(res4b19, 256, 1, 1,'SAME',training,w_summary,'res4b20_branch2a')
		res4b20_branch2b = astrous_conv_bn_relu(res4b20_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b20_branch2b')
		res4b20_branch2c = conv_bn(res4b20_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b20_branch2c')
		res4b20 = add_relu(res4b19,res4b20_branch2c,name='res4b20')

		res4b21_branch2a = conv_bn_relu(res4b20, 256, 1, 1,'SAME',training,w_summary,'res4b21_branch2a')
		res4b21_branch2b = astrous_conv_bn_relu(res4b21_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b21_branch2b')
		res4b21_branch2c = conv_bn(res4b21_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b21_branch2c')
		res4b21 = add_relu(res4b20,res4b21_branch2c,name='res4b21')

		res4b22_branch2a = conv_bn_relu(res4b21, 256, 1, 1,'SAME',training,w_summary,'res4b22_branch2a')
		res4b22_branch2b = astrous_conv_bn_relu(res4b22_branch2a, 256, 3, 2, 'SAME', training, w_summary, False,'res4b22_branch2b')
		res4b22_branch2c = conv_bn(res4b22_branch2b, 1024, 1, 1, 'SAME', training, w_summary, 'res4b22_branch2c')
		res4b22 = add_relu(res4b21,res4b22_branch2c,name='res4b22')

		res5a_branch1 = conv_bn(res4b22, 2048, 1, 2, 'SAME', training, w_summary, 'res5a_branch1')

		res5a_branch2a = conv_bn_relu(res4b22, 512, 1, 2, 'SAME', training, w_summary, 'res5a_branch2a')
		res5a_branch2b = astrous_conv_bn_relu(res5a_branch2a, 512, 3, 2, 'SAME', training, w_summary, False,'res5a_branch2b')
		res5a_branch2c = conv_bn(res5a_branch2b, 2048, 1, 1, 'SAME', training, w_summary, 'res5a_branch2c')
		res5a = add_relu(res5a_branch1, res5a_branch2c, name='res4a')

		res5b_branch2a = conv_bn_relu(res5a, 512, 1, 1, 'SAME', training, w_summary, 'res5b_branch2a')
		res5b_branch2b = astrous_conv_bn_relu(res5b_branch2a, 512, 3, 2, 'SAME', training, w_summary, False,'res5b_branch2b')
		res5b_branch2c = conv_bn(res5b_branch2b, 2048, 1, 1, 'SAME', training, w_summary, 'res5b_branch2c')
		res5b = add_relu(res5a, res5b_branch2c, name='res5b')

		res5c_branch2a = conv_bn_relu(res5b, 512, 1, 1, 'SAME', training, w_summary, 'res5c_branch2a')
		res5c_branch2b = astrous_conv_bn_relu(res5c_branch2a, 512, 3, 2, 'SAME', training, w_summary, False,'res5c_branch2b')
		res5c_branch2c = conv_bn(res5c_branch2b, 2048, 1, 1, 'SAME', training, w_summary, 'res5c_branch2c')
		res5c = add_relu(res5b, res5c_branch2c, name='res5c')

		end_logits = slim.utils.convert_collection_to_dict(end_points_collection)

		end_logits['res3b3'] = res3b3
		end_logits['res4b22'] = res4b22
		end_logits['res5c'] = res5c

		return res5b,end_logits






