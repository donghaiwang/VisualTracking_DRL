from model.candidate_box_process import *
from tensorflow.python.ops import array_ops

def abs_smooth(x):

    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def get_shape(x, rank=None):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):

    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)


def RefineDet_losses(arm_logits, arm_localisations,
                       odm_logits, odm_localisations,
                       gclasses, glocalisations,
                       gscores,batch_size,num_classes,
                       anchors,bbox_label,cls_label,
                       match_threshold=0.5,
                       negative_ratio=3.,
                       alpha=1.,
                       label_smoothing=0.,
                       device='/cpu:0',
                       scope=None):
    with tf.name_scope(scope,'RefineDet_losses'):
        with tf.name_scope('arm_block'):
            arm_logits = [tf.reshape(logit,gscores[i].shape.as_list() + [2]) for i,logit in enumerate(arm_logits)]
            arm_localisations = [tf.reshape(location, gscores[i].shape.as_list() + [4]) for i, location in enumerate(arm_localisations)]
            arm_flogits = tf.concat([tf.reshape(logits,[-1,2]) for logits in arm_logits],axis=0)
            arm_flocalisations = tf.concat([tf.reshape(location,[-1,4]) for location in arm_localisations],axis=0)
            arm_fgclasses = tf.concat([tf.reshape(gclass, [-1]) for gclass in gclasses],axis=0)
            arm_fgscores = tf.concat([tf.reshape(gscore, [-1]) for gscore in gscores], axis=0)
            arm_fglocalisations = tf.concat([tf.reshape(glocalisation, [-1, 4]) for glocalisation in glocalisations], axis=0)

            arm_pmask = arm_fgscores > match_threshold
            arm_fpmask = tf.cast(arm_pmask,tf.float32)
            arm_NumPositives = tf.reduce_sum(arm_fpmask)
            arm_NoClasses = tf.cast(arm_pmask,tf.int32)

            arm_predictions = tf.nn.softmax(arm_flogits)
            arm_nmask = tf.logical_and(tf.logical_not(arm_pmask),arm_fgscores > -0.5)
            arm_fnmask = tf.cast(arm_nmask, tf.float32)

            arm_nvalues = tf.where(arm_nmask,arm_predictions[:, 0],1. - arm_fnmask)
            arm_nvalues_flat = tf.reshape(arm_nvalues, [-1])

            arm_max_neg_entries = tf.cast(tf.reduce_sum(arm_fnmask), tf.int32)
            arm_NumNeg = tf.cast(negative_ratio * arm_NumPositives, tf.int32)
            arm_NumNeg = tf.minimum(arm_NumNeg, arm_max_neg_entries)
            arm_NumNeg = tf.maximum(arm_NumNeg, 1)

            arm_val, arm_idxes = tf.nn.top_k(-arm_nvalues_flat, k=arm_NumNeg)
            arm_max_hard_pred = -arm_val[-1]
            # Final negative mask.
            arm_nmask = tf.logical_and(arm_nmask, arm_nvalues < arm_max_hard_pred)
            arm_fnmask = tf.cast(arm_nmask, tf.float32)

            with tf.name_scope('arm_cross_entropy_pos'):
                arm_cross_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arm_flogits,
                                                                          labels=tf.cast(arm_fgclasses > 0, tf.int64))
                arm_cross_pos = tf.losses.compute_weighted_loss(arm_cross_pos, arm_fpmask)

            with tf.name_scope('arm_cross_entropy_neg'):
                arm_cross_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arm_flogits,
                                                                            labels=arm_NoClasses)
                arm_cross_neg = tf.losses.compute_weighted_loss(arm_cross_neg, arm_fnmask)

            with tf.name_scope('arm_localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * arm_fpmask, axis=-1)
                arm_loc = abs_smooth(arm_flocalisations - arm_fglocalisations)
                arm_loc = tf.losses.compute_weighted_loss(arm_loc, weights)

#################################################### ODM decode && encode################################################
        with tf.name_scope('odm_block'):
            odm_fglocalisations = []
            odm_fgclass = []
            odm_fgscores = []
            for i in range(len(gclasses)):
                refine_gclass = []
                refine_glocalisations=[]
                refine_gscores=[]

                odm_conf_shape = gscores[i].shape.as_list()+[num_classes]
                odm_loc_shape = gscores[i].shape.as_list() + [4]
                arm_loc_decode = tf_bboxes_decode_layer(arm_localisations[i],anchors[i])
                cls_split = tf.split(cls_label, batch_size, axis=0)
                bbox_split = tf.split(bbox_label, batch_size, axis=0)
                arm_loc_decode_split = tf.split(arm_loc_decode,batch_size,axis=0)

                for b in range(batch_size):
                    cls = tf.squeeze(cls_split[b],axis=0,name='cls_batch_'+str(b))
                    bbox = tf.squeeze(bbox_split[b],axis=0,name='bbox_batch_'+str(b))
                    arm_decode = tf.squeeze(arm_loc_decode_split[b],axis=0,name='arm_batch_'+str(b))
                    box_num = tf.reduce_sum(tf.cast(cls > 0, tf.int64, 'sum'+str(b)))
                    gclass_, glocalisation_, gscore_ = bboxes_encode_layer(cls[:box_num],
                                                                           bbox[:box_num],
                                                                           arm_decode,
                                                                           num_classes,
                                                                           is_refine=True)
                    refine_gclass.append(gclass_)
                    refine_glocalisations.append(glocalisation_)
                    refine_gscores.append(gscore_)

                refine_gclass = tf.stack(refine_gclass,axis=0)
                refine_glocalisations = tf.stack(refine_glocalisations, axis=0)
                refine_gscores = tf.stack(refine_gscores, axis=0)
                odm_fgclass.append(refine_gclass)
                odm_fglocalisations.append(refine_glocalisations)
                odm_fgscores.append(refine_gscores)

                odm_logits[i] = tf.reshape(odm_logits[i], odm_conf_shape)
                odm_localisations[i] = tf.reshape(odm_localisations[i], odm_loc_shape)

            odm_flogits = tf.concat([tf.reshape(logits,[-1,2]) for logits in odm_logits],axis=0)
            odm_flocalisations = tf.concat([tf.reshape(location,[-1,4]) for location in odm_localisations],axis=0)
            odm_fgclasses = tf.concat([tf.reshape(gclass, [-1]) for gclass in odm_fgclass],axis=0)
            odm_fgscores = tf.concat([tf.reshape(gscore, [-1]) for gscore in odm_fgscores], axis=0)
            odm_fglocalisations = tf.concat([tf.reshape(glocalisation, [-1, 4]) for glocalisation in odm_fglocalisations], axis=0)



            odm_pmask = odm_fgscores > match_threshold
            odm_pmask = tf.logical_and(odm_pmask, arm_predictions[:, 0] <= 0.99)
            odm_fpmask = tf.cast(odm_pmask, tf.float32)
            odm_NumPositives = tf.reduce_sum(odm_fpmask)
            odm_NoClasses = tf.cast(odm_pmask, tf.int32)

            odm_predictions = tf.nn.softmax(odm_flogits)
            odm_nmask = tf.logical_and(tf.logical_not(odm_pmask), odm_fgscores > -0.5)
            # odm_nmask = tf.logical_and(odm_nmask, arm_predictions[:, 0] <= 0.99)
            odm_fnmask = tf.cast(odm_nmask, tf.float32)

            odm_nvalues = tf.where(odm_nmask, odm_predictions[:, 0], 1. - odm_fnmask)
            odm_nvalues_flat = tf.reshape(odm_nvalues, [-1])

            odm_max_neg_entries = tf.cast(tf.reduce_sum(odm_fnmask), tf.int32)
            odm_NumNeg = tf.cast(negative_ratio * odm_NumPositives, tf.int32)
            odm_NumNeg = tf.minimum(odm_NumNeg, odm_max_neg_entries)
            odm_NumNeg = tf.maximum(odm_NumNeg, 1)

            odm_val, odm_idxes = tf.nn.top_k(-odm_nvalues_flat, k=odm_NumNeg)
            odm_max_hard_pred = -odm_val[-1]
            # Final negative mask.
            odm_nmask = tf.logical_and(odm_nmask, odm_nvalues < odm_max_hard_pred)
            odm_fnmask = tf.cast(odm_nmask, tf.float32)

            # Add cross-entropy loss.
            with tf.name_scope('odm_cross_entropy'):
                odm_total_cross = focal_loss(odm_flogits, tf.one_hot(odm_fgclasses,depth=2))
            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('odm_localization'):
                # Weights Tensor: positive mask + random negative.
                # weights = tf.expand_dims(alpha * tf.cast(odm_pmask,tf.float32), axis=-1)
                # odm_loc = abs_smooth(odm_flocalisations - odm_fglocalisations)
                # odm_loc = tf.losses.compute_weighted_loss(odm_loc, weights)

                odm_loc = tf.reduce_sum(abs_smooth(odm_flocalisations - odm_fglocalisations))
                odm_loc = tf.div(odm_loc, tf.reduce_sum(tf.cast(odm_fgscores > 0.,tf.float32)))



        # Additional total losses...
        with tf.name_scope('arm_total'):
            arm_total_cross = tf.add(arm_cross_pos, arm_cross_neg, 'arm_cross_entropy')
            arm_total_loss = tf.add(arm_total_cross,arm_loc,'arm_total_loss')

        with tf.name_scope('odm_total'):
            odm_total_loss = tf.add(odm_total_cross, odm_loc, 'odm_total_loss')

        with tf.name_scope('total_loss'):
            total_loss = tf.add(arm_total_loss, odm_total_loss, 'total_loss')

        return total_loss,arm_total_loss,odm_total_loss


