from model.candidate_box_process import *

def abs_smooth(x):

    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

def RefineDet_losses(arm_logits, arm_localisations,
                       odm_logits, odm_localisations,
                       gclasses, glocalisations,
                       gscores,feat_maxMask,
                       batch_size,num_classes,
                       anchors,bbox_label,cls_label,
                       match_threshold=0.5,
                       negative_ratio=3.,
                       alpha=1.,
                       label_smoothing=0.,
                       device='/cpu:0',
                       Threshold_change_factor=0.,
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
            arm_pmask_max = tf.concat([tf.reshape(mask, [-1]) for mask in feat_maxMask], axis=0)

            arm_pmask = arm_fgscores > match_threshold
            arm_pmasked = [arm > match_threshold for arm in gscores]
            arm_pmask = tf.logical_or(arm_pmask,arm_pmask_max)
            arm_fpmask = tf.cast(arm_pmask,tf.float32)
            arm_NumPositives = tf.reduce_sum(arm_fpmask)
            arm_NoClasses = tf.zeros_like(arm_pmask, dtype=tf.int32)

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

            arm_pmask_index = tf.squeeze(tf.where(arm_fpmask > 0),axis=-1)
            arm_nmask_index = tf.squeeze(tf.where(arm_fnmask > 0),axis=-1)

            with tf.name_scope('arm_cross_entropy'):
                arm_flogits_pos = tf.gather(arm_flogits,arm_pmask_index)
                arm_fgclasses_pos = tf.gather(tf.cast(arm_fgclasses > 0, tf.int32),arm_pmask_index)
                arm_flogits_neg = tf.gather(arm_flogits,arm_nmask_index)
                arm_fgclasses_neg = tf.gather(arm_NoClasses,arm_nmask_index)
                arm_flogits = tf.concat([arm_flogits_pos,arm_flogits_neg],axis=0)
                arm_fgclasses = tf.concat([arm_fgclasses_pos,arm_fgclasses_neg],axis=0)

                arm_cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arm_flogits,
                                                                          labels=arm_fgclasses)

                arm_cross = tf.cond(tf.greater(arm_NumPositives, 0), lambda:
                                    tf.div(tf.reduce_sum(arm_cross), arm_NumPositives),
                                    lambda: tf.constant(0.))

            with tf.name_scope('arm_localization'):
                arm_flocalisations = tf.gather(arm_flocalisations,arm_pmask_index)
                arm_fglocalisations = tf.gather(arm_fglocalisations, arm_pmask_index)
                arm_loc = abs_smooth(arm_flocalisations - arm_fglocalisations)
                arm_loc = tf.cond(tf.greater(arm_NumPositives, 0), lambda:
                                  tf.div(tf.reduce_sum(arm_loc), arm_NumPositives),
                                  lambda: tf.constant(0.))

#################################################### ODM decode && encode################################################
        with tf.name_scope('odm_block'):
            odm_fglocalisations = []
            odm_fgclass = []
            odm_fgscores = []
            odm_maxMask = []
            arm_loc_decodes = []
            odm_conf_shape = []
            odm_loc_shape = []

            for i in range(len(gclasses)):
                odm_conf_shape.append(gscores[i].shape.as_list()+[num_classes])
                odm_loc_shape.append(gscores[i].shape.as_list() + [4])
                arm_loc_decode = tf_bboxes_decode_layer(arm_localisations[i],anchors[i])
                arm_loc_decodes.append(arm_loc_decode)

            cls_split = tf.split(cls_label, batch_size, axis=0)
            bbox_split = tf.split(bbox_label, batch_size, axis=0)
            for b in range(batch_size):
                cls = tf.squeeze(cls_split[b],axis=0,name='cls_batch_'+str(b))
                bbox = tf.squeeze(bbox_split[b],axis=0,name='bbox_batch_'+str(b))
                arm_decode = [de[b] for de in arm_loc_decodes]
                box_num = tf.reduce_sum(tf.cast(cls > 0, tf.int64, 'sum'+str(b)))
                gclass_, glocalisation_, gscore_, maxMask_ = bboxes_encode(cls[:box_num],
                                                                       bbox[:box_num],
                                                                       arm_decode,
                                                                       num_classes,
                                                                       is_refine=True)
                if(b==0):
                    odm_fgclass = [tf.expand_dims(g,axis=0) for g in gclass_]
                    odm_fglocalisations = [tf.expand_dims(g, axis=0) for g in glocalisation_]
                    odm_fgscores = [tf.expand_dims(g, axis=0) for g in gscore_]
                    odm_maxMask = [tf.expand_dims(g, axis=0) for g in maxMask_]
                else:
                    odm_fgclass = [tf.concat([o,tf.expand_dims(g,axis=0)],axis=0) for o,g in zip(odm_fgclass,gclass_)]
                    odm_fglocalisations = [tf.concat([o,tf.expand_dims(g, axis=0)],axis=0) for o,g in zip(odm_fglocalisations, glocalisation_)]
                    odm_fgscores = [tf.concat([o,tf.expand_dims(g, axis=0)],axis=0) for o,g in zip(odm_fgscores,gscore_)]
                    odm_maxMask = [tf.concat([o,tf.expand_dims(g, axis=0)],axis=0) for o,g in zip(odm_maxMask,maxMask_)]

            odm_pmasked = [odm > match_threshold for odm in odm_fgscores]

            odm_logits = [tf.reshape(odm_logit, conf_shape) for odm_logit,conf_shape in zip(odm_logits,odm_conf_shape)]
            odm_localisations = [tf.reshape(odm_location, loc_shape) for odm_location,loc_shape in zip(odm_localisations,odm_loc_shape)]

            odm_flogits = tf.concat([tf.reshape(logits,[-1,num_classes]) for logits in odm_logits],axis=0)
            odm_flocalisations = tf.concat([tf.reshape(location,[-1,4]) for location in odm_localisations],axis=0)
            odm_fgclasses = tf.concat([tf.reshape(gclass, [-1,]) for gclass in odm_fgclass],axis=0)
            odm_fgscores = tf.concat([tf.reshape(gscore, [-1,]) for gscore in odm_fgscores], axis=0)
            odm_fglocalisations = tf.concat([tf.reshape(glocalisation, [-1, 4]) for glocalisation in odm_fglocalisations], axis=0)
            odm_fmaxMask = tf.concat([tf.reshape(Mask, [-1, ]) for Mask in odm_maxMask], axis=0)

            odm_pmask = odm_fgscores > (match_threshold + Threshold_change_factor)
            odm_pmask = tf.logical_or(odm_pmask,odm_fmaxMask)
            # odm_pmask = tf.logical_and(odm_pmask, arm_predictions[:, 0] <= 0.99)
            odm_fpmask = tf.cast(odm_pmask, tf.float32)
            odm_NumPositives = tf.reduce_sum(odm_fpmask)
            odm_NoClasses = tf.zeros_like(odm_pmask,dtype=tf.int32)

            odm_predictions = tf.nn.softmax(odm_flogits)
            odm_nmask = tf.logical_and(tf.logical_not(odm_pmask), odm_fgscores > -0.5)
            # odm_nmask = tf.logical_and(odm_nmask, odm_fgscores <= match_threshold-Threshold_change_factor)
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

            odm_pmask_index = tf.squeeze(tf.where(odm_fpmask > 0),axis=-1)
            odm_nmask_index = tf.squeeze(tf.where(odm_fnmask > 0),axis=-1)

            with tf.name_scope('odm_cross_entropy'):
                odm_flogits_pos = tf.gather(odm_flogits, odm_pmask_index)
                odm_fgclasses_pos = tf.gather(tf.cast(odm_fgclasses, tf.int32), odm_pmask_index)
                odm_flogits_neg = tf.gather(odm_flogits, odm_nmask_index)
                odm_fgclasses_neg = tf.gather(odm_NoClasses, odm_nmask_index)
                odm_flogits = tf.concat([odm_flogits_pos, odm_flogits_neg], axis=0)
                odm_fgclasses = tf.concat([odm_fgclasses_pos, odm_fgclasses_neg], axis=0)

                odm_cross = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=odm_flogits,
                                                                           labels=odm_fgclasses)
                odm_cross = tf.cond(tf.greater(odm_NumPositives, 0), lambda:
                                    tf.div(tf.reduce_sum(odm_cross), odm_NumPositives),
                                    lambda: tf.constant(0.))

            with tf.name_scope('odm_localization'):
                odm_flocalisations = tf.gather(odm_flocalisations, odm_pmask_index)
                odm_fglocalisations = tf.gather(odm_fglocalisations, odm_pmask_index)
                odm_loc = abs_smooth(odm_flocalisations - odm_fglocalisations)
                odm_loc = tf.cond(tf.greater(odm_NumPositives, 0), lambda:
                                  tf.div(tf.reduce_sum(odm_loc), odm_NumPositives),
                                  lambda: tf.constant(0.))

        # Additional total losses...
        with tf.name_scope('arm_total'):
            arm_total_loss = tf.add(arm_cross,arm_loc,'arm_total_loss')

        with tf.name_scope('odm_total'):
            odm_total_loss = tf.add(odm_cross, odm_loc, 'odm_total_loss')

        with tf.name_scope('total_loss'):
            total_loss = tf.add(arm_total_loss, odm_total_loss, 'total_loss')

        return total_loss,arm_total_loss,odm_total_loss,feat_maxMask,odm_maxMask,arm_loc_decodes



