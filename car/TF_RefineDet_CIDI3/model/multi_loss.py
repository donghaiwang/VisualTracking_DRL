from model.candidate_box_process import *

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
        arm_cross_pos = []
        arm_cross_neg = []
        arm_loc = []
        odm_cross_pos = []
        odm_cross_neg = []
        odm_loc = []

        # refine_idxeses = []

        for i in range(len(arm_logits)):
            with tf.name_scope('arm_block_%i' % i):
                arm_conf_shape = gscores[i].shape.as_list()+[2]
                arm_loc_shape = gscores[i].shape.as_list()+[4]
                arm_logits[i] = tf.reshape(arm_logits[i],arm_conf_shape)
                arm_localisations[i] = tf.reshape(arm_localisations[i],arm_loc_shape)
                pmask = gscores[i]>match_threshold
                fpmask = tf.cast(pmask,tf.float32)
                n_positives = tf.reduce_sum(fpmask)

                no_classes = tf.cast(pmask, tf.int32)
                predictions = tf.nn.softmax(arm_logits[i])
                # predictions = tf.reshape(predictions,arm_conf_shape)


                nmask = tf.logical_and(tf.logical_not(pmask),gscores[i] > -0.5)
                fnmask = tf.cast(nmask, tf.float32)


                nvalues = tf.where(nmask,predictions[:, :, :, :, 0],1. - fnmask)
                nvalues_flat = tf.reshape(nvalues, [-1])

                # n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                # n_neg = tf.maximum(n_neg, tf.size(nvalues_flat) // 8)
                # n_neg = tf.maximum(n_neg, tf.shape(nvalues)[0] * 4)
                # max_neg_entries = 1 + tf.cast(tf.reduce_sum(fnmask), tf.int32)
                # n_neg = tf.minimum(n_neg, max_neg_entries)
                #
                # val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                # minval = val[-1]

                max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)
                n_neg = tf.maximum(n_neg, 1)

                val, refine_idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                max_hard_pred = -val[-1]
                # Final negative mask.
                nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
                fnmask = tf.cast(nmask, tf.float32)

                # Add cross-entropy loss.
                with tf.name_scope('arm_cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arm_logits[i],
                                                                          labels=tf.cast(gclasses[i] > 0, tf.int64))
                    loss = tf.losses.compute_weighted_loss(loss, fpmask)

                    # loss = tf.cond(tf.greater(n_positives, 0), lambda: tf.div(tf.reduce_sum(loss * fpmask),
                    #             tf.cast(n_positives,tf.float32), name='value'), lambda: tf.div(tf.reduce_sum(loss * fpmask),
                    #                               tf.cast(n_positives,tf.float32)+1), name='value')
                    arm_cross_pos.append(loss)

                with tf.name_scope('arm_cross_entropy_neg'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arm_logits[i],
                                                                          labels=no_classes)
                    loss = tf.losses.compute_weighted_loss(loss, fnmask)
                    # loss = tf.cond(tf.greater(n_neg, 0), lambda: tf.div(tf.reduce_sum(loss * fnmask),
                    #            tf.cast(n_neg, tf.float32),name='value'),lambda: tf.div(tf.reduce_sum(loss * fnmask),
                    #                               tf.cast(n_neg, tf.float32) + 1), name='value')
                    arm_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('arm_localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    loss = abs_smooth(arm_localisations[i] - glocalisations[i])
                    loss = tf.losses.compute_weighted_loss(loss, weights)

                    # loss = tf.cond(tf.greater(n_positives, 0), lambda: tf.div(tf.reduce_sum(loss * tf.expand_dims(fpmask,-1)),
                    #                                 tf.cast(n_positives, tf.float32),name='value'),lambda:
                    #                                 tf.div(tf.reduce_sum(loss * tf.expand_dims(fpmask,-1)),
                    #                                 tf.cast(n_positives, tf.float32) + 1), name='value')
                    arm_loc.append(loss)

#################################################### ODM decode && encode################################################
            with tf.name_scope('odm_block_%i' % i):
                odm_conf_shape = gscores[i].shape.as_list()+[num_classes]

                arm_loc_decode = tf_bboxes_decode_layer(arm_localisations[i],anchors[i])

                refine_gclass = []
                refine_glocalisations=[]
                refine_gscores=[]

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

                odm_logits[i] = tf.reshape(odm_logits[i], odm_conf_shape)
                refine_predictions = tf.nn.softmax(odm_logits[i])
                odm_localisations[i] = tf.reshape(odm_localisations[i], arm_loc_shape)
                refine_pmask = refine_gscores > match_threshold
                refine_fpmask = tf.cast(refine_pmask, tf.float32)

                # refine_pvalues = tf.where(refine_pmask, refine_predictions[:, :, :, :, 1], 1. - refine_fpmask)
                # refine_pvalues_flat = tf.reshape(refine_pvalues, [-1])
                # refine_pval, refine_pidxes = tf.nn.top_k(refine_pvalues_flat, k=128)
                # p_max_hard_pred = refine_pval[-1]
                # Final negative mask.
                # refine_pmask_ = tf.logical_and(refine_pmask, refine_pvalues > p_max_hard_pred)
                # refine_fpmask_ = tf.cast(refine_pmask_, tf.float32)

                refine_n_positives = tf.reduce_sum(refine_fpmask)
                refine_no_classes = tf.cast(refine_pmask, tf.int32)

                # predictions = tf.reshape(predictions,arm_conf_shape)
                refine_nmask = tf.logical_and(tf.logical_not(refine_pmask), refine_gscores > -0.5)

                refine_nmask = tf.logical_and(refine_nmask, predictions[:, :, :, :, 0] <= 0.99)

                refine_fnmask = tf.cast(refine_nmask, tf.float32)
                refine_nvalues = tf.where(refine_nmask, refine_predictions[:, :, :, :, 0], 1. - refine_fnmask)
                refine_nvalues_flat = tf.reshape(refine_nvalues, [-1])

                refine_max_neg_entries = tf.cast(tf.reduce_sum(refine_fnmask), tf.int32)
                refine_n_neg = tf.cast(negative_ratio * refine_n_positives, tf.int32)
                refine_n_neg = tf.minimum(refine_n_neg, refine_max_neg_entries)
                refine_n_neg = tf.maximum(refine_n_neg, 1)

                refine_val, refine_idxes = tf.nn.top_k(-refine_nvalues_flat, k=refine_n_neg)

                refine_logtis = tf.gather(tf.reshape(odm_logits[i], [-1,num_classes]), refine_idxes)
                # refine_logtis2 = tf.gather(tf.reshape(odm_logits[i][:, :, :, 1], [-1]), refine_idxes)
                # refine_logtis = tf.stack([refine_logtis1,refine_logtis2],axis=-1)
                refine_no_class = tf.gather(tf.reshape(refine_no_classes,[-1]),refine_idxes)

                # max_hard_pred = -refine_val[-1]
                # # Final negative mask.
                # refine_nmask = tf.logical_and(refine_nmask, refine_nvalues < max_hard_pred)
                # all_refine_nmask.append(refine_nmask) #################################
                # refine_fnmask = tf.cast(refine_nmask, tf.float32)

                # Add cross-entropy loss.
                with tf.name_scope('odm_cross_entropy_pos'):
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=odm_logits[i],
                                                                          labels=refine_gclass)
                    loss = tf.losses.compute_weighted_loss(loss, refine_pmask)
                    # loss = tf.cond(tf.greater(refine_n_positives, 0), lambda: tf.div(tf.reduce_sum(loss * refine_fpmask),
                    #            tf.cast(refine_n_positives, tf.float32),name='value'),lambda: tf.div(tf.reduce_sum(loss * refine_fpmask),
                    #                               tf.cast(refine_n_positives, tf.float32) + 1), name='value')
                    odm_cross_pos.append(loss)

                with tf.name_scope('odm_cross_entropy_neg'):
                    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=odm_logits[i],
                    #                                                       labels=refine_no_classes)
                    # loss = tf.losses.compute_weighted_loss(loss, refine_fnmask)
                    # loss = tf.cond(tf.greater(refine_n_neg, 0), lambda: tf.div(tf.reduce_sum(loss * refine_fnmask),
                    #            tf.cast(refine_n_neg, tf.float32),name='value'),lambda: tf.div(tf.reduce_sum(loss * refine_fnmask),
                    #                               tf.cast(refine_n_neg, tf.float32) + 1), name='value')

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=refine_logtis,
                                                                          labels=refine_no_class)
                    loss = tf.div(tf.reduce_sum(loss*refine_fnmask),tf.cast(refine_n_neg,tf.float32))

                    odm_cross_neg.append(loss)

                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('odm_localization'):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * tf.cast(refine_pmask,tf.float32), axis=-1)
                    loss = abs_smooth(odm_localisations[i] - refine_glocalisations)
                    loss = tf.losses.compute_weighted_loss(loss, weights)
                    # loss = tf.cond(tf.greater(refine_n_positives, 0), lambda: tf.div(tf.reduce_sum(loss * tf.expand_dims(refine_fpmask, -1)),
                    #            tf.cast(refine_n_positives, tf.float32),name='value'),lambda: tf.div(tf.reduce_sum(loss * tf.expand_dims(refine_fpmask, -1)),
                    #                               tf.cast(refine_n_positives, tf.float32) + 1), name='value')
                    odm_loc.append(loss)

        # Additional total losses...
        with tf.name_scope('arm_total'):
            arm_total_cross_pos = tf.add_n(arm_cross_pos, 'arm_cross_entropy_pos')
            arm_total_cross_neg = tf.add_n(arm_cross_neg, 'arm_cross_entropy_neg')
            arm_total_cross = tf.add(arm_total_cross_pos, arm_total_cross_neg, 'arm_cross_entropy')
            arm_total_loc = tf.add_n(arm_loc, 'arm_localization')
            arm_total_loss = tf.add(arm_total_cross,arm_total_loc,'arm_total_loss')

        with tf.name_scope('odm_total'):
            odm_total_cross_pos = tf.add_n(odm_cross_pos, 'odm_cross_entropy_pos')
            odm_total_cross_neg = tf.add_n(odm_cross_neg, 'odm_cross_entropy_neg')
            odm_total_cross = tf.add(odm_total_cross_pos, odm_total_cross_neg, 'odm_cross_entropy')
            odm_total_loc = tf.add_n(odm_loc, 'odm_localization')
            odm_total_loss = tf.add(odm_total_cross, odm_total_loc, 'odm_total_loss')

        with tf.name_scope('total_loss'):
            total_loss = tf.add(arm_total_loss, odm_total_loss, 'total_loss')


        return total_loss,arm_total_loss,odm_total_loss


