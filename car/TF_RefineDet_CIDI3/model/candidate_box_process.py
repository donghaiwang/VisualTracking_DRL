import tensorflow as tf
import numpy as np

def bboxes_encode(labels,
                  bboxes,
                  anchors,
                  num_classes,
                  ignore_threshold=0.5,
                  prior_scaling=[0.1,0.1,0.2,0.2],
                  is_refine=False,
                  dtype=tf.float32):
    if (is_refine):
        xmin = [anchors_layer[:, :, :, 0] for anchors_layer in anchors]
        ymin = [anchors_layer[:, :, :, 1] for anchors_layer in anchors]
        xmax = [anchors_layer[:, :, :, 2] for anchors_layer in anchors]
        ymax = [anchors_layer[:, :, :, 3] for anchors_layer in anchors]
        href = [max - min for max, min in zip(ymax,ymin)]
        wref = [max - min for max, min in zip(xmax,xmin)]
        yref = [y + h / 2. for y, h in zip(ymin, href)]
        xref = [x + w / 2. for x, w in zip(xmin, wref)]
        shape = [(x.shape.as_list()[0], x.shape.as_list()[1], x.shape.as_list()[2]) for x in xref]
    else:
        xref = [anchors_layer[0] for anchors_layer in anchors]
        yref = [anchors_layer[1] for anchors_layer in anchors]
        wref = [anchors_layer[2] for anchors_layer in anchors]
        href = [anchors_layer[3] for anchors_layer in anchors]
        ymin = [yr - hr / 2. for yr, hr in zip(yref, href)]
        xmin = [xr - wr / 2. for xr, wr in zip(xref, wref)]
        ymax = [yr + hr / 2. for yr, hr in zip(yref, href)]
        xmax = [xr + wr / 2. for xr, wr in zip(xref, wref)]
        shape = [(x.shape[0], x.shape[1], href[x1].size) for x1, x in enumerate(xref)]

    vol_anchors = [(x_max - x_min) * (y_max - y_min) for x_max, x_min, y_max, y_min in zip(xmax, xmin, ymax, ymin)]

    feat_labels = [tf.zeros(s, dtype=tf.int32) for s in shape]
    feat_scores = [tf.zeros(s, dtype=dtype) for s in shape]

    feat_ymin = [tf.zeros(s, dtype=dtype) for s in shape]
    feat_xmin = [tf.zeros(s, dtype=dtype) for s in shape]
    feat_ymax = [tf.ones(s, dtype=dtype) for s in shape]
    feat_xmax = [tf.ones(s, dtype=dtype) for s in shape]

    feat_maxMask = [tf.zeros(s, dtype=tf.bool) for s in shape]

    def jaccard_with_anchors(bbox, layer):
        """Compute jaccard score between a box and the anchors.
        """
        int_xmin = tf.maximum(xmin[layer], bbox[0])
        int_ymin = tf.maximum(ymin[layer], bbox[1])
        int_xmax = tf.minimum(xmax[layer], bbox[2])
        int_ymax = tf.minimum(ymax[layer], bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors[layer] - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax, feat_maxMask):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_xmin, feat_ymin, feat_xmax, feat_ymax, feat_maxMask):

        label = labels[i]
        bbox = bboxes[i]
        jaccard = [jaccard_with_anchors(bbox, layer) for layer, _ in enumerate(xmin)]
        max_jaccard = tf.reduce_max([tf.reduce_max(ja) for ja in jaccard]) - 1e-7
        maxJaccard_mask = [tf.greater(jaccard[e], max_jaccard) for e in range(len(jaccard))]
        feat_maxMask = [tf.logical_or(maxJaccard_mask[e],feat_maxMask[e]) for e in range(len(jaccard))]
        mask = [tf.greater(jaccard[e]*tf.cast(maxJaccard_mask[e],tf.float32), feat_scores[e]) for e in range(len(jaccard))]

        mask = [tf.logical_and(mask[e], feat_scores[e] > -0.5) for e in range(len(jaccard))]
        mask = [tf.logical_and(mask[e], label < num_classes) for e in range(len(jaccard))]
        imask = [tf.cast(mask[e], tf.int32) for e in range(len(jaccard))]
        fmask = [tf.cast(mask[e], dtype) for e in range(len(jaccard))]
        # Update values using mask.
        feat_labels = [imask[e] * tf.cast(label, tf.int32) + (1 - imask[e]) * feat_labels[e] for e in range(len(jaccard))]
        feat_scores = [tf.where(mask[e], jaccard[e], feat_scores[e]) for e in range(len(jaccard))]

        feat_xmin = [fmask[e] * bbox[0] + (1 - fmask[e]) * feat_xmin[e] for e in range(len(jaccard))]
        feat_ymin = [fmask[e] * bbox[1] + (1 - fmask[e]) * feat_ymin[e] for e in range(len(jaccard))]
        feat_xmax = [fmask[e] * bbox[2] + (1 - fmask[e]) * feat_xmax[e] for e in range(len(jaccard))]
        feat_ymax = [fmask[e] * bbox[3] + (1 - fmask[e]) * feat_ymax[e] for e in range(len(jaccard))]
        return [i + 1, feat_labels, feat_scores,
                feat_xmin, feat_ymin, feat_xmax, feat_ymax, feat_maxMask]

    def body2(i, feat_labels, feat_scores,
             feat_xmin, feat_ymin, feat_xmax, feat_ymax, feat_maxMask):

        label = labels[i]
        bbox = bboxes[i]
        jaccard = [jaccard_with_anchors(bbox, layer) for layer, _ in enumerate(xmin)]

        mask = [tf.greater(jaccard[e], feat_scores[e]) for e in range(len(jaccard))]
        mask = [tf.logical_and(mask[e],tf.logical_not(feat_maxMask[e])) for e in range(len(jaccard))]

        mask = [tf.logical_and(mask[e], feat_scores[e] > -0.5) for e in range(len(jaccard))]
        mask = [tf.logical_and(mask[e], label < num_classes) for e in range(len(jaccard))]
        imask = [tf.cast(mask[e], tf.int32) for e in range(len(jaccard))]
        fmask = [tf.cast(mask[e], dtype) for e in range(len(jaccard))]
        # Update values using mask.
        feat_labels = [imask[e] * tf.cast(label,tf.int32) + (1 - imask[e]) * feat_labels[e] for e in range(len(jaccard))]
        feat_scores = [tf.where(mask[e], jaccard[e], feat_scores[e]) for e in range(len(jaccard))]

        feat_xmin = [fmask[e] * bbox[0] + (1 - fmask[e]) * feat_xmin[e] for e in range(len(jaccard))]
        feat_ymin = [fmask[e] * bbox[1] + (1 - fmask[e]) * feat_ymin[e] for e in range(len(jaccard))]
        feat_xmax = [fmask[e] * bbox[2] + (1 - fmask[e]) * feat_xmax[e] for e in range(len(jaccard))]
        feat_ymax = [fmask[e] * bbox[3] + (1 - fmask[e]) * feat_ymax[e] for e in range(len(jaccard))]
        return [i + 1, feat_labels, feat_scores,
                feat_xmin, feat_ymin, feat_xmax, feat_ymax, feat_maxMask]

    i = 0
    [i, feat_labels, feat_scores,
     feat_xmin, feat_ymin,
     feat_xmax, feat_ymax,feat_maxMask] = tf.while_loop(condition, body,
                                           [i, feat_labels, feat_scores,
                                            feat_xmin, feat_ymin,
                                            feat_xmax, feat_ymax,feat_maxMask])

    i = 0
    [i, feat_labels, feat_scores,
     feat_xmin, feat_ymin,
     feat_xmax, feat_ymax,feat_maxMask] = tf.while_loop(condition, body2,
                                           [i, feat_labels, feat_scores,
                                            feat_xmin, feat_ymin,
                                            feat_xmax, feat_ymax,feat_maxMask])

    # Transform to center / size.
    feat_cy = [(fymax + fymin) / 2. for fymax, fymin in zip(feat_ymax, feat_ymin)]
    feat_cx = [(fxmax + fxmin) / 2. for fxmax, fxmin in zip(feat_xmax, feat_xmin)]
    feat_h = [(fymax - fymin) for fymax, fymin in zip(feat_ymax, feat_ymin)]
    feat_w = [(fxmax - fxmin) for fxmax, fxmin in zip(feat_xmax, feat_xmin)]
    # Encode features.
    feat_cy = [(feat_cy[e] - yref[e]) / href[e] / prior_scaling[1] for e in range(len(feat_ymax))]
    feat_cx = [(feat_cx[e] - xref[e]) / wref[e] / prior_scaling[0] for e in range(len(feat_ymax))]
    feat_h = [tf.log(feat_h[e] / href[e]) / prior_scaling[3] for e in range(len(feat_ymax))]
    feat_w = [tf.log(feat_w[e] / wref[e]) / prior_scaling[2] for e in range(len(feat_ymax))]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = [tf.stack([feat_cx[e], feat_cy[e], feat_w[e], feat_h[e]], axis=-1) for e in range(len(feat_ymax))]
    return feat_labels, feat_localizations, feat_scores, feat_maxMask


def tf_bboxes_decode_layer(feat_localizations,
                           anchors_layer,
                           prior_scaling=[0.1, 0.1, 0.2, 0.2],
                           is_refine=False):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """
    if(is_refine):
        xmin_= anchors_layer[:,:,:,:,0]
        ymin_= anchors_layer[:,:,:,:,1]
        xmax_= anchors_layer[:,:,:,:,2]
        ymax_= anchors_layer[:,:,:,:,3]
        wref = xmax_ - xmin_
        href = ymax_ - ymin_
        xref = xmin_ + wref / 2.
        yref = ymin_ + href / 2.
    else:
        xref, yref, wref, href = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    return bboxes

def tf_bboxes_decode(feat_localizations,
                     anchors,
                     prior_scaling=[0.1, 0.1, 0.2, 0.2],
                     scope='ssd_bboxes_decode',is_refine=False):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_bboxes_decode_layer(feat_localizations[i],
                                       anchors_layer,
                                       prior_scaling,is_refine=is_refine))
        return bboxes

def bboxes_decode(feat_localizations,
                      anchor_bboxes,
                      prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Return:
      numpy array Nx4: ymin, xmin, ymax, xmax
    """
    # Reshape for easier broadcasting.
    l_shape = feat_localizations.shape
    feat_localizations = np.reshape(feat_localizations, (-1, l_shape[-2], l_shape[-1]))
    xref, yref, wref, href = anchor_bboxes
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    # bboxes: ymin, xmin, xmax, ymax.
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cx - w / 2.
    bboxes[:, :, 1] = cy - h / 2.
    bboxes[:, :, 2] = cx + w / 2.
    bboxes[:, :, 3] = cy + h / 2.
    # Back to original shape.
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=0.5,
                            img_shape=(512, 512),
                            num_classes=2,
                            decode=True):
    """Extract classes, scores and bounding boxes from features in one layer.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    # First decode localizations features if necessary.
    if decode:
        localizations_layer = bboxes_decode(localizations_layer, anchors_layer)

    # Reshape features to: Batches x N x N_labels | 4.
    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = np.argmax(predictions_layer, axis=2)
        scores = np.amax(predictions_layer, axis=2)
        mask = (classes > 0)
        classes = classes[mask]
        scores = scores[mask]
        bboxes = localizations_layer[mask]
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        idxes = np.where(sub_predictions > select_threshold)
        classes = idxes[-1]+1
        scores = sub_predictions[idxes]
        bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes


def bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    # l_layers = []
    # l_idxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
        # Debug information.
        # l_layers.append(i)
        # l_idxes.append((i, idxes))

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_resize(bbox_ref, bboxes):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform.
    """
    bboxes = np.copy(bboxes)
    # Translate.
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    # Resize.
    resize = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= resize[0]
    bboxes[:, 1] /= resize[1]
    bboxes[:, 2] /= resize[0]
    bboxes[:, 3] /= resize[1]
    return bboxes


def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_xmin = np.maximum(bboxes1[0], bboxes2[0])
    int_ymin = np.maximum(bboxes1[1], bboxes2[1])
    int_xmax = np.minimum(bboxes1[2], bboxes2[2])
    int_ymax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_xmin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_ymin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_xmax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_ymax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def bboxes_soft_nms(classes, scores, bboxes, nms_threshold=0.45, confidence_threshold=0.55):
    for i in range(scores.size-1):
        overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
        keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
        need_change_scores = np.logical_not(keep_overlap)
        index = np.where(need_change_scores)

        # scores[i+1:][index] = scores[i+1:][index] * (1 - overlap[index])

        scores[i+1:][index] = scores[i+1:][index] * np.exp(-(pow(overlap[index], 2) / 0.3))

        id = np.argsort(-scores[i+1:])
        scores[i+1:] = scores[i+1:][id]
        classes[i+1:] = classes[i+1:][id]
        bboxes[i+1:] = bboxes[i+1:,][id,]

    index = np.where(scores > confidence_threshold)

    return classes[index], scores[index] ,bboxes[index]








