import tensorflow as tf
import numpy as np
import math

def anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    # h[0] = sizes[0] / img_shape[0]
    # w[0] = sizes[0] / img_shape[1]

    for i, r in enumerate(ratios):
        # r_ = math.sqrt(float(r))
        # r0 = float('%.3f' % r_)
        h[i] = float(sizes[0]) / float(img_shape[0]) / math.sqrt(float(r))
        w[i] = float(sizes[1]) / float(img_shape[1]) * math.sqrt(float(r))
    # for i, r in enumerate(ratios):
    #     h[len(ratios)+i] = 1/2.*sizes[0] / img_shape[0] / r
    #     w[len(ratios)+i] = 1/2.*sizes[1] / img_shape[1] * r
    # for i, r in enumerate(ratios):
    #     h[2*len(ratios)+i] = 1/3.*sizes[0] / img_shape[0] / r
    #     w[2*len(ratios)+i] = 1/3.*sizes[1] / img_shape[1] * r
    # for i, r in enumerate(ratios):
    #     h[3*len(ratios)+i] = 2.*sizes[0] / img_shape[0] / r
    #     w[3*len(ratios)+i] = 2.*sizes[1] / img_shape[1] * r
    # for i, r in enumerate(ratios):
    #     h[len(ratios)+i] = sizes[0] / img_shape[0] / r
    #     w[len(ratios)+i] = sizes[1] / img_shape[1] * r
    return x, y, w, h

def anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, feat in enumerate(layers_shape):
        anchor_bboxes = anchor_one_layer(img_shape,
                                         feat,
                                         anchor_sizes[i],
                                         anchor_ratios[i],
                                         anchor_steps[i],
                                         offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors