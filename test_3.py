# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import args
from PIL import Image
import numpy as np
from utils.data_utils import get_batch_data
from model import yolov3
tf.reset_default_graph()

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
# =============================================================================
# train_dataset = train_dataset.shuffle(args.train_img_cnt)
# =============================================================================
train_dataset = train_dataset.batch(1)#args.batch_size)
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)


iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init_op = iterator.make_initializer(train_dataset)

# get an element from the chosen dataset iterator
image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data `static` shape, so we need to set it manually
image_ids.set_shape([None])
image.set_shape([None, None, None, 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
def box_iou(pred_boxes, valid_true_boxes):
    '''
    param:
        pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        valid_true: [V, 4]
    '''

    # [13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou
#compute loss
loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
anchor_group = [yolo_model.anchors[6:9], yolo_model.anchors[3:6], yolo_model.anchors[0:3]]
for i in range(len(pred_feature_maps)):
        # size in [h, w] format! don't get messed up!
    grid_size = tf.shape(pred_feature_maps[i])[1:3]
    # the downscale ratio in height and weight
    ratio = tf.cast(yolo_model.img_size / grid_size, tf.float32)
    # N: batch_size
    N = tf.cast(tf.shape(pred_feature_maps[i])[0], tf.float32)

    x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = yolo_model.reorg_layer(pred_feature_maps[i], anchor_group[i])

    ###########
    # get mask
    ###########

    # shape: take 416x416 input image and 13*13 feature_map for example:
    # [N, 13, 13, 3, 1]
    object_mask = y_true[i][..., 4:5]

    # the calculation of ignore mask if referred from
    # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    iou = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
    valid_true_boxes = tf.boolean_mask(y_true[i][0, ..., 0:4], tf.cast(object_mask[0, ..., 0], 'bool'))
# =============================================================================
#     iou = yolo_model.box_iou(pred_boxes[0], valid_true_boxes)
#     best_iou = tf.reduce_max(iou, axis=-1)
#     ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
# =============================================================================
    def loop_cond(idx, ignore_mask, iou):
        return tf.less(idx, tf.cast(N, tf.int32))
    def loop_body(idx, ignore_mask, iou):
        # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
        # V: num of true gt box of each image in a batch
        valid_true_boxes = tf.boolean_mask(y_true[i][idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
        # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
        temp_iou = box_iou(pred_boxes[idx], valid_true_boxes)
        # shape: [13, 13, 3]
        best_iou = tf.reduce_max(temp_iou, axis=-1)
        # shape: [13, 13, 3]
        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        # finally will be shape: [N, 13, 13, 3]
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        iou = iou.write(idx, best_iou)
        return idx + 1, ignore_mask, iou
    _, ignore_mask, iou = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask, iou])
    # shape: [N, 13, 13, 3]
    iou = iou.stack()
    iou = tf.expand_dims(iou, -1)
    ignore_mask = ignore_mask.stack()
    # shape: [N, 13, 13, 3, 1]
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    # shape: [N, 13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # get xy coordinates in one cell from the feature_map
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]
    true_xy = y_true[i][..., 0:2] / ratio[::-1] - x_y_offset
    pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

    # get_tw_th
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]
    true_tw_th = y_true[i][..., 2:4] / anchor_group[i]
    pred_tw_th = pred_box_wh / anchor_group[i]
    # for numerical stability
    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                          x=tf.ones_like(true_tw_th), y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                          x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
    pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

    # box size punishment: 
    # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
    # shape: [N, 13, 13, 3, 1]
    box_loss_scale = 2. - (y_true[i][..., 2:3] / tf.cast(yolo_model.img_size[1], tf.float32)) * (y_true[i][..., 3:4] / tf.cast(yolo_model.img_size[0], tf.float32))

    ############
    # loss_part
    ############
    # mix_up weight
    # [N, 13, 13, 3, 1]
    mix_w = y_true[i][..., -1:]
    # shape: [N, 13, 13, 3, 1]
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
    wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

    #iou loss
    iou_loss = (1 - iou)# * object_mask
    iou_loss = tf.where(tf.is_nan(iou_loss), tf.zeros_like(iou_loss), iou_loss)
    iou_loss_sum = tf.reduce_sum(iou_loss) / N
    
    # shape: [N, 13, 13, 3, 1]
    conf_pos_mask = object_mask
    conf_neg_mask = (1 - object_mask) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
    # TODO: may need to balance the pos-neg by multiplying some weights
    conf_loss = conf_loss_pos + conf_loss_neg
    if yolo_model.use_focal_loss:
        alpha = 1.0
        gamma = 2.0
        # TODO: alpha should be a mask array if needed
        focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
        conf_loss *= focal_mask
    conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

    # shape: [N, 13, 13, 3, 1]
    # whether to use label smooth
    if yolo_model.use_label_smooth:
        delta = 0.01
        label_target = (1 - delta) * y_true[i][..., 5:-1] + delta * 1. / yolo_model.class_num
    else:
        label_target = y_true[i][..., 5:-1]
    class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
    class_loss = tf.reduce_sum(class_loss) / N

    loss_xy += xy_loss
    loss_wh += wh_loss
    loss_conf += conf_loss
    loss_class += class_loss
    if i ==1:
        break
total_loss = loss_xy + loss_wh + loss_conf + loss_class
loss = [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)
global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)
    print('\n----------- start to train -----------\n')

    sess.run(train_init_op)
    __y_true, __loss, __pred_feature_maps, __x_y_offset, __pred_boxes, __pred_conf_logits, __pred_prob_logits, \
    __conf_pos_mask, __conf_neg_mask, __object_mask, __iou, _true_xy, _pred_xy, _iou_loss, _iou_loss_sum, \
    __box_loss_scale, __label_target, _valid_true_boxes, _image = sess.run(
        [y_true, loss, pred_feature_maps, x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits, 
         conf_pos_mask, conf_neg_mask, object_mask, iou, true_xy, pred_xy, iou_loss, iou_loss_sum,
         box_loss_scale, label_target, valid_true_boxes, image],
        feed_dict={is_training: True})
    
test_y_true = __y_true[1]
test_x = test_y_true[0,:,:,:,0]
test_y = test_y_true[0,:,:,:,1]
test_w = test_y_true[0,:,:,:,2]
test_h = test_y_true[0,:,:,:,3]
test_o = test_y_true[0,:,:,:,4]
test_mix = test_y_true[0,:,:,:,-1]
test_c = test_y_true[0,:,:,2,5:-1]

test_loss = __loss
test_pred_feature_maps = []
for i in range(3):
    test_pred_feature_maps.append(__pred_feature_maps[i][0,:])

test_x_y_offset = __x_y_offset[:,:,0,:]
test_pred_boxes = __pred_boxes[0,:,:,0,:]
test_pred_conf_logits = __pred_conf_logits[0,:,:,0,0]
test_pred_prob_logits = __pred_prob_logits[0,:,:,0,:]

test_best_iou = __iou[0,:,:,:,0]
# =============================================================================
# test_iou=[]
# for i in range(len(__iou[0,0,0,0,:])):
#     test_iou.append(__iou[0,:,:,:,i])
# =============================================================================
# =============================================================================
# test_valid_true_boxes = __valid_true_boxes
# test_best_iou = __best_iou
# test_ignore_mask_tmp = __ignore_mask_tmp
# =============================================================================
test_label_target = __label_target[0,:,:,1,:]
test__conf_pos_mask = __conf_pos_mask[0,:,:,:,0]
test__conf_neg_mask = __conf_neg_mask[0,:,:,:,0]
test__object_mask = __object_mask[0,:,:,:,0]

test__box_loss_scale = __box_loss_scale[0,:,:,:,0]

test_true_xy = _true_xy[0,:,:,:,0]
test_pred_xy = _pred_xy[0,:,:,:,0]

test_iou_loss = _iou_loss[0,:,:,:,0]
test_iou_loss_sum = _iou_loss_sum

test_valid_true_boxes = _valid_true_boxes
test_image = _image[0]
im = Image.fromarray(np.uint8(test_image*255))
im.show()