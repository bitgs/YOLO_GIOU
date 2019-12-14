# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
tf.reset_default_graph()
import args
import matplotlib.pyplot as plt
from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, parse_gt_rec, voc_ap
from utils.nms_utils import gpu_nms

from model import yolov3


# =============================================================================
# # setting placeholders
# is_training = tf.placeholder(tf.bool, name="phase_train")
# handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
# # register the gpu nms operation here for the following evaluation scheme
# pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
# pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
# gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)
# 
# ##################
# # tf.data pipeline
# ##################
# 
# val_dataset = tf.data.TextLineDataset(args.val_file)
# val_dataset = val_dataset.batch(1)
# val_dataset = val_dataset.map(
#     lambda x: tf.py_func(get_batch_data,
#                          inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False, args.letterbox_resize],
#                          Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
#     num_parallel_calls=args.num_threads
# )
# val_dataset.prefetch(args.prefetech_buffer)
# 
# iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
# val_init_op = iterator.make_initializer(val_dataset)
# 
# # get an element from the chosen dataset iterator
# image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
# y_true = [y_true_13, y_true_26, y_true_52]
# 
# # tf.data pipeline will lose the data `static` shape, so we need to set it manually
# image_ids.set_shape([None])
# image.set_shape([None, None, None, 3])
# for y in y_true:
#     y.set_shape([None, None, None, None, None])
# 
# ##################
# # Model definition
# ##################
# yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
# with tf.variable_scope('yolov3'):
#     pred_feature_maps = yolo_model.forward(image, is_training=is_training)
# loss = yolo_model.compute_loss(pred_feature_maps, y_true)
# y_pred = yolo_model.predict(pred_feature_maps)
# 
# # setting restore parts and vars to update
# saver_to_restore = tf.train.Saver()
# gpu_options = tf.GPUOptions(allow_growth=True)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     saver_to_restore.restore(sess, args.restore_path)
# 
#     print('\n----------- start to eval -----------\n')
# 
#     best_mAP = -np.Inf
# 
#     sess.run(val_init_op)
# 
#     val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
#         AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
# 
#     val_preds = []
# 
#     for j in trange(args.val_img_cnt):
#         __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss],
#                                                  feed_dict={is_training: False})
#         pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)
#         val_preds.extend(pred_content)
#         val_loss_total.update(__loss[0])
#         val_loss_xy.update(__loss[1])
#         val_loss_wh.update(__loss[2])
#         val_loss_conf.update(__loss[3])
#         val_loss_class.update(__loss[4])
#     # calc mAP
#     rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
#     gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)
# =============================================================================

iou_thres = args.eval_threshold
use_07_metric = False
print(val_preds[:10])
for classidx in range(args.class_num):
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])
    
    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    img_ids = [img_ids[x] for x in sorted_ind]
    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']
        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    plt.plot(rec, prec)
    plt.show()
    break
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    rec =  tp[-1] / float(npos)
    prec = tp[-1] / float(nd)
    rec_total.update(rec, npos)
    prec_total.update(prec, nd)
    ap_total.update(ap, 1)

mAP = ap_total.average
Recall = rec_total.average
Precision = prec_total.average
loss_total, loss_xy, loss_wh, loss_conf, loss_class = val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average