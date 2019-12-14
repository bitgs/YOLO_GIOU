#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:35:25 2019

@author: yuan
"""
import tensorflow as tf
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
import args

# calc mAP
rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)