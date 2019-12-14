# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import args
from PIL import Image
import numpy as np
from utils.data_utils import get_batch_data
from model_2 import yolov3
import inception_resnet_v2 as inception
tf.reset_default_graph()

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
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

image.set_shape([None, 416, 416, 3])

##################
# Model definition
##################
num_classes = 1001
with tf.contrib.slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
    f1,f2,f3 = inception.inception_resnet_v2(image, num_classes, is_training=is_training)


# setting restore parts and vars to update
saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)
global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver_to_restore.restore(sess, '/home/yuan/Downloads/inception_resnet_v2_2016_08_30.ckpt')
    print('\n----------- start to train -----------\n')

    sess.run(train_init_op)
    _image, _f1,_f2,_f3= sess.run([image, f1,f2,f3], feed_dict={is_training: False})


a,b,c= _f1,_f2,_f3
test_image = _image[0]
im = Image.fromarray(np.uint8(test_image*255))
im.show()