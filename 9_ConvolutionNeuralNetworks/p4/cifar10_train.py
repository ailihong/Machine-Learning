# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


from tensorflow.models.image.cifar10 import cifar10



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/Users/water/Documents/cs498takehomefinal/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('train_dir0', '/Users/water/Documents/cs498takehomefinal/cifar10_train0',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir1', '/Users/water/Documents/cs498takehomefinal/cifar10_train1',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir2', '/Users/water/Documents/cs498takehomefinal/cifar10_train2',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir3', '/Users/water/Documents/cs498takehomefinal/cifar10_train3',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir4', '/Users/water/Documents/cs498takehomefinal/cifar10_train4',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir5', '/Users/water/Documents/cs498takehomefinal/cifar10_train5',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir6', '/Users/water/Documents/cs498takehomefinal/cifar10_train6',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir7', '/Users/water/Documents/cs498takehomefinal/cifar10_train7',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir8', '/Users/water/Documents/cs498takehomefinal/cifar10_train8',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir9', '/Users/water/Documents/cs498takehomefinal/cifar10_train9',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir10', '/Users/water/Documents/cs498takehomefinal/cifar10_train10',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir11', '/Users/water/Documents/cs498takehomefinal/cifar10_train11',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir12', '/Users/water/Documents/cs498takehomefinal/cifar10_train12',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir13', '/Users/water/Documents/cs498takehomefinal/cifar10_train13',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir14', '/Users/water/Documents/cs498takehomefinal/cifar10_train14',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir15', '/Users/water/Documents/cs498takehomefinal/cifar10_train15',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir16', '/Users/water/Documents/cs498takehomefinal/cifar10_train16',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir17', '/Users/water/Documents/cs498takehomefinal/cifar10_train17',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir18', '/Users/water/Documents/cs498takehomefinal/cifar10_train18',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir19', '/Users/water/Documents/cs498takehomefinal/cifar10_train19',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_string('train_dir20', '/Users/water/Documents/cs498takehomefinal/cifar10_train20',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir21', '/Users/water/Documents/cs498takehomefinal/cifar10_train21',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir22', '/Users/water/Documents/cs498takehomefinal/cifar10_train22',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir23', '/Users/water/Documents/cs498takehomefinal/cifar10_train23',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir24', '/Users/water/Documents/cs498takehomefinal/cifar10_train24',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir25', '/Users/water/Documents/cs498takehomefinal/cifar10_train25',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir26', '/Users/water/Documents/cs498takehomefinal/cifar10_train26',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir27', '/Users/water/Documents/cs498takehomefinal/cifar10_train27',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir28', '/Users/water/Documents/cs498takehomefinal/cifar10_train28',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir29', '/Users/water/Documents/cs498takehomefinal/cifar10_train29',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir30', '/Users/water/Documents/cs498takehomefinal/cifar10_train30',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir31', '/Users/water/Documents/cs498takehomefinal/cifar10_train31',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir32', '/Users/water/Documents/cs498takehomefinal/cifar10_train32',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir33', '/Users/water/Documents/cs498takehomefinal/cifar10_train33',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir34', '/Users/water/Documents/cs498takehomefinal/cifar10_train34',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir35', '/Users/water/Documents/cs498takehomefinal/cifar10_train35',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir36', '/Users/water/Documents/cs498takehomefinal/cifar10_train36',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir37', '/Users/water/Documents/cs498takehomefinal/cifar10_train37',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir38', '/Users/water/Documents/cs498takehomefinal/cifar10_train38',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir39', '/Users/water/Documents/cs498takehomefinal/cifar10_train39',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_string('train_dir40', '/Users/water/Documents/cs498takehomefinal/cifar10_train40',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir41', '/Users/water/Documents/cs498takehomefinal/cifar10_train41',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir42', '/Users/water/Documents/cs498takehomefinal/cifar10_train42',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir43', '/Users/water/Documents/cs498takehomefinal/cifar10_train43',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir44', '/Users/water/Documents/cs498takehomefinal/cifar10_train44',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir45', '/Users/water/Documents/cs498takehomefinal/cifar10_train45',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir46', '/Users/water/Documents/cs498takehomefinal/cifar10_train46',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir47', '/Users/water/Documents/cs498takehomefinal/cifar10_train47',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir48', '/Users/water/Documents/cs498takehomefinal/cifar10_train48',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir49', '/Users/water/Documents/cs498takehomefinal/cifar10_train49',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir50', '/Users/water/Documents/cs498takehomefinal/cifar10_train50',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir51', '/Users/water/Documents/cs498takehomefinal/cifar10_train51',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir52', '/Users/water/Documents/cs498takehomefinal/cifar10_train52',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir53', '/Users/water/Documents/cs498takehomefinal/cifar10_train53',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir54', '/Users/water/Documents/cs498takehomefinal/cifar10_train54',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir55', '/Users/water/Documents/cs498takehomefinal/cifar10_train55',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir56', '/Users/water/Documents/cs498takehomefinal/cifar10_train56',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir57', '/Users/water/Documents/cs498takehomefinal/cifar10_train57',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir58', '/Users/water/Documents/cs498takehomefinal/cifar10_train58',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir59', '/Users/water/Documents/cs498takehomefinal/cifar10_train59',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('train_dir60', '/Users/water/Documents/cs498takehomefinal/cifar10_train60',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir61', '/Users/water/Documents/cs498takehomefinal/cifar10_train61',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir62', '/Users/water/Documents/cs498takehomefinal/cifar10_train62',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir63', '/Users/water/Documents/cs498takehomefinal/cifar10_train63',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir64', '/Users/water/Documents/cs498takehomefinal/cifar10_train64',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir65', '/Users/water/Documents/cs498takehomefinal/cifar10_train65',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir66', '/Users/water/Documents/cs498takehomefinal/cifar10_train66',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir67', '/Users/water/Documents/cs498takehomefinal/cifar10_train67',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir68', '/Users/water/Documents/cs498takehomefinal/cifar10_train68',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir69', '/Users/water/Documents/cs498takehomefinal/cifar10_train69',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir70', '/Users/water/Documents/cs498takehomefinal/cifar10_train70',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir71', '/Users/water/Documents/cs498takehomefinal/cifar10_train71',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir72', '/Users/water/Documents/cs498takehomefinal/cifar10_train72',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir73', '/Users/water/Documents/cs498takehomefinal/cifar10_train73',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir74', '/Users/water/Documents/cs498takehomefinal/cifar10_train74',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir75', '/Users/water/Documents/cs498takehomefinal/cifar10_train75',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir76', '/Users/water/Documents/cs498takehomefinal/cifar10_train76',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir77', '/Users/water/Documents/cs498takehomefinal/cifar10_train77',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir78', '/Users/water/Documents/cs498takehomefinal/cifar10_train78',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir79', '/Users/water/Documents/cs498takehomefinal/cifar10_train79',
                           """Directory where to write event logs """
                           """and checkpoint.""")


tf.app.flags.DEFINE_string('train_dir80', '/Users/water/Documents/cs498takehomefinal/cifar10_train80',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir81', '/Users/water/Documents/cs498takehomefinal/cifar10_train81',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir82', '/Users/water/Documents/cs498takehomefinal/cifar10_train82',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir83', '/Users/water/Documents/cs498takehomefinal/cifar10_train83',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir84', '/Users/water/Documents/cs498takehomefinal/cifar10_train84',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir85', '/Users/water/Documents/cs498takehomefinal/cifar10_train85',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir86', '/Users/water/Documents/cs498takehomefinal/cifar10_train86',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir87', '/Users/water/Documents/cs498takehomefinal/cifar10_train87',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir88', '/Users/water/Documents/cs498takehomefinal/cifar10_train88',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir89', '/Users/water/Documents/cs498takehomefinal/cifar10_train89',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir90', '/Users/water/Documents/cs498takehomefinal/cifar10_train90',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir91', '/Users/water/Documents/cs498takehomefinal/cifar10_train91',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir92', '/Users/water/Documents/cs498takehomefinal/cifar10_train92',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir93', '/Users/water/Documents/cs498takehomefinal/cifar10_train93',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir94', '/Users/water/Documents/cs498takehomefinal/cifar10_train94',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir95', '/Users/water/Documents/cs498takehomefinal/cifar10_train95',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir96', '/Users/water/Documents/cs498takehomefinal/cifar10_train96',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir97', '/Users/water/Documents/cs498takehomefinal/cifar10_train97',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir98', '/Users/water/Documents/cs498takehomefinal/cifar10_train98',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir99', '/Users/water/Documents/cs498takehomefinal/cifar10_train99',
                           """Directory where to write event logs """
                           """and checkpoint.""")





tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)


    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

    summary_writer0 = tf.train.SummaryWriter(FLAGS.train_dir0)
    summary_writer1= tf.train.SummaryWriter(FLAGS.train_dir1)
    summary_writer2 = tf.train.SummaryWriter(FLAGS.train_dir2)
    summary_writer3 = tf.train.SummaryWriter(FLAGS.train_dir3)
    summary_writer4 = tf.train.SummaryWriter(FLAGS.train_dir4)
    summary_writer5 = tf.train.SummaryWriter(FLAGS.train_dir5)
    summary_writer6 = tf.train.SummaryWriter(FLAGS.train_dir6)
    summary_writer7 = tf.train.SummaryWriter(FLAGS.train_dir7)
    summary_writer8 = tf.train.SummaryWriter(FLAGS.train_dir8)
    summary_writer9 = tf.train.SummaryWriter(FLAGS.train_dir9)
    summary_writer10 = tf.train.SummaryWriter(FLAGS.train_dir10)
    summary_writer11 = tf.train.SummaryWriter(FLAGS.train_dir11)
    summary_writer12 = tf.train.SummaryWriter(FLAGS.train_dir12)
    summary_writer13 = tf.train.SummaryWriter(FLAGS.train_dir13)
    summary_writer14 = tf.train.SummaryWriter(FLAGS.train_dir14)
    summary_writer15 = tf.train.SummaryWriter(FLAGS.train_dir15)
    summary_writer16 = tf.train.SummaryWriter(FLAGS.train_dir16)
    summary_writer17 = tf.train.SummaryWriter(FLAGS.train_dir17)
    summary_writer18 = tf.train.SummaryWriter(FLAGS.train_dir18)
    summary_writer19 = tf.train.SummaryWriter(FLAGS.train_dir19)

    summary_writer20 = tf.train.SummaryWriter(FLAGS.train_dir20)
    summary_writer21 = tf.train.SummaryWriter(FLAGS.train_dir21)
    summary_writer22 = tf.train.SummaryWriter(FLAGS.train_dir22)
    summary_writer23 = tf.train.SummaryWriter(FLAGS.train_dir23)
    summary_writer24 = tf.train.SummaryWriter(FLAGS.train_dir24)
    summary_writer25 = tf.train.SummaryWriter(FLAGS.train_dir25)
    summary_writer26 = tf.train.SummaryWriter(FLAGS.train_dir26)
    summary_writer27 = tf.train.SummaryWriter(FLAGS.train_dir27)
    summary_writer28 = tf.train.SummaryWriter(FLAGS.train_dir28)
    summary_writer29 = tf.train.SummaryWriter(FLAGS.train_dir29)
    summary_writer30 = tf.train.SummaryWriter(FLAGS.train_dir30)
    summary_writer31 = tf.train.SummaryWriter(FLAGS.train_dir31)
    summary_writer32 = tf.train.SummaryWriter(FLAGS.train_dir32)
    summary_writer33 = tf.train.SummaryWriter(FLAGS.train_dir33)
    summary_writer34 = tf.train.SummaryWriter(FLAGS.train_dir34)
    summary_writer35 = tf.train.SummaryWriter(FLAGS.train_dir35)
    summary_writer36 = tf.train.SummaryWriter(FLAGS.train_dir36)
    summary_writer37 = tf.train.SummaryWriter(FLAGS.train_dir37)
    summary_writer38 = tf.train.SummaryWriter(FLAGS.train_dir38)
    summary_writer39 = tf.train.SummaryWriter(FLAGS.train_dir39)

    summary_writer40 = tf.train.SummaryWriter(FLAGS.train_dir40)
    summary_writer41 = tf.train.SummaryWriter(FLAGS.train_dir41)
    summary_writer42 = tf.train.SummaryWriter(FLAGS.train_dir42)
    summary_writer43 = tf.train.SummaryWriter(FLAGS.train_dir43)
    summary_writer44 = tf.train.SummaryWriter(FLAGS.train_dir44)
    summary_writer45 = tf.train.SummaryWriter(FLAGS.train_dir45)
    summary_writer46 = tf.train.SummaryWriter(FLAGS.train_dir46)
    summary_writer47 = tf.train.SummaryWriter(FLAGS.train_dir47)
    summary_writer48 = tf.train.SummaryWriter(FLAGS.train_dir48)
    summary_writer49 = tf.train.SummaryWriter(FLAGS.train_dir49)
    summary_writer50 = tf.train.SummaryWriter(FLAGS.train_dir50)
    summary_writer51 = tf.train.SummaryWriter(FLAGS.train_dir51)
    summary_writer52 = tf.train.SummaryWriter(FLAGS.train_dir52)
    summary_writer53 = tf.train.SummaryWriter(FLAGS.train_dir53)
    summary_writer54 = tf.train.SummaryWriter(FLAGS.train_dir54)
    summary_writer55 = tf.train.SummaryWriter(FLAGS.train_dir55)
    summary_writer56 = tf.train.SummaryWriter(FLAGS.train_dir56)
    summary_writer57 = tf.train.SummaryWriter(FLAGS.train_dir57)
    summary_writer58 = tf.train.SummaryWriter(FLAGS.train_dir58)
    summary_writer59 = tf.train.SummaryWriter(FLAGS.train_dir59)

    summary_writer60 = tf.train.SummaryWriter(FLAGS.train_dir60)
    summary_writer61 = tf.train.SummaryWriter(FLAGS.train_dir61)
    summary_writer62 = tf.train.SummaryWriter(FLAGS.train_dir62)
    summary_writer63 = tf.train.SummaryWriter(FLAGS.train_dir63)
    summary_writer64 = tf.train.SummaryWriter(FLAGS.train_dir64)
    summary_writer65 = tf.train.SummaryWriter(FLAGS.train_dir65)
    summary_writer66 = tf.train.SummaryWriter(FLAGS.train_dir66)
    summary_writer67 = tf.train.SummaryWriter(FLAGS.train_dir67)
    summary_writer68 = tf.train.SummaryWriter(FLAGS.train_dir68)
    summary_writer69 = tf.train.SummaryWriter(FLAGS.train_dir69)
    summary_writer70 = tf.train.SummaryWriter(FLAGS.train_dir70)
    summary_writer71 = tf.train.SummaryWriter(FLAGS.train_dir71)
    summary_writer72 = tf.train.SummaryWriter(FLAGS.train_dir72)
    summary_writer73 = tf.train.SummaryWriter(FLAGS.train_dir73)
    summary_writer74 = tf.train.SummaryWriter(FLAGS.train_dir74)
    summary_writer75 = tf.train.SummaryWriter(FLAGS.train_dir75)
    summary_writer76 = tf.train.SummaryWriter(FLAGS.train_dir76)
    summary_writer77 = tf.train.SummaryWriter(FLAGS.train_dir77)
    summary_writer78 = tf.train.SummaryWriter(FLAGS.train_dir78)
    summary_writer79 = tf.train.SummaryWriter(FLAGS.train_dir79)

    summary_writer80 = tf.train.SummaryWriter(FLAGS.train_dir80)
    summary_writer81 = tf.train.SummaryWriter(FLAGS.train_dir81)
    summary_writer82 = tf.train.SummaryWriter(FLAGS.train_dir82)
    summary_writer83 = tf.train.SummaryWriter(FLAGS.train_dir83)
    summary_writer84 = tf.train.SummaryWriter(FLAGS.train_dir84)
    summary_writer85 = tf.train.SummaryWriter(FLAGS.train_dir85)
    summary_writer86 = tf.train.SummaryWriter(FLAGS.train_dir86)
    summary_writer87 = tf.train.SummaryWriter(FLAGS.train_dir87)
    summary_writer88 = tf.train.SummaryWriter(FLAGS.train_dir88)
    summary_writer89 = tf.train.SummaryWriter(FLAGS.train_dir89)
    summary_writer90 = tf.train.SummaryWriter(FLAGS.train_dir90)
    summary_writer91 = tf.train.SummaryWriter(FLAGS.train_dir91)
    summary_writer92 = tf.train.SummaryWriter(FLAGS.train_dir92)
    summary_writer93 = tf.train.SummaryWriter(FLAGS.train_dir93)
    summary_writer94 = tf.train.SummaryWriter(FLAGS.train_dir94)
    summary_writer95 = tf.train.SummaryWriter(FLAGS.train_dir95)
    summary_writer96 = tf.train.SummaryWriter(FLAGS.train_dir96)
    summary_writer97 = tf.train.SummaryWriter(FLAGS.train_dir97)
    summary_writer98 = tf.train.SummaryWriter(FLAGS.train_dir98)
    summary_writer99 = tf.train.SummaryWriter(FLAGS.train_dir99)
   


    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'


      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        summary_writer0.add_summary(summary_str, step)
        summary_writer1.add_summary(summary_str, step)
        summary_writer2.add_summary(summary_str, step)
        summary_writer3.add_summary(summary_str, step)
        summary_writer4.add_summary(summary_str, step)
        summary_writer5.add_summary(summary_str, step)
        summary_writer6.add_summary(summary_str, step)
        summary_writer7.add_summary(summary_str, step)
        summary_writer8.add_summary(summary_str, step)
        summary_writer9.add_summary(summary_str, step)
        summary_writer10.add_summary(summary_str, step)
        summary_writer11.add_summary(summary_str, step)
        summary_writer12.add_summary(summary_str, step)
        summary_writer13.add_summary(summary_str, step)
        summary_writer14.add_summary(summary_str, step)
        summary_writer15.add_summary(summary_str, step)
        summary_writer16.add_summary(summary_str, step)
        summary_writer17.add_summary(summary_str, step)
        summary_writer18.add_summary(summary_str, step)
        summary_writer19.add_summary(summary_str, step)

        summary_writer20.add_summary(summary_str, step)
        summary_writer21.add_summary(summary_str, step)
        summary_writer22.add_summary(summary_str, step)
        summary_writer23.add_summary(summary_str, step)
        summary_writer24.add_summary(summary_str, step)
        summary_writer25.add_summary(summary_str, step)
        summary_writer26.add_summary(summary_str, step)
        summary_writer27.add_summary(summary_str, step)
        summary_writer28.add_summary(summary_str, step)
        summary_writer29.add_summary(summary_str, step)
        summary_writer30.add_summary(summary_str, step)
        summary_writer31.add_summary(summary_str, step)
        summary_writer32.add_summary(summary_str, step)
        summary_writer33.add_summary(summary_str, step)
        summary_writer34.add_summary(summary_str, step)
        summary_writer35.add_summary(summary_str, step)
        summary_writer36.add_summary(summary_str, step)
        summary_writer37.add_summary(summary_str, step)
        summary_writer38.add_summary(summary_str, step)
        summary_writer39.add_summary(summary_str, step)

        summary_writer40.add_summary(summary_str, step)
        summary_writer41.add_summary(summary_str, step)
        summary_writer42.add_summary(summary_str, step)
        summary_writer43.add_summary(summary_str, step)
        summary_writer44.add_summary(summary_str, step)
        summary_writer45.add_summary(summary_str, step)
        summary_writer46.add_summary(summary_str, step)
        summary_writer47.add_summary(summary_str, step)
        summary_writer48.add_summary(summary_str, step)
        summary_writer49.add_summary(summary_str, step)
        summary_writer50.add_summary(summary_str, step)
        summary_writer51.add_summary(summary_str, step)
        summary_writer52.add_summary(summary_str, step)
        summary_writer53.add_summary(summary_str, step)
        summary_writer54.add_summary(summary_str, step)
        summary_writer55.add_summary(summary_str, step)
        summary_writer56.add_summary(summary_str, step)
        summary_writer57.add_summary(summary_str, step)
        summary_writer58.add_summary(summary_str, step)
        summary_writer59.add_summary(summary_str, step)

        summary_writer60.add_summary(summary_str, step)
        summary_writer61.add_summary(summary_str, step)
        summary_writer62.add_summary(summary_str, step)
        summary_writer63.add_summary(summary_str, step)
        summary_writer64.add_summary(summary_str, step)
        summary_writer65.add_summary(summary_str, step)
        summary_writer66.add_summary(summary_str, step)
        summary_writer67.add_summary(summary_str, step)
        summary_writer68.add_summary(summary_str, step)
        summary_writer69.add_summary(summary_str, step)
        summary_writer70.add_summary(summary_str, step)
        summary_writer71.add_summary(summary_str, step)
        summary_writer72.add_summary(summary_str, step)
        summary_writer73.add_summary(summary_str, step)
        summary_writer74.add_summary(summary_str, step)
        summary_writer75.add_summary(summary_str, step)
        summary_writer76.add_summary(summary_str, step)
        summary_writer77.add_summary(summary_str, step)
        summary_writer78.add_summary(summary_str, step)
        summary_writer79.add_summary(summary_str, step)

        summary_writer80.add_summary(summary_str, step)
        summary_writer81.add_summary(summary_str, step)
        summary_writer82.add_summary(summary_str, step)
        summary_writer83.add_summary(summary_str, step)
        summary_writer84.add_summary(summary_str, step)
        summary_writer85.add_summary(summary_str, step)
        summary_writer86.add_summary(summary_str, step)
        summary_writer87.add_summary(summary_str, step)
        summary_writer88.add_summary(summary_str, step)
        summary_writer89.add_summary(summary_str, step)
        summary_writer90.add_summary(summary_str, step)
        summary_writer91.add_summary(summary_str, step)
        summary_writer92.add_summary(summary_str, step)
        summary_writer93.add_summary(summary_str, step)
        summary_writer94.add_summary(summary_str, step)
        summary_writer95.add_summary(summary_str, step)
        summary_writer96.add_summary(summary_str, step)
        summary_writer97.add_summary(summary_str, step)
        summary_writer98.add_summary(summary_str, step)
        summary_writer99.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      # if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #   checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      #   saver.save(sess, checkpoint_path, global_step=step/100)

        # hard cord here!!!
      if step==100:
        checkpoint_path = os.path.join(FLAGS.train_dir0, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==200:
        checkpoint_path = os.path.join(FLAGS.train_dir1, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==300:
        checkpoint_path = os.path.join(FLAGS.train_dir2, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==400:
        checkpoint_path = os.path.join(FLAGS.train_dir3, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==500:
        checkpoint_path = os.path.join(FLAGS.train_dir4, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==600:
        checkpoint_path = os.path.join(FLAGS.train_dir5, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==700:
        checkpoint_path = os.path.join(FLAGS.train_dir6, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==800:
        checkpoint_path = os.path.join(FLAGS.train_dir7, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==900:
        checkpoint_path = os.path.join(FLAGS.train_dir8, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1000:
        checkpoint_path = os.path.join(FLAGS.train_dir9, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1100:
        checkpoint_path = os.path.join(FLAGS.train_dir10, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1200:
        checkpoint_path = os.path.join(FLAGS.train_dir11, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1300:
        checkpoint_path = os.path.join(FLAGS.train_dir12, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1400:
        checkpoint_path = os.path.join(FLAGS.train_dir13, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1500:
        checkpoint_path = os.path.join(FLAGS.train_dir14, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1600:
        checkpoint_path = os.path.join(FLAGS.train_dir15, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1700:
        checkpoint_path = os.path.join(FLAGS.train_dir16, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1800:
        checkpoint_path = os.path.join(FLAGS.train_dir17, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==1900:
        checkpoint_path = os.path.join(FLAGS.train_dir18, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2000:
        checkpoint_path = os.path.join(FLAGS.train_dir19, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==2100:
        checkpoint_path = os.path.join(FLAGS.train_dir20, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==2200:
        checkpoint_path = os.path.join(FLAGS.train_dir21, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==2300:
        checkpoint_path = os.path.join(FLAGS.train_dir22, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2400:
        checkpoint_path = os.path.join(FLAGS.train_dir23, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2500:
        checkpoint_path = os.path.join(FLAGS.train_dir24, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2600:
        checkpoint_path = os.path.join(FLAGS.train_dir25, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2700:
        checkpoint_path = os.path.join(FLAGS.train_dir26, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2800:
        checkpoint_path = os.path.join(FLAGS.train_dir27, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==2900:
        checkpoint_path = os.path.join(FLAGS.train_dir28, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3000:
        checkpoint_path = os.path.join(FLAGS.train_dir29, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3100:
        checkpoint_path = os.path.join(FLAGS.train_dir30, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3200:
        checkpoint_path = os.path.join(FLAGS.train_dir31, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3300:
        checkpoint_path = os.path.join(FLAGS.train_dir32, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3400:
        checkpoint_path = os.path.join(FLAGS.train_dir33, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3500:
        checkpoint_path = os.path.join(FLAGS.train_dir34, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3600:
        checkpoint_path = os.path.join(FLAGS.train_dir35, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3700:
        checkpoint_path = os.path.join(FLAGS.train_dir36, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3800:
        checkpoint_path = os.path.join(FLAGS.train_dir37, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==3900:
        checkpoint_path = os.path.join(FLAGS.train_dir38, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4000:
        checkpoint_path = os.path.join(FLAGS.train_dir39, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==4100:
        checkpoint_path = os.path.join(FLAGS.train_dir40, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==4200:
        checkpoint_path = os.path.join(FLAGS.train_dir41, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==4300:
        checkpoint_path = os.path.join(FLAGS.train_dir42, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4400:
        checkpoint_path = os.path.join(FLAGS.train_dir43, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4500:
        checkpoint_path = os.path.join(FLAGS.train_dir44, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4600:
        checkpoint_path = os.path.join(FLAGS.train_dir45, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4700:
        checkpoint_path = os.path.join(FLAGS.train_dir46, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4800:
        checkpoint_path = os.path.join(FLAGS.train_dir47, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==4900:
        checkpoint_path = os.path.join(FLAGS.train_dir48, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5000:
        checkpoint_path = os.path.join(FLAGS.train_dir49, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5100:
        checkpoint_path = os.path.join(FLAGS.train_dir50, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5200:
        checkpoint_path = os.path.join(FLAGS.train_dir51, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5300:
        checkpoint_path = os.path.join(FLAGS.train_dir52, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5400:
        checkpoint_path = os.path.join(FLAGS.train_dir53, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5500:
        checkpoint_path = os.path.join(FLAGS.train_dir54, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5600:
        checkpoint_path = os.path.join(FLAGS.train_dir55, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5700:
        checkpoint_path = os.path.join(FLAGS.train_dir56, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5800:
        checkpoint_path = os.path.join(FLAGS.train_dir57, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==5900:
        checkpoint_path = os.path.join(FLAGS.train_dir58, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6000:
        checkpoint_path = os.path.join(FLAGS.train_dir59, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==6100:
        checkpoint_path = os.path.join(FLAGS.train_dir60, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==6200:
        checkpoint_path = os.path.join(FLAGS.train_dir61, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==6300:
        checkpoint_path = os.path.join(FLAGS.train_dir62, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6400:
        checkpoint_path = os.path.join(FLAGS.train_dir63, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6500:
        checkpoint_path = os.path.join(FLAGS.train_dir64, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6600:
        checkpoint_path = os.path.join(FLAGS.train_dir65, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6700:
        checkpoint_path = os.path.join(FLAGS.train_dir66, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6800:
        checkpoint_path = os.path.join(FLAGS.train_dir67, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==6900:
        checkpoint_path = os.path.join(FLAGS.train_dir68, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7000:
        checkpoint_path = os.path.join(FLAGS.train_dir69, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7100:
        checkpoint_path = os.path.join(FLAGS.train_dir70, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7200:
        checkpoint_path = os.path.join(FLAGS.train_dir71, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7300:
        checkpoint_path = os.path.join(FLAGS.train_dir72, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7400:
        checkpoint_path = os.path.join(FLAGS.train_dir73, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7500:
        checkpoint_path = os.path.join(FLAGS.train_dir74, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7600:
        checkpoint_path = os.path.join(FLAGS.train_dir75, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7700:
        checkpoint_path = os.path.join(FLAGS.train_dir76, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7800:
        checkpoint_path = os.path.join(FLAGS.train_dir77, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==7900:
        checkpoint_path = os.path.join(FLAGS.train_dir78, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8000:
        checkpoint_path = os.path.join(FLAGS.train_dir79, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==8100:
        checkpoint_path = os.path.join(FLAGS.train_dir80, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==8200:
        checkpoint_path = os.path.join(FLAGS.train_dir81, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      if step==8300:
        checkpoint_path = os.path.join(FLAGS.train_dir82, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8400:
        checkpoint_path = os.path.join(FLAGS.train_dir83, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8500:
        checkpoint_path = os.path.join(FLAGS.train_dir84, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8600:
        checkpoint_path = os.path.join(FLAGS.train_dir85, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8700:
        checkpoint_path = os.path.join(FLAGS.train_dir86, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8800:
        checkpoint_path = os.path.join(FLAGS.train_dir87, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==8900:
        checkpoint_path = os.path.join(FLAGS.train_dir88, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9000:
        checkpoint_path = os.path.join(FLAGS.train_dir89, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9100:
        checkpoint_path = os.path.join(FLAGS.train_dir90, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9200:
        checkpoint_path = os.path.join(FLAGS.train_dir91, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9300:
        checkpoint_path = os.path.join(FLAGS.train_dir92, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9400:
        checkpoint_path = os.path.join(FLAGS.train_dir93, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9500:
        checkpoint_path = os.path.join(FLAGS.train_dir94, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9600:
        checkpoint_path = os.path.join(FLAGS.train_dir95, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9700:
        checkpoint_path = os.path.join(FLAGS.train_dir96, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9800:
        checkpoint_path = os.path.join(FLAGS.train_dir97, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==9900:
        checkpoint_path = os.path.join(FLAGS.train_dir98, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      if step==10000:
        checkpoint_path = os.path.join(FLAGS.train_dir99, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      



def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()

  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  if tf.gfile.Exists(FLAGS.train_dir0):
    tf.gfile.DeleteRecursively(FLAGS.train_dir0)
  tf.gfile.MakeDirs(FLAGS.train_dir0)

  if tf.gfile.Exists(FLAGS.train_dir1):
    tf.gfile.DeleteRecursively(FLAGS.train_dir1)
  tf.gfile.MakeDirs(FLAGS.train_dir1)

  if tf.gfile.Exists(FLAGS.train_dir2):
    tf.gfile.DeleteRecursively(FLAGS.train_dir2)
  tf.gfile.MakeDirs(FLAGS.train_dir2)

  if tf.gfile.Exists(FLAGS.train_dir3):
    tf.gfile.DeleteRecursively(FLAGS.train_dir3)
  tf.gfile.MakeDirs(FLAGS.train_dir3)

  if tf.gfile.Exists(FLAGS.train_dir4):
    tf.gfile.DeleteRecursively(FLAGS.train_dir4)
  tf.gfile.MakeDirs(FLAGS.train_dir4)

  if tf.gfile.Exists(FLAGS.train_dir5):
    tf.gfile.DeleteRecursively(FLAGS.train_dir5)
  tf.gfile.MakeDirs(FLAGS.train_dir5)

  if tf.gfile.Exists(FLAGS.train_dir6):
    tf.gfile.DeleteRecursively(FLAGS.train_dir6)
  tf.gfile.MakeDirs(FLAGS.train_dir6)

  if tf.gfile.Exists(FLAGS.train_dir7):
    tf.gfile.DeleteRecursively(FLAGS.train_dir7)
  tf.gfile.MakeDirs(FLAGS.train_dir7)

  if tf.gfile.Exists(FLAGS.train_dir8):
    tf.gfile.DeleteRecursively(FLAGS.train_dir8)
  tf.gfile.MakeDirs(FLAGS.train_dir8)

  if tf.gfile.Exists(FLAGS.train_dir9):
    tf.gfile.DeleteRecursively(FLAGS.train_dir9)
  tf.gfile.MakeDirs(FLAGS.train_dir9)

  if tf.gfile.Exists(FLAGS.train_dir10):
    tf.gfile.DeleteRecursively(FLAGS.train_dir10)
  tf.gfile.MakeDirs(FLAGS.train_dir10)

  if tf.gfile.Exists(FLAGS.train_dir11):
    tf.gfile.DeleteRecursively(FLAGS.train_dir11)
  tf.gfile.MakeDirs(FLAGS.train_dir11)

  if tf.gfile.Exists(FLAGS.train_dir12):
    tf.gfile.DeleteRecursively(FLAGS.train_dir12)
  tf.gfile.MakeDirs(FLAGS.train_dir12)

  if tf.gfile.Exists(FLAGS.train_dir13):
    tf.gfile.DeleteRecursively(FLAGS.train_dir13)
  tf.gfile.MakeDirs(FLAGS.train_dir13)

  if tf.gfile.Exists(FLAGS.train_dir14):
    tf.gfile.DeleteRecursively(FLAGS.train_dir14)
  tf.gfile.MakeDirs(FLAGS.train_dir14)

  if tf.gfile.Exists(FLAGS.train_dir15):
    tf.gfile.DeleteRecursively(FLAGS.train_dir15)
  tf.gfile.MakeDirs(FLAGS.train_dir15)

  if tf.gfile.Exists(FLAGS.train_dir16):
    tf.gfile.DeleteRecursively(FLAGS.train_dir16)
  tf.gfile.MakeDirs(FLAGS.train_dir16)

  if tf.gfile.Exists(FLAGS.train_dir17):
    tf.gfile.DeleteRecursively(FLAGS.train_dir17)
  tf.gfile.MakeDirs(FLAGS.train_dir17)

  if tf.gfile.Exists(FLAGS.train_dir18):
    tf.gfile.DeleteRecursively(FLAGS.train_dir18)
  tf.gfile.MakeDirs(FLAGS.train_dir18)

  if tf.gfile.Exists(FLAGS.train_dir19):
    tf.gfile.DeleteRecursively(FLAGS.train_dir19)
  tf.gfile.MakeDirs(FLAGS.train_dir19)


  if tf.gfile.Exists(FLAGS.train_dir20):
    tf.gfile.DeleteRecursively(FLAGS.train_dir20)
  tf.gfile.MakeDirs(FLAGS.train_dir20)

  if tf.gfile.Exists(FLAGS.train_dir21):
    tf.gfile.DeleteRecursively(FLAGS.train_dir21)
  tf.gfile.MakeDirs(FLAGS.train_dir21)

  if tf.gfile.Exists(FLAGS.train_dir22):
    tf.gfile.DeleteRecursively(FLAGS.train_dir22)
  tf.gfile.MakeDirs(FLAGS.train_dir22)

  if tf.gfile.Exists(FLAGS.train_dir23):
    tf.gfile.DeleteRecursively(FLAGS.train_dir23)
  tf.gfile.MakeDirs(FLAGS.train_dir23)

  if tf.gfile.Exists(FLAGS.train_dir24):
    tf.gfile.DeleteRecursively(FLAGS.train_dir24)
  tf.gfile.MakeDirs(FLAGS.train_dir24)

  if tf.gfile.Exists(FLAGS.train_dir25):
    tf.gfile.DeleteRecursively(FLAGS.train_dir25)
  tf.gfile.MakeDirs(FLAGS.train_dir25)

  if tf.gfile.Exists(FLAGS.train_dir26):
    tf.gfile.DeleteRecursively(FLAGS.train_dir26)
  tf.gfile.MakeDirs(FLAGS.train_dir26)

  if tf.gfile.Exists(FLAGS.train_dir27):
    tf.gfile.DeleteRecursively(FLAGS.train_dir27)
  tf.gfile.MakeDirs(FLAGS.train_dir27)

  if tf.gfile.Exists(FLAGS.train_dir28):
    tf.gfile.DeleteRecursively(FLAGS.train_dir28)
  tf.gfile.MakeDirs(FLAGS.train_dir28)

  if tf.gfile.Exists(FLAGS.train_dir29):
    tf.gfile.DeleteRecursively(FLAGS.train_dir29)
  tf.gfile.MakeDirs(FLAGS.train_dir29)

  if tf.gfile.Exists(FLAGS.train_dir30):
    tf.gfile.DeleteRecursively(FLAGS.train_dir30)
  tf.gfile.MakeDirs(FLAGS.train_dir30)

  if tf.gfile.Exists(FLAGS.train_dir31):
    tf.gfile.DeleteRecursively(FLAGS.train_dir31)
  tf.gfile.MakeDirs(FLAGS.train_dir31)

  if tf.gfile.Exists(FLAGS.train_dir32):
    tf.gfile.DeleteRecursively(FLAGS.train_dir32)
  tf.gfile.MakeDirs(FLAGS.train_dir32)

  if tf.gfile.Exists(FLAGS.train_dir33):
    tf.gfile.DeleteRecursively(FLAGS.train_dir33)
  tf.gfile.MakeDirs(FLAGS.train_dir33)

  if tf.gfile.Exists(FLAGS.train_dir34):
    tf.gfile.DeleteRecursively(FLAGS.train_dir34)
  tf.gfile.MakeDirs(FLAGS.train_dir34)

  if tf.gfile.Exists(FLAGS.train_dir35):
    tf.gfile.DeleteRecursively(FLAGS.train_dir35)
  tf.gfile.MakeDirs(FLAGS.train_dir35)

  if tf.gfile.Exists(FLAGS.train_dir36):
    tf.gfile.DeleteRecursively(FLAGS.train_dir36)
  tf.gfile.MakeDirs(FLAGS.train_dir36)

  if tf.gfile.Exists(FLAGS.train_dir37):
    tf.gfile.DeleteRecursively(FLAGS.train_dir37)
  tf.gfile.MakeDirs(FLAGS.train_dir37)

  if tf.gfile.Exists(FLAGS.train_dir38):
    tf.gfile.DeleteRecursively(FLAGS.train_dir38)
  tf.gfile.MakeDirs(FLAGS.train_dir38)

  if tf.gfile.Exists(FLAGS.train_dir39):
    tf.gfile.DeleteRecursively(FLAGS.train_dir39)
  tf.gfile.MakeDirs(FLAGS.train_dir39)


  if tf.gfile.Exists(FLAGS.train_dir40):
    tf.gfile.DeleteRecursively(FLAGS.train_dir40)
  tf.gfile.MakeDirs(FLAGS.train_dir40)

  if tf.gfile.Exists(FLAGS.train_dir41):
    tf.gfile.DeleteRecursively(FLAGS.train_dir41)
  tf.gfile.MakeDirs(FLAGS.train_dir41)

  if tf.gfile.Exists(FLAGS.train_dir42):
    tf.gfile.DeleteRecursively(FLAGS.train_dir42)
  tf.gfile.MakeDirs(FLAGS.train_dir42)

  if tf.gfile.Exists(FLAGS.train_dir43):
    tf.gfile.DeleteRecursively(FLAGS.train_dir43)
  tf.gfile.MakeDirs(FLAGS.train_dir43)

  if tf.gfile.Exists(FLAGS.train_dir44):
    tf.gfile.DeleteRecursively(FLAGS.train_dir44)
  tf.gfile.MakeDirs(FLAGS.train_dir44)

  if tf.gfile.Exists(FLAGS.train_dir45):
    tf.gfile.DeleteRecursively(FLAGS.train_dir45)
  tf.gfile.MakeDirs(FLAGS.train_dir45)

  if tf.gfile.Exists(FLAGS.train_dir46):
    tf.gfile.DeleteRecursively(FLAGS.train_dir46)
  tf.gfile.MakeDirs(FLAGS.train_dir46)

  if tf.gfile.Exists(FLAGS.train_dir47):
    tf.gfile.DeleteRecursively(FLAGS.train_dir47)
  tf.gfile.MakeDirs(FLAGS.train_dir47)

  if tf.gfile.Exists(FLAGS.train_dir48):
    tf.gfile.DeleteRecursively(FLAGS.train_dir48)
  tf.gfile.MakeDirs(FLAGS.train_dir48)

  if tf.gfile.Exists(FLAGS.train_dir49):
    tf.gfile.DeleteRecursively(FLAGS.train_dir49)
  tf.gfile.MakeDirs(FLAGS.train_dir49)

  if tf.gfile.Exists(FLAGS.train_dir50):
    tf.gfile.DeleteRecursively(FLAGS.train_dir50)
  tf.gfile.MakeDirs(FLAGS.train_dir50)

  if tf.gfile.Exists(FLAGS.train_dir51):
    tf.gfile.DeleteRecursively(FLAGS.train_dir51)
  tf.gfile.MakeDirs(FLAGS.train_dir51)

  if tf.gfile.Exists(FLAGS.train_dir52):
    tf.gfile.DeleteRecursively(FLAGS.train_dir52)
  tf.gfile.MakeDirs(FLAGS.train_dir52)

  if tf.gfile.Exists(FLAGS.train_dir53):
    tf.gfile.DeleteRecursively(FLAGS.train_dir53)
  tf.gfile.MakeDirs(FLAGS.train_dir53)

  if tf.gfile.Exists(FLAGS.train_dir54):
    tf.gfile.DeleteRecursively(FLAGS.train_dir54)
  tf.gfile.MakeDirs(FLAGS.train_dir54)

  if tf.gfile.Exists(FLAGS.train_dir55):
    tf.gfile.DeleteRecursively(FLAGS.train_dir55)
  tf.gfile.MakeDirs(FLAGS.train_dir55)

  if tf.gfile.Exists(FLAGS.train_dir56):
    tf.gfile.DeleteRecursively(FLAGS.train_dir56)
  tf.gfile.MakeDirs(FLAGS.train_dir56)

  if tf.gfile.Exists(FLAGS.train_dir57):
    tf.gfile.DeleteRecursively(FLAGS.train_dir57)
  tf.gfile.MakeDirs(FLAGS.train_dir57)

  if tf.gfile.Exists(FLAGS.train_dir58):
    tf.gfile.DeleteRecursively(FLAGS.train_dir58)
  tf.gfile.MakeDirs(FLAGS.train_dir58)

  if tf.gfile.Exists(FLAGS.train_dir59):
    tf.gfile.DeleteRecursively(FLAGS.train_dir59)
  tf.gfile.MakeDirs(FLAGS.train_dir59)

  if tf.gfile.Exists(FLAGS.train_dir60):
    tf.gfile.DeleteRecursively(FLAGS.train_dir60)
  tf.gfile.MakeDirs(FLAGS.train_dir60)

  if tf.gfile.Exists(FLAGS.train_dir61):
    tf.gfile.DeleteRecursively(FLAGS.train_dir61)
  tf.gfile.MakeDirs(FLAGS.train_dir61)

  if tf.gfile.Exists(FLAGS.train_dir62):
    tf.gfile.DeleteRecursively(FLAGS.train_dir62)
  tf.gfile.MakeDirs(FLAGS.train_dir62)

  if tf.gfile.Exists(FLAGS.train_dir63):
    tf.gfile.DeleteRecursively(FLAGS.train_dir63)
  tf.gfile.MakeDirs(FLAGS.train_dir63)

  if tf.gfile.Exists(FLAGS.train_dir64):
    tf.gfile.DeleteRecursively(FLAGS.train_dir64)
  tf.gfile.MakeDirs(FLAGS.train_dir64)

  if tf.gfile.Exists(FLAGS.train_dir65):
    tf.gfile.DeleteRecursively(FLAGS.train_dir65)
  tf.gfile.MakeDirs(FLAGS.train_dir65)

  if tf.gfile.Exists(FLAGS.train_dir66):
    tf.gfile.DeleteRecursively(FLAGS.train_dir66)
  tf.gfile.MakeDirs(FLAGS.train_dir66)

  if tf.gfile.Exists(FLAGS.train_dir67):
    tf.gfile.DeleteRecursively(FLAGS.train_dir67)
  tf.gfile.MakeDirs(FLAGS.train_dir67)

  if tf.gfile.Exists(FLAGS.train_dir68):
    tf.gfile.DeleteRecursively(FLAGS.train_dir68)
  tf.gfile.MakeDirs(FLAGS.train_dir68)

  if tf.gfile.Exists(FLAGS.train_dir69):
    tf.gfile.DeleteRecursively(FLAGS.train_dir69)
  tf.gfile.MakeDirs(FLAGS.train_dir69)

  if tf.gfile.Exists(FLAGS.train_dir70):
    tf.gfile.DeleteRecursively(FLAGS.train_dir70)
  tf.gfile.MakeDirs(FLAGS.train_dir70)

  if tf.gfile.Exists(FLAGS.train_dir71):
    tf.gfile.DeleteRecursively(FLAGS.train_dir71)
  tf.gfile.MakeDirs(FLAGS.train_dir71)

  if tf.gfile.Exists(FLAGS.train_dir72):
    tf.gfile.DeleteRecursively(FLAGS.train_dir72)
  tf.gfile.MakeDirs(FLAGS.train_dir72)

  if tf.gfile.Exists(FLAGS.train_dir73):
    tf.gfile.DeleteRecursively(FLAGS.train_dir73)
  tf.gfile.MakeDirs(FLAGS.train_dir73)

  if tf.gfile.Exists(FLAGS.train_dir74):
    tf.gfile.DeleteRecursively(FLAGS.train_dir74)
  tf.gfile.MakeDirs(FLAGS.train_dir74)

  if tf.gfile.Exists(FLAGS.train_dir75):
    tf.gfile.DeleteRecursively(FLAGS.train_dir75)
  tf.gfile.MakeDirs(FLAGS.train_dir75)

  if tf.gfile.Exists(FLAGS.train_dir76):
    tf.gfile.DeleteRecursively(FLAGS.train_dir76)
  tf.gfile.MakeDirs(FLAGS.train_dir76)

  if tf.gfile.Exists(FLAGS.train_dir77):
    tf.gfile.DeleteRecursively(FLAGS.train_dir77)
  tf.gfile.MakeDirs(FLAGS.train_dir77)

  if tf.gfile.Exists(FLAGS.train_dir78):
    tf.gfile.DeleteRecursively(FLAGS.train_dir78)
  tf.gfile.MakeDirs(FLAGS.train_dir78)

  if tf.gfile.Exists(FLAGS.train_dir79):
    tf.gfile.DeleteRecursively(FLAGS.train_dir79)
  tf.gfile.MakeDirs(FLAGS.train_dir79)

  if tf.gfile.Exists(FLAGS.train_dir80):
    tf.gfile.DeleteRecursively(FLAGS.train_dir80)
  tf.gfile.MakeDirs(FLAGS.train_dir80)

  if tf.gfile.Exists(FLAGS.train_dir81):
    tf.gfile.DeleteRecursively(FLAGS.train_dir81)
  tf.gfile.MakeDirs(FLAGS.train_dir81)

  if tf.gfile.Exists(FLAGS.train_dir82):
    tf.gfile.DeleteRecursively(FLAGS.train_dir82)
  tf.gfile.MakeDirs(FLAGS.train_dir82)

  if tf.gfile.Exists(FLAGS.train_dir83):
    tf.gfile.DeleteRecursively(FLAGS.train_dir83)
  tf.gfile.MakeDirs(FLAGS.train_dir83)

  if tf.gfile.Exists(FLAGS.train_dir84):
    tf.gfile.DeleteRecursively(FLAGS.train_dir84)
  tf.gfile.MakeDirs(FLAGS.train_dir84)

  if tf.gfile.Exists(FLAGS.train_dir85):
    tf.gfile.DeleteRecursively(FLAGS.train_dir85)
  tf.gfile.MakeDirs(FLAGS.train_dir85)

  if tf.gfile.Exists(FLAGS.train_dir86):
    tf.gfile.DeleteRecursively(FLAGS.train_dir86)
  tf.gfile.MakeDirs(FLAGS.train_dir86)

  if tf.gfile.Exists(FLAGS.train_dir87):
    tf.gfile.DeleteRecursively(FLAGS.train_dir87)
  tf.gfile.MakeDirs(FLAGS.train_dir87)

  if tf.gfile.Exists(FLAGS.train_dir88):
    tf.gfile.DeleteRecursively(FLAGS.train_dir88)
  tf.gfile.MakeDirs(FLAGS.train_dir88)

  if tf.gfile.Exists(FLAGS.train_dir89):
    tf.gfile.DeleteRecursively(FLAGS.train_dir89)
  tf.gfile.MakeDirs(FLAGS.train_dir89)

  if tf.gfile.Exists(FLAGS.train_dir90):
    tf.gfile.DeleteRecursively(FLAGS.train_dir90)
  tf.gfile.MakeDirs(FLAGS.train_dir90)

  if tf.gfile.Exists(FLAGS.train_dir91):
    tf.gfile.DeleteRecursively(FLAGS.train_dir91)
  tf.gfile.MakeDirs(FLAGS.train_dir91)

  if tf.gfile.Exists(FLAGS.train_dir92):
    tf.gfile.DeleteRecursively(FLAGS.train_dir92)
  tf.gfile.MakeDirs(FLAGS.train_dir92)

  if tf.gfile.Exists(FLAGS.train_dir93):
    tf.gfile.DeleteRecursively(FLAGS.train_dir93)
  tf.gfile.MakeDirs(FLAGS.train_dir93)

  if tf.gfile.Exists(FLAGS.train_dir94):
    tf.gfile.DeleteRecursively(FLAGS.train_dir94)
  tf.gfile.MakeDirs(FLAGS.train_dir94)

  if tf.gfile.Exists(FLAGS.train_dir95):
    tf.gfile.DeleteRecursively(FLAGS.train_dir95)
  tf.gfile.MakeDirs(FLAGS.train_dir95)

  if tf.gfile.Exists(FLAGS.train_dir96):
    tf.gfile.DeleteRecursively(FLAGS.train_dir96)
  tf.gfile.MakeDirs(FLAGS.train_dir96)

  if tf.gfile.Exists(FLAGS.train_dir97):
    tf.gfile.DeleteRecursively(FLAGS.train_dir97)
  tf.gfile.MakeDirs(FLAGS.train_dir97)

  if tf.gfile.Exists(FLAGS.train_dir98):
    tf.gfile.DeleteRecursively(FLAGS.train_dir98)
  tf.gfile.MakeDirs(FLAGS.train_dir98)

  if tf.gfile.Exists(FLAGS.train_dir99):
    tf.gfile.DeleteRecursively(FLAGS.train_dir99)
  tf.gfile.MakeDirs(FLAGS.train_dir99)

  train()


if __name__ == '__main__':
  tf.app.run()