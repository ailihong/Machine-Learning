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
  train()


if __name__ == '__main__':
  tf.app.run()