from processing.data_reader import DataReader
from models.model_picker import ModelPicker
import tensorflow as tf
import numpy as np
import json
import os
import time
import argparse


def train(restore):
    # reading config and init modules
    config_path = '%s/config.json' % os.path.dirname(os.path.abspath(__file__))
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
    model_picker = ModelPicker(config['models'])
    reader = DataReader(config['processing']['reader'])

    with tf.name_scope('data_in'):
        if config['models']['choose_model'] == 'conv':
            data_in = tf.placeholder(
                tf.float32, [None, None, None, 3], name='data_in')
    with tf.name_scope('label_in'):
        label_in = tf.placeholder(
            tf.float32, [None, 4, config['general']['label_class']],
            name='label_in')

    with tf.Session() as sess:
        model = model_picker.pick_model()
        if not restore:
            train_op, loss, logits = model.build(data_in, label_in)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
        else:
            saved_model_path = config['general']['saved_model_path']
            saved_meta_path = saved_model_path + 'model.ckpt.meta'
            print('Restore saved model which stored at %s' % saved_meta_path)
            train_op, loss, logits = model.build(data_in, label_in)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(
                sess, tf.train.latest_checkpoint(saved_model_path))
        loss_summary = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(
            config['general']['tensor_log'], sess.graph)
        total_sec = 0
        # train
        for step in range(1, config['general']['steps'] + 1):
            start_time = time.time()
            # prepare datas
            train_data, train_labels = reader.next_batch('train')
            # optimize
            sess.run(train_op, feed_dict={
                        data_in: train_data,
                        label_in: train_labels
                    })

            if step // 100 != 0 and step % 100 == 0 or step == 1:
                train_loss, loss_summ, prediction = sess.run(
                        [loss, loss_summary, logits],
                        feed_dict={
                            data_in: train_data,
                            label_in: train_labels
                        })
                writer.add_summary(loss_summ, step)
                saver.save(
                    sess, config['general']['model_path'],
                    global_step=tf.train.get_global_step())
                print('Save model at step = %s' % (step))
                print('loss = %s, step = %s (%s sec)'
                      % (train_loss, step, total_sec))

                print('logits[0]')
                print(prediction[0])
                print('labels[0]')
                print(train_labels[0])
                total_sec = 0
                # Early stop
                need_early_stop = config['general']['early_stop']
                early_stop_loss = config['general']['ealry_stop_loss']
                if need_early_stop and train_loss < early_stop_loss:
                    print('Early stop at loss %f' % train_loss)
                    break
            total_sec += time.time() - start_time
        saver.save(sess, config['general']['model_path'])
        print('Save model at final step = %s, spend %fs' % (step, total_sec))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model runner')
    parser.add_argument('--restore', type=bool, default=True,
                        help='Restore pre-trained model sotre in config')
    args = parser.parse_args()
    train(args.restore)
