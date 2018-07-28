from processing.data_reader import DataReader
import tensorflow as tf
import numpy as np
import cv2 as cv
import time
import argparse
import os
import json


class ModelEvaler:
    def __init__(self, config):
        self.reader = DataReader(config['reader'])
        self.config = config['evaler']

    def eval(self, show_img):
        def resolve(tensor):
            """
            tensor is numpy array, not tf tensor
            """
            result = [
                np.argmax(tensor[0]),
                np.argmax(tensor[1]),
                np.argmax(tensor[2]),
                np.argmax(tensor[3])
            ]
            return result

        hit, miss = 0, 0
        imgs, labels = self.reader.get_data('eval')
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.config['meta_path'])
            saver.restore(
                sess, tf.train.latest_checkpoint(self.config['model_path']))
            graph = tf.get_default_graph()
            data_in = graph.get_tensor_by_name('data_in/data_in:0')
            logits = graph.get_tensor_by_name('logits_layer/Reshape:0')
            for idx in range(len(imgs)):
                start_time = time.time()
                pred_img = np.reshape(imgs[idx], (1, 50, 200, 3))
                pred = sess.run(logits, feed_dict={
                    data_in: pred_img
                })
                pred = pred[0]
                label = resolve(labels[idx])
                pred = resolve(pred)
                if pred == label:
                    hit += 1
                else:
                    miss += 1
                accu = hit / (hit + miss)
                print('Complete image %i predict, spend %fs, accu %f' %
                      (idx, (time.time() - start_time), accu))
                print(label)
                print(pred)
                if show_img:
                    cv.imshow('img', imgs[idx])
                    cv.waitKey()
        print('Hit: %i, Miss: %i, Accu: %f' % (hit, miss, accu))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model evaluator')
    parser.add_argument('--show', type=bool, default=False,
                        help='Restore pre-trained model sotre in config')
    args = parser.parse_args()
    config_path = '%s/config.json' % os.path.dirname(os.path.abspath(__file__))
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
    evaler = ModelEvaler(config['processing'])
    evaler.eval(args.show)
