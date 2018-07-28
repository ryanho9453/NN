from pprint import pprint
import tensorflow as tf
import numpy as np
import cv2 as cv
import time
import argparse
import os
import json


class ModelPredictor:
    def __init__(self):
        config_path = '%s/config.json' % os.path.dirname(
            os.path.abspath(__file__))
        with open(config_path, 'r') as f:
            config = json.loads(f.read())
        self.config = config['processing']['predictor']

    def pred(self, img):
        """
        img: a numpy array with shape(50, 200, 3)
        """
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

        img = [img]
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.config['meta_path'])
            saver.restore(
                sess, tf.train.latest_checkpoint(self.config['model_path']))
            graph = tf.get_default_graph()
            data_in = graph.get_tensor_by_name('data_in/data_in:0')
            logits = graph.get_tensor_by_name('logits_layer/Reshape:0')
            pred_img = np.reshape(img, (1, 50, 200, 3))
            pred = sess.run(logits, feed_dict={
                data_in: img
            })
            pred = resolve(pred[0])
        return pred
