from pprint import pprint
import tensorflow as tf


sess = tf.Session()
base_path = '/Users/pzn666/Documents/data_enlight/projects/text_detector \
            /captcha/saved_models/saved_model'
model_path = base_path + '/model.ckpt.meta'
saver = tf.train.import_meta_graph(model_path)
saver.restore(sess, tf.train.latest_checkpoint(base_path))


graph = tf.get_default_graph()
tensors = [n.name for n in graph.as_graph_def().node]
pprint(tensors)
