import tensorflow as tf


class ConvModel:
    def __init__(self, config):
        self.config = config

    def build(self, data_in, labels):
        # pred - dict of softmax results
        # logits - output layer
        pred, logits = self.__build_model(data_in)
        train_op, loss = self.__build_train_op(logits, labels)
        return train_op, loss, logits

    def __build_model(self, data_in):
        filters = self.config['conv_base_filters']
        with tf.variable_scope('conv_blk'):
            conv1_1 = self.__get_conv_layer(data_in, filters, 'conv1_1')
            conv1_2 = self.__get_conv_layer(conv1_1, filters, 'conv1_2')
            pool1 = self.__get_pool_layer(conv1_2, 'pool1')
            filters = filters * 2
            conv2_1 = self.__get_conv_layer(pool1, filters, 'conv2_1')
            conv2_2 = self.__get_conv_layer(conv2_1, filters, 'conv2_2')
            pool2 = self.__get_pool_layer(conv2_2, 'pool2')

        # fc layer
        with tf.variable_scope('dense_layer'):

            pool2_flat = tf.reshape(pool2, [-1, 76800])  # equal to (1, 76800)
            dense = tf.layers.dense(
                inputs=pool2_flat,
                units=self.config['fc_neu_num'],
                activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                inputs=dense, rate=self.config['keep_prob'])

        # output layer
        with tf.variable_scope('logits_layer'):

            """ why the shape of softmax ?  the number of labels """

            logits = tf.layers.dense(
                inputs=dropout, units=self.config['label_class'] * 4)
            logits = tf.reshape(logits, [-1, 4, self.config['label_class']])
            pred = {
                "result": logits
            }
        return pred, logits

    def __build_train_op(self, logits, labels):
        with tf.name_scope('loss'):
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config['learn_rate'])
            train_op = optimizer.minimize(loss=loss, name='minimize')
        return train_op, loss

    def __get_conv_layer(self, data_in, filters, name,
                         kernel_size=[3, 3], drop=False, pad=0):

        with tf.variable_scope(name):
            conv = tf.layers.conv2d(
                inputs=data_in,
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=tf.nn.relu6
            )
            if drop:
                conv = tf.layers.dropout(conv, rate=self.config['keep_prob'])
            return conv

    def __get_pool_layer(self, data_in, name):
        """
        output_size = [(n + 2p - f) / s] + 1

        n = input_size
        """
        return tf.layers.max_pooling2d(
            inputs=data_in, pool_size=[2, 2], strides=2, name=name)
