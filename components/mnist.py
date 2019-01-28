import tensorflow as tf

from .component import Component
from .records import NumericRecordIterator
from .simple import ImageResizer


class MnistNetwork(Component):
    name = 'MNIST Network'

    def _run(self, image, training_flag):
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            network = tf.layers.conv2d(image, filters=64, kernel_size=5, strides=2, padding='same')
            network = tf.layers.batch_normalization(network, training=training_flag)
            network = tf.layers.dropout(network, rate=.1, training=training_flag)
            network = tf.nn.relu(network)
            network = tf.layers.conv2d(network, filters=128, kernel_size=5, strides=2, padding='same')
            network = tf.layers.batch_normalization(network, training=training_flag)
            network = tf.layers.dropout(network, rate=.3, training=training_flag)
            network = tf.nn.relu(network)
            network = tf.layers.conv2d(network, filters=256, kernel_size=5, strides=2, padding='same')
            network = tf.layers.batch_normalization(network, training=training_flag)
            network = tf.layers.dropout(network, rate=.5, training=training_flag)
            network = tf.nn.relu(network)
            network = tf.layers.flatten(network)
            network = tf.layers.dense(network, 512)
            network = tf.layers.batch_normalization(network, training=training_flag)
            network = tf.nn.relu(network)
            network = tf.layers.dropout(network, rate=.5, training=training_flag)
            network = tf.layers.dense(network, 10)
        return network


class CrossEntropyLoss(Component):
    name = 'Cross Entropy Loss'

    @staticmethod
    def _run(logit, label):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label))


class ClassificationAccuracy(Component):
    name = 'Classification Accuracy'

    @staticmethod
    def _run(logit, label):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, axis=1), label), dtype=tf.float32))


class MnistRecordToTensor(Component):
    name = 'MNIST Record -> Tensor'

    def __init__(self, size):
        super().__init__()
        size = list(size) + [3]
        self.components['resizer'] = ImageResizer(size=size, resize_method='nothing')

    def _run(self, record):
        features = {
            'image': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64),
        }
        parsed_features = tf.parse_single_example(record, features)
        image = tf.to_float(tf.image.decode_png(parsed_features['image']))
        image = self.components['resizer'].run(image)
        return image, parsed_features['label']


class MnistModel(Component):
    name = 'Mnist Model'

    def __init__(self, components, n_processes):
        """
        :param components: {'record_to_tensor': Component, 'preprocessor': Component, 'network': Component}
        """
        super().__init__(components=components)
        self.n_processes = n_processes
        self.components['loss'] = CrossEntropyLoss()
        self.components['accuracy'] = ClassificationAccuracy()

    def _run(self, path, training_flag, batch_size):
        image, label = NumericRecordIterator(
            components={'record_to_tensor': self.components['record_to_tensor']},
            n_processes=self.n_processes, skip_size=0, repeat_flag=True, shuffle_flag=training_flag,
            batch_size=batch_size
        ).run(path)
        image = self.components['preprocessor'].run(image)
        logit = self.components['network'].run(image, training_flag=training_flag)
        return [
            tf.identity(self.components['loss'].run(label=label, logit=logit), name='loss'),
            tf.identity(self.components['accuracy'].run(label=label, logit=logit), name='accuracy'),
        ]
