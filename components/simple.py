import tensorflow as tf
from .component import Component

from .records import TfRecordIterator
from utils import substrings_to_re


class DatasetSizeCounter(Component):
    name = 'Dataset Size Counter'

    def __init__(self, path):
        super().__init__(log_flag=True)
        self.path = path

    def _run(self):
        size = 0

        for records in TfRecordIterator(path=self.path, batch_size=1).run():
            size += len(records)

        self.log(str(size))
        return size


class Initializer(Component):
    name = 'Initializer'

    @staticmethod
    def _run(session):
        session.run(tf.global_variables_initializer())


class CheckPointInitializer(Component):
    name = 'Checkpoint Initializer'

    def __init__(self, path, include_scope, exclude_scope):
        super().__init__()
        self.path = path
        self.include_scope = include_scope
        self.exclude_scope = exclude_scope

    def _run(self, session):
        session.run(tf.global_variables_initializer())
        tf.train.Saver(tf.global_variables(
            scope=substrings_to_re(include=self.include_scope, exclude=self.exclude_scope)
        )).restore(session, self.path)


class Optimizer(Component):
    name = 'Optimizer'

    def __init__(self, tensor_name, learning_rate, decay_step):
        super().__init__()
        self.tensor_name = tensor_name
        self.learning_rate = learning_rate
        self.decay_step = decay_step

    def _run(self, session=None, metrics=None):
        if session is None:
            global_step = tf.train.get_global_step()
            learning_rate = LearningRate(decay_step=self.decay_step).run(
                learning_rate=self.learning_rate, global_step=global_step
            )
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tensor = tf.get_default_graph().get_tensor_by_name(self.tensor_name)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.operation = optimizer.minimize(tensor)
            self.summary = SummaryMaker().run(metrics, prefix='training')
        else:
            values = session.run([self.operation] + metrics + [self.summary])
            return values[1:-1], values[-1:]


class Validator(Component):
    name = 'Validator'

    def _run(self, session=None, metrics=None):
        if session is None:
            self.summary = SummaryMaker().run(metrics, prefix='validation')
        else:
            values = session.run(metrics + [self.summary])
            return values[:-1], values[-1:]


class NameGetter(Component):
    name = 'Tensors -> Names'

    @staticmethod
    def _run(tensors):
        return [tensor.name.split('/')[-1].split(':')[0] for tensor in tensors]


class SummaryMaker(Component):
    name = 'Tensors -> Summary'

    @staticmethod
    def _run(tensors, prefix):
        summaries = []
        for tensor, name in zip(tensors, NameGetter().run(tensors)):
            if not len(tensor.shape):
                summaries.append(tf.summary.scalar(name, tensor, family=prefix))
            else:
                summaries.append(tf.summary.image(name, tensor, max_outputs=1, family=prefix))
        return tf.summary.merge(summaries)


class ImageResizer(Component):
    name = 'Image Resizer'

    def __init__(self, size, resize_method):
        """
        :param size: tuple(int, int); size of the output image
        :param resize_method: str; 'random_crop', 'crop_or_pad', 'nearest_neighbor' or 'nothing'
        """
        super().__init__()
        self.size = size
        self.resize_method = resize_method

    def _run(self, image):
        if self.resize_method == 'random_crop':
            image = tf.random_crop(image, size=(self.size[0], self.size[1], self.size[2]))
        elif self.resize_method == 'crop_or_pad':
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_image_with_crop_or_pad(image, target_height=self.size[0], target_width=self.size[1])
            image = tf.squeeze(image, axis=0)
        elif self.resize_method == 'nearest_neighbor':
            image = tf.expand_dims(image, axis=0)
            image = tf.image.resize_nearest_neighbor(image, size=self.size, align_corners=True)
            image = tf.squeeze(image, axis=0)
        else:  # self.resize_method == 'nothing'
            image = tf.reshape(image, (self.size[0], self.size[1], self.size[2]))
        return image


class LearningRate(Component):
    name = 'Learning Rate'

    def __init__(self, decay_step):
        super().__init__()
        self.decay_step = decay_step

    def _run(self, learning_rate, global_step):
        if self.decay_step:
            learning_rate = tf.divide(learning_rate, tf.pow(
                2.0, tf.cast(tf.div(global_step, self.decay_step), tf.float32)
            ))
        return learning_rate


class Preprocessor(Component):
    name = 'Preprocessor'

    @staticmethod
    def _run(image):
        return (image - 128.0) / 128.0
