import os
import numpy as np
import tensorflow as tf

from .component import Component
from .records import NumericRecordIterator
from .mnist import CrossEntropyLoss, ClassificationAccuracy
from .simple import SummaryMaker, LearningRate
from utils import substrings_to_re


class CycadaModel(Component):
    name = 'CyCADA Model'

    def __init__(self, components, n_processes):
        """
        :param components: {
            'record_to_tensor': Component, 'preprocessor': Component, 'network': Component, 'generator': Component,
            'discriminator': Component,
        }
        """
        super().__init__(components=components)
        self.n_processes = n_processes
        self.components['loss'] = CrossEntropyLoss()
        self.components['accuracy'] = ClassificationAccuracy()

    def _run(self, path, training_flag, batch_size):
        self.components['iterator'] = NumericRecordIterator(
            components={'record_to_tensor': self.components['record_to_tensor']},
            n_processes=self.n_processes, skip_size=0, repeat_flag=True, shuffle_flag=training_flag,
            batch_size=batch_size
        )
        source_path = os.path.join(path, 'source')
        source_image, source_label = self.components['iterator'].run(source_path)
        source_image = self.components['preprocessor'].run(source_image)
        source_image = tf.identity(source_image, name='source_image')
        target_path = os.path.join(path, 'target')
        target_image, target_label = self.components['iterator'].run(target_path)
        target_image = self.components['preprocessor'].run(target_image)
        target_image = tf.identity(target_image, name='target_image')

        # source image cycles
        with tf.variable_scope('source/generator', reuse=tf.AUTO_REUSE):
            source_target_image = self.components['generator'].run(source_image, training_flag=training_flag)
        source_target_image = tf.identity(source_target_image, name='source_target_image')
        with tf.variable_scope('target/generator', reuse=tf.AUTO_REUSE):
            source_target_source_image = self.components['generator'].run(
                source_target_image, training_flag=training_flag
            )
        source_target_source_image = tf.identity(source_target_source_image, name='source_target_source_image')
        with tf.variable_scope('source', reuse=tf.AUTO_REUSE):
            source_image_source_prediction = self.components['network'].run(image=source_image, training_flag=False)
        with tf.variable_scope('source', reuse=tf.AUTO_REUSE):
            source_target_image_source_prediction = self.components['network'].run(
                image=source_target_image, training_flag=False
            )

        # target image cycles
        with tf.variable_scope('target/generator', reuse=tf.AUTO_REUSE):
            target_source_image = self.components['generator'].run(target_image, training_flag=training_flag)
        target_source_image = tf.identity(target_source_image, name='target_source_image')
        with tf.variable_scope('source/generator', reuse=tf.AUTO_REUSE):
            target_source_target_image = self.components['generator'].run(
                target_source_image, training_flag=training_flag
            )
        target_source_target_image = tf.identity(target_source_target_image, name='target_source_target_image')
        with tf.variable_scope('source', reuse=tf.AUTO_REUSE):
            target_image_source_prediction = self.components['network'].run(
                image=target_image, training_flag=False
            )
        with tf.variable_scope('source', reuse=tf.AUTO_REUSE):
            target_source_image_source_prediction = self.components['network'].run(
                image=target_source_image, training_flag=False
            )

        # discriminators
        with tf.variable_scope('source/discriminator', reuse=tf.AUTO_REUSE):
            source_logit = self.components['discriminator'].run(source_image, training_flag=training_flag)
        with tf.variable_scope('source/discriminator', reuse=tf.AUTO_REUSE):
            target_source_logit = self.components['discriminator'].run(target_source_image, training_flag=training_flag)
        with tf.variable_scope('target/discriminator', reuse=tf.AUTO_REUSE):
            target_logit = self.components['discriminator'].run(target_image, training_flag=training_flag)
        with tf.variable_scope('target/discriminator', reuse=tf.AUTO_REUSE):
            source_target_logit = self.components['discriminator'].run(source_target_image, training_flag=training_flag)
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            source_target_image_target_prediction = self.components['network'].run(
                image=source_target_image, training_flag=training_flag
            )
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            target_image_target_prediction = self.components['network'].run(image=target_image, training_flag=False)

        # losses
        source_cycle_loss = self._cycle_loss(real=source_image, fake=source_target_source_image)
        target_cycle_loss = self._cycle_loss(real=target_image, fake=target_source_target_image)
        source_consistensy_task_loss = self.components['loss'].run(
            label=tf.argmax(source_image_source_prediction, axis=1),
            logit=source_target_image_source_prediction
        )
        target_consistensy_task_loss = self.components['loss'].run(
            label=tf.argmax(target_image_source_prediction, axis=1),
            logit=target_source_image_source_prediction
        )
        source_target_task_loss = tf.identity(self.components['loss'].run(
            label=source_label,
            logit=source_target_image_target_prediction
        ), name='source_target_task_loss')
        target_task_loss = tf.identity(self.components['loss'].run(
            label=target_label,
            logit=target_image_target_prediction
        ), name='target_task_loss')
        source_positive_loss = self._gan_loss(logit=source_logit, label=1)
        target_positive_loss = self._gan_loss(logit=target_logit, label=1)
        source_target_positive_loss = self._gan_loss(logit=source_target_logit, label=1)
        target_source_positive_loss = self._gan_loss(logit=target_source_logit, label=1)
        source_target_negative_loss = self._gan_loss(logit=source_target_logit, label=0)
        target_source_negative_loss = self._gan_loss(logit=target_source_logit, label=0)

        generator_loss = tf.add_n((
            source_cycle_loss, target_cycle_loss, source_target_positive_loss, target_source_positive_loss,
            source_consistensy_task_loss, target_consistensy_task_loss, source_target_task_loss
        ), name='generator_loss')
        discriminator_loss = tf.multiply(tf.add_n((
            source_positive_loss, target_positive_loss, source_target_negative_loss, target_source_negative_loss
        )), .25, name='discriminator_loss')

        return (
            generator_loss, source_target_task_loss, target_task_loss,
            tf.identity(
                self.components['accuracy'].run(label=source_label, logit=source_target_image_target_prediction),
                name='source_target_accuracy'
            ),
            tf.identity(
                self.components['accuracy'].run(label=target_label, logit=target_image_target_prediction),
                name='target_accuracy'
            ),
            tf.multiply(tf.add_n((source_cycle_loss, target_cycle_loss,)), .5, name='cycle_loss'),
            discriminator_loss
        )

    @staticmethod
    def _cycle_loss(real, fake):
        return tf.reduce_mean(tf.abs(tf.subtract(real, fake)), name='cycle_loss')

    @staticmethod
    def _gan_loss(logit, label):
        label = tf.ones_like(logit) if label else tf.zeros_like(logit)
        return tf.reduce_mean(tf.square(tf.subtract(logit, label)), name='gan_loss')


class CycadaGenerator(Component):
    name = 'CyCADA Generator'

    def _run(self, image, training_flag):
        network = self._convolution_block(image, n_filters=64, kernel_size=7, stride=1, activation=tf.nn.relu)
        network = self._convolution_block(network, n_filters=128, kernel_size=3, stride=2, activation=tf.nn.relu)
        residual = self._convolution_block(network, n_filters=256, kernel_size=3, stride=2, activation=tf.nn.relu)
        network = self._convolution_block(residual, n_filters=256, kernel_size=3, stride=1, activation=tf.nn.relu)
        network = self._convolution_block(network, n_filters=256, kernel_size=3, stride=1, activation=None)
        residual += network
        network = self._convolution_block(residual, n_filters=256, kernel_size=3, stride=1, activation=tf.nn.relu)
        network = self._convolution_block(network, n_filters=256, kernel_size=3, stride=1, activation=None)
        network += residual
        network = self._deconvolution_block(
            network, n_filters=128, kernel_size=3, stride=2, normalization_flag=True, activation=tf.nn.relu
        )
        network = self._deconvolution_block(
            network, n_filters=64, kernel_size=3, stride=2, normalization_flag=True, activation=tf.nn.relu
        )
        return self._deconvolution_block(
            network, n_filters=3, kernel_size=7, stride=1, normalization_flag=False, activation=tf.nn.tanh
        )

    @staticmethod
    def _convolution_block(network, n_filters, kernel_size, stride, activation):
        padding_size = kernel_size // 2
        network = tf.pad(
            network, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='symmetric'
        )
        network = tf.layers.conv2d(
            network, filters=n_filters, kernel_size=kernel_size, strides=stride, padding='valid'
        )
        return tf.contrib.layers.instance_norm(network, activation_fn=activation)

    @staticmethod
    def _deconvolution_block(network, n_filters, kernel_size, stride, normalization_flag, activation):
        if stride == 2:
            network = tf.keras.layers.UpSampling2D()(network)
        padding_size = kernel_size // 2
        network = tf.pad(
            network, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], mode='symmetric'
        )
        network = tf.layers.conv2d(
            network, filters=n_filters, kernel_size=kernel_size, strides=1, padding='valid'
        )
        if normalization_flag:
            network = tf.contrib.layers.instance_norm(network)
        return activation(network)


class CycadaDiscriminator(Component):
    name = 'CyCADA Discriminator'

    def _run(self, image, training_flag):
        network = self._block(image, n_filters=64, stride=2, activation=tf.nn.leaky_relu, normalization_flag=False)
        network = self._block(network, n_filters=128, stride=2, activation=tf.nn.leaky_relu, normalization_flag=True)
        network = self._block(network, n_filters=256, stride=2, activation=tf.nn.leaky_relu, normalization_flag=True)
        network = self._block(network, n_filters=512, stride=2, activation=tf.nn.leaky_relu, normalization_flag=True)
        network = self._block(network, n_filters=512, stride=1, activation=tf.nn.leaky_relu, normalization_flag=True)
        network = self._block(network, n_filters=1, stride=1, activation=tf.nn.sigmoid, normalization_flag=False)
        return network

    @staticmethod
    def _block(network, n_filters, stride, activation, normalization_flag):
        network = tf.pad(network, [[0, 0], [2, 2], [2, 2], [0, 0]])
        network = tf.layers.conv2d(network, filters=n_filters, kernel_size=4, strides=stride, padding='valid')
        if normalization_flag:
            network = tf.contrib.layers.instance_norm(network)
        if activation is not None:
            network = activation(network)
        return network


class CycadaInitializer(Component):
    name = 'CyCADA Initializer'

    def __init__(self, path, exclude_scope):
        super().__init__()
        self.path = path
        self.exclude_scope = exclude_scope

    def _run(self, session):
        session.run(tf.global_variables_initializer())
        variables = {}
        for variable in tf.global_variables(scope=substrings_to_re(include=('network',), exclude=self.exclude_scope)):
            self.verbose(variable.name, new_line=True)
            variables[variable.name[variable.name.find('network'):].split(':')[0]] = variable
        tf.train.Saver(variables).restore(sess=session, save_path=self.path)


class GANOptimizer(Component):
    name = 'GAN Optimizer'

    def __init__(
        self, generator_learning_rate, discriminator_learning_rate, decay_step, delimiter, include_scope,
        generator_step, discriminator_step, summary_step, image_tensor_names
    ):
        """
        :param decay_step: step of decreasing learning rate in 2 times or None
        :param delimiter: for run(..., metrics) metrics[:delimiter] are for generator, else - for discriminator
        :param include_scope: list of extra include scopes with updatable variables for generator or None
        """
        super().__init__()
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.decay_step = decay_step
        self.delimiter = delimiter
        self.include_scope = include_scope
        self.generator_step = generator_step
        self.discriminator_step = discriminator_step
        self.summary_step = summary_step
        self.image_tensor_names = image_tensor_names
        self.components['summary'] = SummaryMaker()
        self.components['learning_rate'] = LearningRate(decay_step=self.decay_step)

    def _run(self, session=None, metrics=None):
        if session is None:
            graph = tf.get_default_graph()
            self.global_step = tf.train.get_global_step()
            include_scope = ('generator',)
            if self.include_scope is not None:
                include_scope += self.include_scope

            with tf.control_dependencies(tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=substrings_to_re(include=include_scope, exclude=None)
            )):
                generator_learning_rate = self.components['learning_rate'].run(
                    learning_rate=self.generator_learning_rate, global_step=self.global_step
                )
                generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate)
                self.generator_operation = generator_optimizer.minimize(
                    loss=graph.get_tensor_by_name('generator_loss:0'),
                    var_list=tf.global_variables(
                        scope=substrings_to_re(include=include_scope, exclude=None)
                    )
                )
                self.generator_summary = self.components['summary'].run(metrics[:self.delimiter], prefix='t')
                self.image_summary = self.components['summary'].run(
                    tuple(graph.get_tensor_by_name(name) for name in self.image_tensor_names), prefix='t'
                )

            with tf.control_dependencies(tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope=substrings_to_re(include=('discriminator',), exclude=None)
            )):
                discriminator_learning_rate = self.components['learning_rate'].run(
                    learning_rate=self.discriminator_learning_rate, global_step=self.global_step
                )
                discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=discriminator_learning_rate)
                self.discriminator_operation = discriminator_optimizer.minimize(
                    loss=graph.get_tensor_by_name('discriminator_loss:0'),
                    var_list=tf.global_variables(scope=substrings_to_re(include=('discriminator',), exclude=None))
                )
                self.discriminator_summary = self.components['summary'].run(metrics[self.delimiter:], prefix='t')
        else:
            step = session.run(self.global_step)

            # running generator
            tensors = [[], metrics[:self.delimiter], self.generator_summary, []]
            if self.summary_step and step % self.summary_step == self.summary_step - 1:
                tensors[3] = self.image_summary
            if self.generator_step and step % self.generator_step == self.generator_step - 1:
                tensors[0] = self.generator_operation
            values = session.run(tensors)
            result = values[1]
            summaries = [values[2]]
            if self.summary_step and step % self.summary_step == self.summary_step - 1:
                summaries.append(values[3])

            # running discriminator
            tensors = [[], metrics[self.delimiter:], self.discriminator_summary]
            if self.discriminator_step and step % self.discriminator_step == self.discriminator_step - 1:
                tensors[0] = self.discriminator_operation
            values = session.run(tensors)
            result += values[1]
            summaries.append(values[2])
            return np.array(result), summaries
