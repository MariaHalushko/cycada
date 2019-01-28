import os
import numpy as np
import tensorflow as tf

from .component import Component
from .simple import NameGetter
from utils import generate_file_name


class Trainer(Component):
    name = 'Trainer'

    def __init__(
        self, components, batch_size, n_steps, path, model_path, training_step, validation_step, log_step, save_step,
        tensorboard_path
    ):
        """
        :param components: {
                'model': Component, 'initializer': Component, 'optimizer': Component, 'validator': Component
            }
            Model returns a list of metric tensors.
        """
        super().__init__(components, log_flag=True)
        self.n_steps = n_steps
        self.model_path = model_path
        self.training_step = training_step
        self.validation_step = validation_step
        self.log_step = log_step
        self.save_step = save_step
        self.components['name_getter'] = NameGetter()
        self.writer = tf.summary.FileWriter(tensorboard_path, flush_secs=10)
        self.global_step = tf.train.create_global_step()
        self.global_step_increment = tf.assign_add(self.global_step, 1)
        if self.training_step:
            training_path = os.path.join(path, 'training')
            self.training_metric_tensors = self.components['model'].run(
                path=training_path, training_flag=True, batch_size=batch_size
            )
            self.training_metric_names = self.components['name_getter'].run(self.training_metric_tensors)
            self.training_metrics = np.empty((len(self.training_metric_names), self.log_step))
            self.components['optimizer'].run(metrics=self.training_metric_tensors)
        if self.validation_step:
            validation_path = os.path.join(path, 'validation')
            self.validation_metric_tensors = self.components['model'].run(
                path=validation_path, training_flag=False, batch_size=batch_size
            )
            self.validation_metric_names = self.components['name_getter'].run(self.validation_metric_tensors)
            self.validation_length = self.log_step // self.validation_step
            self.validation_metrics = np.empty((len(self.validation_metric_names), self.validation_length))
            self.components['validator'].run(metrics=self.validation_metric_tensors)
        if self.save_step:
            self.saver = tf.train.Saver()

    def _run(self):
        with tf.Session() as session:
            self.components['initializer'].run(session)

            while True:
                step = session.run(self.global_step)
                if step == self.n_steps:
                    break
                session.run(self.global_step_increment)
                if self.training_step:
                    training_metrics, training_summaries = self.components['optimizer'].run(
                        session=session, metrics=self.training_metric_tensors
                    )
                    for summary in training_summaries:
                        self.writer.add_summary(summary, global_step=step)
                    index = step % self.log_step
                    self.training_metrics[:, index] = training_metrics
                    string = self._make_string(self.training_metrics[:, index], self.training_metric_names)
                    self.verbose(f'  training: step {step}/{self.n_steps}: {string}')
                if self.validation_step and step % self.validation_step == self.validation_step - 1:
                    validation_metrics, validation_summaries = self.components['validator'].run(
                        session=session, metrics=self.validation_metric_tensors
                    )
                    for summary in validation_summaries:
                        self.writer.add_summary(summary, global_step=step)
                    index = step // self.validation_step % self.validation_length
                    self.validation_metrics[:, index] = validation_metrics
                    string = self._make_string(self.validation_metrics[:, index], self.validation_metric_names)
                    self.verbose(f'validation: step {step}/{self.n_steps}: {string}')
                if self.log_step and step % self.log_step == self.log_step - 1:
                    if self.training_step:
                        training_metric_mean = np.mean(self.training_metrics, axis=1)
                        self.log(f'step: {step}/{self.n_steps}')
                        string = self._make_string(training_metric_mean, self.training_metric_names)
                        self.log(f'training: {string}')
                    if self.validation_step:
                        validation_metric_mean = np.mean(self.validation_metrics, axis=1)
                        string = self._make_string(validation_metric_mean, self.validation_metric_names)
                        self.log(f'validation: {string}')
                if self.save_step and step % self.save_step == self.save_step - 1:
                    model_path = os.path.join(self.model_path, generate_file_name(0)[:-16])
                    self.saver.save(session, model_path)
                    self.log(f'save: {model_path}')

    @staticmethod
    def _make_string(metrics, metric_names):
        result = ''
        for metric, name in zip(metrics, metric_names):
            result += f'{name}: {metric:.4e}, '
        return result[:-2]
