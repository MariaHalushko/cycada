import tensorflow as tf

from utils import list_directory, int_feature, float_feature, bytes_feature
from .component import Component


class TfRecordIterator(Component):
    name = 'TfRecords->NumpyGenerator'

    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def _run(self):
        records = []
        for file_name in list_directory(self.path):
            for record in tf.python_io.tf_record_iterator(file_name):
                records.append(record)
                if len(records) == self.batch_size:
                    yield records
                    records = []

        if len(records) > 0:
            yield records


class TfRecordToDictValues(Component):
    name = 'TfRecord->DictValues'

    def __init__(self):
        super().__init__()

    def _run(self, description, record):
        """
        :param description: ((key, dtype), ...)
        """
        values = []
        example = tf.train.Example()
        example.ParseFromString(record)
        for entry_index, (key, dtype) in enumerate(description):
            feature = example.features.feature[key]

            if dtype == 'int':
                value_list = feature.int64_list
            elif dtype == 'float':
                value_list = feature.float_list
            else:  # dtype == 'bytes'
                value_list = feature.bytes_list

            values.append(value_list.value)

        return values


class DictValuesToTfRecord(Component):
    name = 'DictValues->TfRecord'

    def __init__(self):
        super().__init__()

    def _run(self, description, values):
        """
        :param description: ((key, dtype), ...)
        """
        feature = {}
        for index, (key, dtype) in enumerate(description):
            value = values[index]

            if dtype == 'int':
                to_feature = int_feature
            elif dtype == 'float':
                to_feature = float_feature
            else:  # dtype == 'bytes'
                to_feature = bytes_feature

            feature[key] = to_feature(value)
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class NumericRecordIterator(Component):
    name = 'TfRecords->Tensors'

    def __init__(self, components, n_processes, skip_size, repeat_flag, shuffle_flag, batch_size):
        """
        :param components: {'record_to_tensor': Component}
        """
        super().__init__(components=components)
        self.n_processes = n_processes
        self.skip_size = skip_size
        self.repeat_flag = repeat_flag
        self.shuffle_flag = shuffle_flag
        self.batch_size = batch_size
        self.buffer_size = batch_size * 100

    def _run(self, path):
        dataset = tf.data.TFRecordDataset(list_directory(path))
        if self.skip_size:
            dataset = dataset.skip(self.skip_size)
        dataset = dataset.map(self.components['record_to_tensor'].run, num_parallel_calls=self.n_processes)
        if self.shuffle_flag:
            dataset = dataset.shuffle(self.buffer_size)
        if self.repeat_flag:
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size).prefetch(self.n_processes)
        return dataset.make_one_shot_iterator().get_next()
