import os
import numpy as np
import tensorflow as tf
from datetime import datetime


def list_directory(path):
    paths = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    np.random.shuffle(paths)
    return paths


def generate_file_name(index):
    return f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}_{index}.tfrecords'


def substrings_to_re(include, exclude):
    included = rf'(?=.*({"|".join(include)}).*$)' if include is not None else ''
    excluded = rf'(?=^((?!(?:{"|".join(exclude)})).)*$)' if exclude is not None else ''
    return rf'{included}{excluded}'


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

