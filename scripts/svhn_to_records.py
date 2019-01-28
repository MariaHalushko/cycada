import io
import os
import sys
import scipy.io
import shutil
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.getcwd())
from components import DictValuesToTfRecord
from utils import generate_file_name


def run(in_path, out_path, n_results):
    values_ro_record = DictValuesToTfRecord()
    writer = tf.python_io.TFRecordWriter(os.path.join(out_path, generate_file_name(0)))
    description = [('image', 'bytes'), ('label', 'int')]
    mat = scipy.io.loadmat(in_path)
    for i, (image, label) in enumerate(zip(np.transpose(mat['X'], (3, 0, 1, 2)), np.mod(mat['y'], 10))):
        if i >= n_results >= 0:
            break
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, 'png')
        values = [[buffer.getvalue()], [label]]
        writer.write(values_ro_record.run(description=description, values=values))
    writer.close()


if __name__ == '__main__':
    # SVHN dataset download link: http://ufldl.stanford.edu/housenumbers/
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_results', type=int, default=-1)
    options = vars(parser.parse_args())

    in_path = options['in_path']
    out_path = options['out_path']
    shutil.rmtree(out_path)
    os.mkdir(out_path)
    training_out_path = os.path.join(out_path, 'training')
    training_in_path = os.path.join(in_path, 'train_32x32.mat')
    os.mkdir(training_out_path)
    run(in_path=training_in_path, out_path=training_out_path, n_results=options['n_results'])
    validation_out_path = os.path.join(out_path, 'validation')
    validation_in_path = os.path.join(in_path, 'test_32x32.mat')
    os.mkdir(validation_out_path)
    run(in_path=validation_in_path, out_path=validation_out_path, n_results=options['n_results'])
