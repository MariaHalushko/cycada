import io
import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.getcwd())
from components import DictValuesToTfRecord
from utils import generate_file_name


def run(images, labels, size, path, n_results):
    values_ro_record = DictValuesToTfRecord()
    writer = tf.python_io.TFRecordWriter(os.path.join(path, generate_file_name(0)))
    description = [('image', 'bytes'), ('label', 'int')]
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= n_results >= 0:
            break
        buffer = io.BytesIO()
        image = np.stack((image, np.zeros_like(image), np.zeros_like(image)), axis=2)
        Image.fromarray(image).resize((size, size), resample=Image.BILINEAR).save(buffer, 'png')
        values = [[buffer.getvalue()], [label]]
        writer.write(values_ro_record.run(description=description, values=values))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--n_results', type=int, default=-1)
    options = vars(parser.parse_args())

    (training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()
    path = options['path']
    shutil.rmtree(path)
    os.mkdir(path)
    training_path = os.path.join(path, 'training')
    os.mkdir(training_path)
    run(
        images=training_images, labels=training_labels, size=options['size'], path=training_path,
        n_results=options['n_results']
    )
    validation_path = os.path.join(path, 'validation')
    os.mkdir(validation_path)
    run(
        images=validation_images, labels=validation_labels, size=options['size'], path=validation_path,
        n_results=options['n_results']
    )
