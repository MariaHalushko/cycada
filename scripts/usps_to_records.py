import io
import os
import sys
import h5py
import shutil
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.getcwd())
from components import DictValuesToTfRecord
from utils import generate_file_name


def run(images, labels, size, out_path, n_results):
    values_ro_record = DictValuesToTfRecord()
    writer = tf.python_io.TFRecordWriter(os.path.join(out_path, generate_file_name(0)))
    description = [('image', 'bytes'), ('label', 'int')]
    for i, (image, label) in enumerate(zip(images, labels)):
        if i >= n_results >= 0:
            break
        image = (image.reshape(16, 16) * 255).astype(np.uint8)
        image = np.stack((image, np.zeros_like(image), np.zeros_like(image)), axis=2)
        buffer = io.BytesIO()
        Image.fromarray(image).resize((size, size), resample=Image.BILINEAR).save(buffer, 'png')
        values = [[buffer.getvalue()], [label]]
        writer.write(values_ro_record.run(description=description, values=values))
    writer.close()


if __name__ == '__main__':
    # USPS dataset download link: https://www.kaggle.com/bistaumanga/usps-dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_results', type=int, default=-1)
    parser.add_argument('--size', type=int, default=32)
    options = vars(parser.parse_args())

    with h5py.File(options['in_path'], 'r') as file:
        training_data = file.get('train')
        training_images = training_data.get('data')[:]
        training_labels = training_data.get('target')[:]
        validation_data = file.get('test')
        validation_images = validation_data.get('data')[:]
        validation_labels = validation_data.get('target')[:]

    out_path = options['out_path']
    shutil.rmtree(out_path)
    os.mkdir(out_path)
    training_out_path = os.path.join(out_path, 'training')
    os.mkdir(training_out_path)
    run(
        images=training_images, labels=training_labels, size=options['size'], out_path=training_out_path,
        n_results=options['n_results']
    )
    validation_out_path = os.path.join(out_path, 'validation')
    os.mkdir(validation_out_path)
    run(
        images=validation_images, labels=validation_labels, size=options['size'], out_path=validation_out_path,
        n_results=options['n_results']
    )
