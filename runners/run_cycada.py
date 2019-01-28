import os
import sys
import argparse
import logging.config

sys.path.insert(0, os.getcwd())
from components import *

# https://arxiv.org/pdf/1711.03213.pdf

BASE_PATH = '/tmp'
MODEL_PATH = os.path.join(BASE_PATH, 'models')
TENSORBOARD_PATH = os.path.join(BASE_PATH, 'tensorboard')
LOG_CONFIG = dict(
    version=1,
    formatters={
        'detailed': {
            'format': '%(levelname)s %(asctime)s %(message)s',
            'datefmt': '%m/%d/%Y %I:%M:%S'
        },
    },
    handlers={
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'detailed',
            'filename': '/tmp/log.txt',
        }
    },
    root={
        'handlers': ['file'],
        'level': logging.DEBUG,
    },
)
logging.config.dictConfig(LOG_CONFIG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n_steps', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    options = vars(parser.parse_args())

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    if not os.path.exists(TENSORBOARD_PATH):
        os.mkdir(TENSORBOARD_PATH)

    model = CycadaModel(
        components={
            'record_to_tensor': MnistRecordToTensor(size=(32, 32)),
            'preprocessor': Preprocessor(),
            'network': MnistNetwork(),
            'generator': CycadaGenerator(),
            'discriminator': CycadaDiscriminator()
        },
        n_processes=2
    )
    initializer = CycadaInitializer(path=options['model_path'], exclude_scope=('Adam',))
    optimizer = GANOptimizer(
        generator_learning_rate=0.0001, discriminator_learning_rate=0.0001,
        decay_step=0, delimiter=6, include_scope=('target/network',),
        generator_step=1, discriminator_step=1, summary_step=20,
        image_tensor_names=(
            'source_image:0',
            'source_target_image:0',
            'target_image:0',
            'target_source_image:0',
            'source_target_source_image:0',
            'target_source_target_image:0',
        )
    )
    Trainer(
        components={
            'model': model,
            'initializer': initializer,
            'optimizer': optimizer,
        },
        batch_size=options['batch_size'], n_steps=options['n_steps'], path=options['path'], model_path=MODEL_PATH,
        tensorboard_path=TENSORBOARD_PATH, validation_step=0, log_step=10, save_step=0, training_step=1
    ).run()
