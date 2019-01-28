import os
import sys
import argparse
import logging.config

sys.path.insert(0, os.getcwd())
from components import *

BASE_PATH = '/tmp'
MODEL_PATH = os.path.join(BASE_PATH, 'models')
TENSORBOARD_PATH = os.path.join(BASE_PATH, 'tensorboard')
LOG_PATH = os.path.join(BASE_PATH, 'log.txt')
LOG_CONFIG = dict(
    version=1,
    formatters={
        'detailed': {
            'format': '%(levelname)s %(asctime)s %(message)s',
            'datefmt':'%m/%d/%Y %I:%M:%S'
        },
    },
    handlers={
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'detailed',
            'filename': LOG_PATH,
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
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_steps', type=int, required=True)
    parser.add_argument('--validation_step', type=int, required=True)
    options = vars(parser.parse_args())

    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    if not os.path.exists(TENSORBOARD_PATH):
        os.mkdir(TENSORBOARD_PATH)

    model = MnistModel(
        components={
            'record_to_tensor': MnistRecordToTensor(size=(32, 32)),
            'preprocessor': Preprocessor(),
            'network': MnistNetwork()
        },
        n_processes=2
    )
    Trainer(
        components={
            'model': model,
            'initializer': Initializer(),
            'optimizer': Optimizer(learning_rate=0.0001, decay_step=0, tensor_name='loss:0'),
            'validator': Validator()
        },
        batch_size=options['batch_size'], n_steps=options['n_steps'], path=options['path'], model_path=MODEL_PATH,
        tensorboard_path=TENSORBOARD_PATH, validation_step=options['validation_step'], log_step=1000,
        save_step=1000, training_step=1
    ).run()
