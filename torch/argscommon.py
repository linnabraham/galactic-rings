
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import argparse
import datetime
project_name = "ps-rings"

parser=argparse.ArgumentParser()
parser.add_argument('--log_level',  default='INFO',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help='Logging level')
parser.add_argument('--tf_log_level',  default='ERROR',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help='Tensorflow Logging level')
parser.add_argument('-input_shape', nargs="+", type=int, default=(240,240), help="target size to resize images to before training")
parser.add_argument('-channels', type=int, default=3, help="Number of channels in the image data")
parser.add_argument('-random_state', type=int, default=42, help="seed for random processes for reproducibility")

def setup_logger(log_level):
    #logging.basicConfig(level=logging.DEBUG) 
    logger = logging.getLogger(project_name)#logger = logging.getLogger(__name__)
    # Set the logging level based on the input argument
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter("[%(filename)s:%(lineno)d][%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S.%f")

    def custom_time(*args):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    formatter.formatTime = custom_time
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

