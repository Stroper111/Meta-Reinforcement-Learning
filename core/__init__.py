# ignore keras/tf warning and set logging to Errors
import warnings
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Error ('3' Fatal)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

from .multi_env import MultiEnv
