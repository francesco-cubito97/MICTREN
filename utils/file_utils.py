from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import errno
import random
import numpy as np
import torch

def create_dir(path):
    """
    Create a new directory for the path give
    """
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    
def seeds_init(seed):
    """
    Set the seed for all principal components
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)