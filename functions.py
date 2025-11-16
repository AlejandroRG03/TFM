import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uproot


def read_root(path, tree, columns, nrows=None):
    """Read data from .root"""


    with uproot.open(path) as f:

        tree = f[tree]

        if nrows == None: # Load the entire dataset

            df = tree.arrays(columns, library='pd')

        else: # Load firs nrows of the dataset

            df = tree.arrays(columns, library='pd', entry_start=0, entry_stop=nrows)
    
    return df