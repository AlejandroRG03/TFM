import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot


def read_root(path, tree, columns, nrows = None):
    """
    Docstring for read_root function

    :param path: Path to the ROOT file
    :param tree: Name of the tree to read
    :param columns: List of columns to read
    :param nrows: Number of rows to read (default is None, which means all rows)
    
    :return: DataFrame containing the requested data
    """


    with uproot.open(path) as f:

        tree = f[tree]

        if nrows == None: # Load the entire dataset

            df = tree.arrays(columns, library='pd')

        else: # Load first nrows of the dataset

            df = tree.arrays(columns, library='pd', entry_start=0, entry_stop=nrows)
    
    return df


def vector_norm(data, x='x', y='y', z='z'):
    """
    Docstring for vector_norm function

    :param data: DataFrame containing vector components in columns 'x', 'y', 'z'
    :param x: Name of the column containing the x-component
    :param y: Name of the column containing the y-component
    :param z: Name of the column containing the z-component

    :return: Column Series containing the vector norms
    """

    return np.sqrt(data[x]**2 + data[y]**2 + data[z]**2)

def compute_angles(axis, vectors, x='x', y='y', z='z'):
    """
    Docstring for compute_angles function

    :param axis: Numpy array representing the reference axis (normalized)
    :param vectors: DataFrame containing vector components in columns 'x', 'y', 'z'
    
    :return: Series containing the angles in radians between each vector and the reference axis
    """

    dot_products = (vectors[x] * axis[0] +
                    vectors[y] * axis[1] +
                    vectors[z] * axis[2])
    
    cosines = dot_products / (vector_norm(vectors, x, y, z))

    angles = np.arccos(cosines)

    return angles

