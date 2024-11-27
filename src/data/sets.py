def load_sets(path='../data/processed/', val=False):
    """Load the different locally save sets

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    import numpy as np
    import os.path

    X_train = np.load(f'{path}X_train.npy', allow_pickle=True) if os.path.isfile(f'{path}X_train.npy') else None
    X_val = np.load(f'{path}X_val.npy', allow_pickle=True) if os.path.isfile(f'{path}X_val.npy') else None
    X_test  = np.load(f'{path}X_test.npy', allow_pickle=True ) if os.path.isfile(f'{path}X_test.npy')  else None
    y_train = np.load(f'{path}y_train.npy', allow_pickle=True) if os.path.isfile(f'{path}y_train.npy') else None
    y_val = np.load(f'{path}y_val.npy', allow_pickle=True) if os.path.isfile(f'{path}y_val.npy') else None
    y_test  = np.load(f'{path}y_test.npy', allow_pickle=True ) if os.path.isfile(f'{path}y_test.npy')  else None

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataframes(path='../data/processed/', val=False):
    """Load the different locally save dataframes

    Parameters
    ----------
    path : str
        Path to the folder where the sets are saved (default: '../data/processed/')

    Returns
    -------
    Pandas Dataframe
        Features for the training set
    Pandas Dataframe
        Target for the training set
    Pandas Dataframe
        Features for the validation set
    Pandas Dataframe
        Target for the validation set
    Pandas Dataframe
        Features for the testing set
    Pandas Dataframe
        Target for the testing set
    """
    import os.path
    import pandas as pd

    X_train = pd.read_pickle(f'{path}X_train.pkl')
    y_train = pd.read_pickle(f'{path}y_train.pkl')
    X_val = pd.read_pickle(f'{path}X_val.pkl')
    y_val = pd.read_pickle(f'{path}y_val.pkl')
    X_test = pd.read_pickle(f'{path}X_test.pkl')
    y_test = pd.read_pickle(f'{path}y_test.pkl')
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_sets(X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, path='../data/processed/'):
    """Save the different sets locally

    Parameters
    ----------
    X_train: Numpy Array
        Features for the training set
    y_train: Numpy Array
        Target for the training set
    X_val: Numpy Array
        Features for the validation set
    y_val: Numpy Array
        Target for the validation set
    X_test: Numpy Array
        Features for the testing set
    y_test: Numpy Array
        Target for the testing set
    path : str
        Path to the folder where the sets will be saved (default: '../data/processed/')

    Returns
    -------
    """
    import numpy as np

    if X_train is not None:
      np.save(f'{path}X_train', X_train)
      X_train.to_pickle(f'{path}X_train.pkl')
    if X_val is not None:
      np.save(f'{path}X_val',   X_val)
      X_val.to_pickle(f'{path}X_val.pkl')
    if X_test is not None:
      np.save(f'{path}X_test',  X_test)
      X_test.to_pickle(f'{path}X_test.pkl')
    if y_train is not None:
      np.save(f'{path}y_train', y_train)
      y_train.to_pickle(f'{path}y_train.pkl')
    if y_val is not None:
      np.save(f'{path}y_val',   y_val)
      y_val.to_pickle(f'{path}y_val.pkl')
    if y_test is not None:
      np.save(f'{path}y_test',  y_test)
      y_test.to_pickle(f'{path}y_test.pkl')
