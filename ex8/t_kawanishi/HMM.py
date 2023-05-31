"""Function of HMM."""

import pickle

def load_pickle(path:str)->dict:
    """load pickle file

    Args:
        path (str): path to the pickle

    Returns:
        dict: pickle contents
    """
    data = pickle.load(open(path, "rb"))
    return data