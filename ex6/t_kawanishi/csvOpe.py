"""csv operation."""
import csv

import numpy as np


def read_csv(path: str) -> np.ndarray:
    """read out csv to matrix

    Args:
        path (str): the csv file path

    Returns:
        np.ndarray: data matrix
    """
    # read scv file and change type
    data_array = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            sub_group = []
            for value in line:
                sub_group.append(float(value))
            data_array.append(sub_group)

    # group by dimension
    data_array = np.array(data_array)

    return data_array


if __name__ == "__main__":
    pass
