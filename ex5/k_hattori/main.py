import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import random


def load_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        buf = [row for row in reader]
    # Convert to ndarray in float64
    array = np.array(buf[1:])
    array = array.astype(np.float64)

    return array

def k_means(data, k):
    rand_index = random.sample(data.shape[0], k)
    centroids = data[rand_index]
    for i in range(data.shape[0]):
        distance = np.zeros(k)
        distance = np.linalg.norm(data[i] - centroids)


    


def main():
    # make parser
    parser = argparse.ArgumentParser(
        prog='main.py',
        usage='Demonstration of argparser',
        description='description',
        epilog='end',
        add_help=True,
    )
    # add arguments
    parser.add_argument('-f', dest='filename', help='Filename', required=True)
    parser.add_argument('-k', dest='k', type=int,
                        help='number for clustering',
                        required=False, default=4)
    # parse arguments
    args = parser.parse_args()


if __name__ == "__main__":
    main()
    exit(1)