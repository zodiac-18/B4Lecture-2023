"""To adapt HMM."""
import argparse
import re

import numpy as np

import HMM

def main():
    # create parser
    parser = argparse.ArgumentParser(
        description="This is a program to model prediction by HMM")
    parser.add_argument("path",help="path to the file")

    # get arguments
    args = parser.parse_args()

    # create file name
    f_name = re.sub(r".+\\","",args.path)
    f_name = re.sub(r"\..+","",f_name)

    # read out pickle
    """
    data
        answer_models (100)
        output (100,100)
        models
            PI (5,3,1) or (5,5,1)
            A (5,3,3) or (5,5,5)
            B (5,3,5) or (5,5,5)
    """
    data = HMM.load_pickle(args.path)
    print(np.array(data["models"]['A']))


if __name__ == "__main__":
    main()
