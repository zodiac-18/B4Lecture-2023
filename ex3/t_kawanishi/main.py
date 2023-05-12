"Regression analysis"
import argparse

import csv
import numpy as np
import matplotlib.pyplot as plt


#least squares method 2-dimension
def lsm_2(dataset: np.ndarray, deg: int) -> np.ndarray:
    """To adapt least squares method to 2-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function

    Returns:
        np.ndarray: the coefficient of the function
    """
    #create matrix for lsm
    A = []
    B = []
    for i in reversed(range(deg,deg*2+1)):
        A_sub = []
        for j in range(deg+1):
            A_sub.append(np.sum(np.power(dataset[0],i-j)))
        B.append(np.sum(np.power(dataset[0],i-deg)*dataset[1]))
        A.append(A_sub)
    A = np.array(A)
    A = np.linalg.inv(A)
    B = np.array(B)

    return (A@B)

#least squares method 3-dimension
def lsm_3(dataset: np.ndarray, deg: int) -> np.ndarray:
    """To adapt least squares method to 3-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function

    Returns:
        np.ndarray: the coefficient of the function
    """
    pass



    return None

#generate point to plot
def genpoint(dataset: np.ndarray, function: np.ndarray, quant = 10000) -> tuple[np.ndarray, np.ndarray]:
    """To generate function's data to plot

    Args:
        dataset (np.ndarray): As the name implies
        function (np.ndarray): The function coefficient
        quant (int): The quantity of the data list. Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: function data list for plot
                                        first is x-aixs and next is y-axis
    """
    x_max = np.max(dataset[0])
    x_min = np.min(dataset[0])
    x_group = np.linspace(x_min, x_max, quant)
    y_group = np.zeros(quant)
    for i in range(len(function)):
        y_group += np.power(x_group,i) * np.flipud(function)[i]

    return x_group, y_group



if __name__ == "__main__":
    #get parser
    parser = argparse.ArgumentParser(description="This program is to adapt least squares method to the dataset for regression analysis")
    parser.add_argument("path",help="The path of dataset")
    parser.add_argument("-f", "--f_name", help="The name of saved graph", default="", type=str)
    parser.add_argument("-d", "--deg", help="The degree of the graph wants to generate", default=1, type=int)

    #read out parser
    args = parser.parse_args()

    #read scv file
    data_array = []
    with open(args.path,'r') as f:
        reader = csv.reader(f)
        reader_data = next(reader)
        for line in reader:
            data_array.append(line)

    #change type
    for i in range(len(data_array)):
        for j in range(len(data_array[i])):
            data_array[i][j] = float(data_array[i][j])
    data_array = np.array(data_array)
    data = data_array.T



    #plot data
    if len(data) == 2:
        func = lsm_2(data, args.deg)
        x_group, y_group = genpoint(data,func)
        ax = plt.subplot()
        ax.plot(data[0],data[1],'.',label="dataset")
        ax.plot(x_group,y_group,label="hi")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc=0)
        plt.show()
    elif len(data) == 3:
        ax = plt.subplot(projection="3d")
        ax.plot(data[0],data[1],data[2],'.',c='b',label="dataset")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc=0)
        plt.show()
    else:
        print("Error: data dimension should in 2 or 3 but " + str(len(data_array[0])))
