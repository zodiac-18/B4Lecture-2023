"Regression analysis"
import argparse

import csv
import numpy as np
import matplotlib.pyplot as plt


#least squares method 2-dimension
def lsm_2(dataset: np.ndarray, deg: int, reg = False, stren = 1.0) -> np.ndarray:
    """To adapt least squares method to 2-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function
        reg (bool): decide whether to regularization or not to
        stren (float): the strength of the regularization

    Returns:
        np.ndarray: the coefficient of the function
    """
    # example: argmin all y-(a1x**2 + a2x + a3)**2
    # 1-degree -> 2-arg, 2-degree -> 3-arg

    # generate matrix for compute
    # to adapt (X^(T)X)^(-1)X^(T)Y
    X = np.array([[np.power(dataset[0][j],i) for i in range(deg+1)] for j in range(len(dataset[0]))])
    XT = X.T
    if reg:
        I = np.eye(deg+1)
        A = np.linalg.inv(XT@X + stren*I)
    else:
        A = np.linalg.inv(XT@X)

    return A@XT@data[1]

#least squares method 3-dimension
def lsm_3(dataset: np.ndarray, deg: int) -> np.ndarray:
    """To adapt least squares method to 3-dimension dataset and return the decided degree function.

    Args:
        dataset (np.ndarray): As the name implies
        deg (int): the degree of the function

    Returns:
        np.ndarray: the coefficient of the function
    """
    # example: argmin all z-(a1x**2 + a2y**2 + a3x + a4y + a5)**2
    # 1-degree -> 3-arg, 2-degree -> 5-arg
    # TODO: add function about create 3-dimension's least squares matrix
    """
    4,0  2,2  3,0  2,1  2,0
    2,2  0,4  1,2  0,3  0,2
    3,0  1,2  2,0  1,1  1,0
    2,1  0,3  1,1  0,2  0,1
    2,0  0,2  1,0  0,1  0,0
    """
    X = np.ravel(np.array([[i,0,0,i] for i in reversed(range(deg+1))]))
    print(X+X)
    A = np.array([np.sum(np.power(dataset[0], i)*np.power(dataset[1],j)) for i in range(deg+1)])


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
    #decide plot range
    x_max = np.max(dataset[0])
    x_min = np.min(dataset[0])

    #genereate point x
    x_group = np.linspace(x_min, x_max, quant)
    y_group = np.zeros(quant)

    #generate point y
    for i in range(len(function)):
        y_group += np.power(x_group,i) * function[i]

    return x_group, y_group



if __name__ == "__main__":
    #get parser
    parser = argparse.ArgumentParser(description="This program is to adapt least squares method to the dataset for regression analysis")
    parser.add_argument("path",help="The path of dataset")
    parser.add_argument("-f", "--f_name", help="The name of saved graph", default="", type=str)
    parser.add_argument("-d", "--deg", help="The degree of the graph wants to generate", default=1, type=int)
    parser.add_argument("-r","--reg",help="adapt regularization",action="store_true")
    parser.add_argument("-s","--strength",help="the strength of the regularization",default=1.0,type=float)

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
        func = lsm_2(data, args.deg, reg=args.reg, stren=args.strength)
        char = ""
        char +=('{:.03f}'.format(func[0]))
        for i in (range(1,args.deg+1)):
            if func[i] >0:
                char += "+"
            char +=(str('{:.03f}'.format(func[i])) + "x^" + str(i))
        x_group, y_group = genpoint(data,func)
        ax = plt.subplot()
        ax.plot(data[0],data[1],'.',label="dataset")
        ax.plot(x_group,y_group,label=char)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc=0)
        plt.show()
    elif len(data) == 3:
        lsm_3(data, args.deg)
        ax = plt.subplot(projection="3d")
        ax.plot(data[0],data[1],data[2],'.',c='b',label="dataset")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc=0)
        plt.show()
    else:
        raise ValueError("data dimension should in 2 or 3 but " + str(len(data_array[0])))
