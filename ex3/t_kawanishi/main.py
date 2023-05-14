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
    # example: argmin all y-(a1x**2 + a2x + a3)**2
    # 1-degree -> 2-arg, 2-degree -> 3-arg
    #create matrix for lsm
    A = np.array([np.sum(np.power(dataset[0], i-j)) for i in reversed(range(deg,deg*2+1)) for j in range(deg+1)]).reshape([deg+1,deg+1]) # maybe needed norm?
    B = np.array([np.sum(np.power(dataset[0],i-deg)*dataset[1]) for i in reversed(range(deg,deg*2+1))])
    A_inv = np.linalg.inv(A) # inverse of A

    return (A_inv@B)

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
        char = ""
        char +=('{:.03f}'.format(func[args.deg]) + "x^" + str(args.deg))
        for i in reversed(range(args.deg)):
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
        print("Error: data dimension should in 2 or 3 but " + str(len(data_array[0])))
