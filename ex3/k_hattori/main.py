import numpy as np
import matplotlib.pyplot as plt
import csv

def load_csv(path):

    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        buf = [row for row in reader]
    # Convert to ndarray in float64
    array = np.array(buf[1:])
    array = array.astype(np.float64)

    return array

def lreg(x, y, n):
    N = n + 1
    if x.ndim > 1:
        N += x.shape[1]
    # Create a matrix of explanatory variables
    poly_x = np.zeros([N, x.shape[0]])
    for i in range(N):
        poly_x[i,:] = x** i
    poly_x = poly_x.T

    # Calculate a matrix of Regression coefficients beta
    tmp = np.dot(poly_x.T, poly_x)
    tmp = np.dot(np.linalg.inv(tmp), poly_x.T)
    beta = np.dot(tmp, y)

    # Calculate regression results
    y_predict = 0
    for i in range(N):
        y_predict += x** i * beta[i]

    return y_predict

def main():
    # Load csv files
    x1 = load_csv('ex3\data1.csv')
    x2 = load_csv('ex3\data2.csv')
    x3 = load_csv('ex3\data3.csv')
    # Plot data
    plt.scatter(x1[:,0],x1[:,1])
    plt.show()
    plt.scatter(x2[:,0],x2[:,1])
    plt.show()
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3[:,0], x3[:,1], x3[:,2])
    plt.show()



if __name__ == "__main__":
    main()
    exit(1)
