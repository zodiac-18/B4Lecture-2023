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

def ridge_reg(x, y, n, k):
    if x.ndim == 1:
        x = x.reshape(-1, len(x)).T
    N = n * x.shape[1] + 1

    # Create a matrix of explanatory variables and identity matrix
    poly_x = np.zeros([x.shape[0], N])
    # Power number
    j = 0
    # row number
    l = 0
    for i in range(N):
        # Reset power number(When x has 2 or more rows)
        if i and i % (n+1) == 0:
            j = 1
            l += 1
        poly_x[:,i] = x[:,l]** j
        j += 1
    I = np.eye(N)
    print(poly_x.shape)
    # Calculate a matrix of Regression coefficients beta
    tmp = np.dot(poly_x.T, poly_x)
    tmp = np.dot(np.linalg.inv(tmp+k*I), poly_x.T)
    beta = np.dot(tmp, y)

    # Calculate regression results
    y_predict = 0
    for i in range(N):
        y_predict += poly_x[:,i] * beta[i]
    if y_predict.ndim > 1:
        y_result = 0
        for i in range(y_predict.shape[1]):
            y_result += y_predict[:,i]
        y_result -= y_predict.shape[1] - 1

    return y_predict

def main():
    # Load csv files
    data1 = load_csv('ex3\data1.csv')
    data2 = load_csv('ex3\data2.csv')
    data3 = load_csv('ex3\data3.csv')

    # Plot data
    x1 = data1[:,0]
    y1 = data1[:,1]
    plt.scatter(x1, y1)
    plt.show()

    x2 = data2[:,0]
    y2 = data2[:,1]
    plt.scatter(x2, y2)
    plt.show()

    x3 = data3[:,0]
    y3 = data3[:,1]
    z3 = data3[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3, y3, z3)
    plt.show()

    y1_predict = ridge_reg(x1, y1, 1, 0)
    plt.scatter(x1, y1)
    plt.plot(x1, y1_predict)
    plt.show()

    z3_predict = ridge_reg(data3[:,:2], z3, 1, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3, y3, z3_predict)
    plt.show()



if __name__ == "__main__":
    main()
    exit(1)
