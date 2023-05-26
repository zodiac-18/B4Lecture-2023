#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Least squares method."""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

class LinearRegression:
   def __init__(self, degree = 2):
      self.degree = degree
      self.v = 0
      self.w = None
      
   def Polynominal(self, x):
      degree = self.degree
      phi_x = np.array([x**(j) for j in range(degree+1)]).T
      return phi_x
   
   def fit(self, x, y, lamb = 0.0):
      phi = self.Polynominal(x)
      self.w = (np.linalg.inv(phi.T @ phi + lamb * np.eye(phi.shape[1])) @ (phi.T)) @ y
   
   def predict(self, x):
      poly = self.Polynominal(x)
      phi = np.zeros((x.shape[0], self.w.shape[0]))
      for i in range(phi.shape[1]):
         phi[:, i] = poly[i]
      y_out = self.w @ phi
      return y_out

def main():
    """Apply linear regression to the input data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to the input data")
    parser.add_argument(
        "-dx", "--degreex", help="Regression degree of x", type=int, required=True
    )
    parser.add_argument(
        "-dy", "--degreey", help="Regression degree of y (dimension > 2)", default=3, type=int
    )
    parser.add_argument(
        "-l",
        "--lamb",
        help="Coefficient of normalization",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    path = args.path
    
    df = pd.read_csv(path, sep=',')
    data = np.array(df)
    degree_x = args.degreex
    degree_y = args.degreey
    norm_lamb = args.lamb

    dim = data.shape[1]
    
    if dim == 2:
      x, y = data.T
      model = LinearRegression(degree_x)
      model.fit(x, y)
    
    if dim == 2:
      x1 = np.linspace(x.min(), x.max())

      # predict
      prediction = model.predict(x1)

      # plot
      plt.plot(x1, prediction, label = "data")
      plt.scatter(x[:, 0], y, c="r", label="Observed data")
      plt.title("data1")
      plt.xlabel("$x_0$")
      plt.ylabel("$y$")
      plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=1)
      plt.tight_layout()
      plt.show()


if "__main__" == __name__:
    main()