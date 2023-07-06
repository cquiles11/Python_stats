#!/usr/bin/env python

# Import the programs to use
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pyplot
import sys 

# Assign the csv a variable
filename = sys.argv[1]

# Read the variable
dataset = pd.read_csv(filename)

# Visualize the raw data
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("python_scatter_plot.png")

# Build the module
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])

# Visualize the module
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('y vs x')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig("python_linearregression_model.png")