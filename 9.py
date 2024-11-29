import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.asmatrix(np.eye((m)))  # Replaced np.mat with np.asmatrix
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# load data points
data = pd.read_csv('LR.csv')
colA = np.array(data['colA'])
colB = np.array(data['colB'])

# preparing and adding 1
mcolA = np.asmatrix(colA)
mcolB = np.asmatrix(colB)

# convert to matrix form
m = np.shape(mcolA)[1]
one = np.ones((1, m), dtype=int)

# horizontally stack
X = np.hstack((one.T, mcolA.T))

# set k here (0.5)
ypred = localWeightRegression(X, mcolB, 0.5)

# Sort Index for plotting
SortIndex = X[:, 1].argsort(0)
xsort = X[SortIndex][:, 0]

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(colA, colB, color='green')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)

# Labeling the axes
plt.xlabel('colA')
plt.ylabel('colB')

# Show the plot
plt.show()
