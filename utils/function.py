import numpy as np


# Compute the Entropy
def Entropy_H(x):
    return -(np.log2(x) * x).sum()


# Compute Binary Entropy
def Entropy_h(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


# Compute Joint Entropy
def Joint_Entropy(pxy):
    return - (pxy * np.log2(pxy)).sum()


def Conditional_Entropy_H_Y_givenBy_X_x(x, py_x):
    return -(py_x[x] * np.log2(py_x[x])).sum()


def Conditional_Entropy_H_Y_givenBy_X(pxy, py_x):
    return -(pxy * np.log2(py_x)).sum()


def MutualInformation_X_Y(pxy, px, py):
    return (pxy * np.log2(pxy / px * py)).sum()


def MutualInformation_X_Y_py_x(py_x, px, py):
    return (py_x * px * np.log2(py_x / py)).sum()
