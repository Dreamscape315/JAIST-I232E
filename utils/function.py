import numpy as np


def Entropy_H(x):
    """
    Compute the Entropy H(X)
    :param x: the set of possibility
    :return:: Entropy
    """
    return -(np.log2(x) * x).sum()


def Entropy_h(p):
    """
    Compute Binary Entropy h(x)
    :param p: p
    :return: Entropy
    """
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def Joint_Entropy(pxy):
    """
    Compute Joint Entropy H(X,Y)
    :param pxy: 2D vector of joint possibilities
    :return: Entropy
    """
    return - (pxy * np.log2(pxy)).sum()


def Conditional_Entropy_H_Y_givenBy_X_x(x, py_x):
    """
    Compute Conditional Entropy H(Y|X=x)
    :param x: The condition x (start with 0)
    :param py_x:The 2D vector of P(y|x)
    :return: Entropy
    """
    return -(py_x[x] * np.log2(py_x[x])).sum()


def Conditional_Entropy_H_Y_givenBy_X(pxy, py_x):
    return -(pxy * np.log2(py_x)).sum()


def MutualInformation_X_Y(pxy, px, py):
    return (pxy * np.log2(pxy / px * py)).sum()


def MutualInformation_X_Y_py_x(py_x, px, py):
    return (py_x * px * np.log2(py_x / py)).sum()
