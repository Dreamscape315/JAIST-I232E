import numpy as np
import sympy


def Entropy_H(x):
    """
    Compute the Entropy H(X)
    :param x: the set of possibility
    :return:: Entropy
    """
    return -(np.log2(x, where=x != 0) * x).sum() + 0


def Entropy_h(p):
    """
    Compute Binary Entropy h(x)
    :param p: p
    :return: Entropy
    """
    return -p * np.log2(p, where=p != 0) - (1 - p) * np.log2(1 - p, where=(1 - p) != 0) + 0


def Joint_Entropy(pxy):
    """
    Compute Joint Entropy H(X,Y)
    :param pxy: 2D vector of joint possibilities
    :return: Entropy
    """
    return - (pxy * np.log2(pxy, where=pxy != 0)).sum() + 0


def Joint_Entropy_Sympy(pxy):
    """
    Compute Joint Entropy H(X,Y)
    :param pxy: 2D vector of joint possibilities
    :return: Entropy
    """
    return - (pxy * sympy.log(pxy, 2, where=pxy != 0)).sum() + 0


def Conditional_Entropy_H_Y_givenBy_X_x(x, py_x):
    """
    Compute Conditional Entropy H(Y|X=x)
    :param x: The condition x (start with 0)
    :param py_x:The 2D vector of P(y|x)
    :return: Entropy
    """
    print(np.log2(py_x[x], where=py_x[x] != 0))
    return -(py_x[x] * np.log2(py_x[x], where=py_x[x] != 0)).sum() + 0


def Conditional_Entropy_H_Y_givenBy_X(pxy, py_x):
    """
    Compute Conditional Entropy H(Y|X)
    :param pxy:
    :param py_x:
    :return: Entropy
    """
    return -(pxy * np.log2(py_x, where=pxy != 0)).sum() + 0


def Conditional_Entropy_H_Y_givenBy_X_II(hxy, hx):
    """
    Compute Conditional Entropy H(Y|X) = H(X,Y) - H(X)
    :param hxy:
    :param hx:
    :return:Entropy
    """
    return hxy - hx + 0


def Marginalization_PX(pxy):
    """
    Compute px by pxy
    :param pxy: pxy
    :return: px
    """
    return np.sum(pxy, axis=1)


def Marginalization_PY(pxy):
    """
    Compute py by pxy
    :param pxy: pxy
    :return: py
    """
    return np.sum(pxy, axis=0)


def Bayes_Py_I(px, py_x):
    """
    Compute py by Bayes Probability
    :param px:
    :param py_x:
    :return: py
    """
    return np.dot(px, py_x)


def MutualInformation_X_Y(pxy, px, py):
    return (pxy * np.log2(pxy / px * py)).sum() + 0


def MutualInformation_X_Y_py_x(py_x, px, py):
    return (py_x * px * np.log2(py_x / py)).sum() + 0


def MarkovChain(s1, P, n):
    """
    Compute the Markov Chain
    :param s1: The initial state
    :param P: The transition matrix
    :param n: The number of steps
    :return: The probability of the state after n steps
    """
    s = s1
    for i in range(n-1):
        s = np.dot(s, P)
        print(s)
    return s
