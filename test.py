import numpy as np

import utils
import utils.function as utils

px = [[1 / 8, 1 / 4], [1 / 2, 1 / 8]]

print(utils.Entropy_H(px))

p1 = np.array([[1, 2],
               [3, 4]])

print(p1 * p1)
print(np.dot(p1, p1))
