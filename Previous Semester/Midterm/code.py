# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:33:01 2021

@author: 91960
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

n = np.arange(10) + 1

f4 = np.log2(n)

f5 = np.exp2(np.sqrt(f4))

#f3 = np.power(n, n)

f2 = np.power(n, 1/3)

f1 = np.power(10, n)

plt.figure()
plt.plot(f2, c = 'red', label = 'f2')
plt.plot(f1, c = 'blue', label = 'f1')
plt.legend()
plt.show()

