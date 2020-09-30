import  matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 3000, 1)
y = 1 * (0.99 ** x)
plt.plot(x, y)
plt.xlabel('0.99')
plt.show()
