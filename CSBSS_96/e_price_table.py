import numpy as np
import matplotlib.pyplot as plt

e_price_table = np.zeros(96)
for e in range(96):
    e_price_table[e] = 1
    if 0 <= e <= 27 or 92 <= e <= 95:
        e_price_table[e] = 0.5
    elif 28 <= e <= 43 or 76 <= e <= 91:
        e_price_table[e] = 1.5
    else:
        e_price_table[e] = 1


plt.plot(np.arange(len(e_price_table)), e_price_table)
plt.ylabel('time')
plt.xlabel('price')
plt.show()
