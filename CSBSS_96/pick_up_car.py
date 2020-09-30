from queue import Queue
import numpy as np
import sys

q_user = Queue()
q_user.put(4)
q_user.put(0.5)
q_user.put(0.78)
q_user.put(0.4)
q_user.put(2)
q_user.put(0.7)
q_user.put(1)
q_user.put(0.8)
q_user.put(1.2)
q_user.put(1.5)
car = np.array([0.8, 0.6, 0.75, 0.9, 0.65])
car = np.sort(car)

for i in range(q_user.qsize()):
    if len(car) == 0:
        pass
    else:
        demand = q_user.get()
        if (len(car) == 1) and car[0] < demand:
            q_user.put(demand)
        elif (len(car) == 1) and car[0] >= demand:
            car = np.delete(car, 0)
        elif demand <= car[0]:
            car = np.delete(car, 0)
        else:
            found = False
            for j in range(len(car) - 1):
                if car[j] < demand <= car[j+1]:
                    found = True
                    car = np.delete(car, j+1)
            if not found:
                q_user.put(demand)

print(q_user.qsize())
print("----")
print(q_user.get())
print(q_user.get())
print(q_user.get())
print(q_user.get())
print(q_user.get())
sys.exit(0)
