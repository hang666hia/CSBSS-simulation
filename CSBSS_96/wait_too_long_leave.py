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
q_user.put(0.75)
q_user.put(0.95)
wait_time = np.array([1, 4, 2, 4, 3, 4, 4, 2])
leave_index = np.array([])
for i in range(len(wait_time)):
    if wait_time[i] == 4:
        leave_index = np.append(leave_index, i)
print(leave_index)
count = 0
for i in range(q_user.qsize()):
        a = q_user.get()
        if (len(leave_index) != 0) and (leave_index[0] == i):
            print(i)
            leave_index = np.delete(leave_index, 0)
            wait_time = np.delete(wait_time, i-count)
            count += 1

        else:
            q_user.put(a)
print("count:",count)
print(wait_time)
print(q_user.qsize())
print(q_user.get())
print(q_user.get())
print(q_user.get())
print(q_user.get())
sys.exit(0)
