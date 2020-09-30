import numpy as np


def get_vector(int_action, len_of_vector):
    result = np.zeros(len_of_vector, dtype=int)
    for i in range(len_of_vector):
        result[i] = int_action % 3
        int_action = int_action // 3
    return result

def count_nonzero(my_vector):
    count = 0
    for x in my_vector:
        if x != 0:
            count +=1
    return count
def count_new_full(my_vector):
    count = 0
    for x in my_vector:
        if (x == -1) or (x == 2):
            count += 1
    return count


if __name__ == "__main__":
    print (get_vector(41853,10))