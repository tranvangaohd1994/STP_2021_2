import sys
import random

input =  sys.stdin.readline

def inp():
    return (int(input()))

def insert_sort(my_list):
    n = len(my_list)
    for i in range(n):
        for j in range(i):
            if (my_list[i] < my_list[j]):
                my_list[i], my_list[j] = my_list[j], my_list[i]
    return my_list

def __main__():
    n = inp()
    my_list = []
    for i in range(0, n):
        val = random.randint(1, 100)
        my_list.append(val)
    print(my_list)
    my_list[-2] = my_list[-2] % 4
    print (my_list)
    after_sort = insert_sort(my_list)
    print(after_sort[:10])
    print(after_sort[-10:])

__main__()