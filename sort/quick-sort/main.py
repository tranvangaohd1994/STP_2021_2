import random

def partition(arr, left, right):
    p = arr[left]
    j = left
    for i in range(left+1, right):
        if arr[i] < p :
            j += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[left], arr[j] = arr[j], arr[left]
    return j

def quick_sort(arr, low, hight):
    if (low < hight):
        m = partition (arr, low, hight)
        quick_sort(arr, low, m)
        quick_sort(arr, m+1, hight)

def __main__():
    n = int(input())
    arr = []
    for i in range(0,n):
        val = random.randint(0,n)
        arr.append(val)
    print(arr)
    arr[-2] = arr[-2] % 4
    print(arr)
    quick_sort(arr,0,len(arr))
    print(arr)
__main__()