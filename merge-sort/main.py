import random

def merge(arr, low, mid, hight):
    b = []
    i = low
    j = mid
    while i < mid and j < hight :
        if arr[i] <= arr[j] :
            b.append(arr[i])
            i += 1
        else :
            b.append(arr[j])
            j += 1
    while i < mid:
        b.append(arr[i])
        i += 1
    while j < hight:
        b.append(arr[j])
        j += 1
    n = hight - low
    for k in range(0, n):
        arr[low+k] = b[k]
    
def merge_sort(arr, low, hight):
    if (low < hight - 1):
        mid = (low + hight)//2
        merge_sort(arr, low, mid)
        merge_sort(arr, mid, hight)
        merge(arr, low, mid, hight)

def inp():
    return (int(input()))

def __main__():
    n = inp()
    arr = []
    for i in range(0, n):
        val = random.randint(1, 100)
        arr.append(val)
    print(arr)
    arr[-2] = arr[-2] % 4
    print(arr)
    merge_sort(arr, 0, len(arr))
    print(arr[0:10])
    print(arr[-10:])

__main__()