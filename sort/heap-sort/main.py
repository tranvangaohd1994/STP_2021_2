import random

def heapify(arr, n, i):
    max = i
    left = 2*i + 1
    right = 2*i + 2

    if left<n and arr[left]>arr[max]:
        max = left
    if right<n and arr[right]>arr[max]:
        max = right
    if max != i :
        arr[i], arr[max] = arr[max], arr[i]
        heapify(arr, n, max)

def heap_sort(arr, n):
    for i in range(n//2-1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def __main__():
    n = int(input())
    arr = []
    for i in range(0,n):
        val = random.randint(1,100)
        arr.append(val)
    print(arr)
    arr[-2] = arr[-2] % 4
    print(arr)
    heap_sort(arr, len(arr))
    print(arr)
    print(arr[:10])
    print(arr[-10:])
__main__()