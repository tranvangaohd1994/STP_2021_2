import random

def insert_sort(arr):
    for i in range(1, len(arr), 1):
        largest = i
        for j in range(i-1, -1, -1):
            if arr[largest] < arr[j]:
                arr[largest], arr[j] = arr[j], arr[largest]
                largest = j
            else: break

def partition(arr, low: int, high: int):
    pivot = arr[high]
    left = low-1
    for i in range(low, high):
        if arr[i] <= pivot:
            left+=1
            arr[left], arr[i] = arr[i], arr[left]
    
    arr[left+1], arr[high] = arr[high], arr[left+1]
    return left+1


def quick_sort(arr, l: int, r: int):
    if len(arr) == 1: return arr
    if l<r:
        pi = partition(arr,l,r)

        quick_sort(arr,l, pi-1)
        quick_sort(arr, pi+1, r)

def merge_sort(arr):
    if(len(arr) > 1):
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i=0 # left iterator
        j=0 # right iterator
        k=0 # main list iterator

        while i<len(left) and j<len(right):
            if left[i] <= right[j]:
                arr[k] = left[i]
                i+=1
            else:
                arr[k] = right[j]
                j+=1
            k+=1

        while i<len(left):
            arr[k] = left[i]
            i+=1
            k+=1

        while j<len(right):
            arr[k] = right[j]
            j+=1
            k+=1

def heapify(arr, n, i):
    largest = i
    l = 2*i + 1
    r = 2*i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    for i in range(len(arr)//2, -1, -1):
        heapify(arr, len(arr), i)

    for i in range(len(arr)-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

n = int(input())
rlist = []

for i in range(0,n):
    rlist.append(random.randint(0, 1000000))

# print(rlist)
req = rlist[-2] % 4
if req == 0:
    print('run insert sort')
    insert_sort(rlist)
elif req == 1:
    print('run quick sort')
    quick_sort(rlist, 0, n-1)
elif req == 2:
    print('run merge sort')
    merge_sort(rlist)
else:
    print('run heap sort')
    heap_sort(rlist)
print(rlist[:10])
print(rlist[-10:])
