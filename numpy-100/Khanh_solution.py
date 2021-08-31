import numpy as np
from numpy.core.fromnumeric import size
import sys

print(np.__version__)

init = np.zeros(10, dtype=int)
print(init)

print(init.size*init.itemsize)

# print(np.info(np.add))

ex6_array = np.zeros(10, dtype=int)
ex6_array[4] = 1
print(ex6_array)

ex7_array = np.arange(10, 50)
print(ex7_array)

ex8_array = ex7_array[::-1]
print(ex8_array)

ex9 = np.reshape(np.arange(9), (3, 3))
print(ex9)

ex10 = np.nonzero([1,2,0,0,4,0])
print(ex10)

ex11 = np.eye(3)
print(ex11)

ex12 = np.random.random((3, 3, 3))
print(ex12)

ex13 = np.random.random((10, 10))
print(ex13.max())
print(ex13.min())

ex14 = np.random.random((30))
print(ex14)
print(ex14.mean())

ex15 = np.ones((9, 9))
ex15[1:-1, 1:-1] = 0
print(ex15)

ex16 = np.ones((3, 3))
ex16 = np.pad(ex16, [(1, 1), (1, 1)], 'constant')
print(ex16)

print(0*np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan-np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3*0.1)

ex18 = np.diag(1 + np.arange(4), -1)
print(ex18)

ex19 = np.indices((8, 8)).sum(axis = 0) % 2
print(ex19)

ex20 = np.unravel_index(99, (6,7,8))
print(ex20)

ex21 = np.array(([0,1],[1,0]))
print(np.tile(ex21, (4,4)))

ex22 = np.random.randint(low=100, size=(5,5))
ex22 = (ex22 - np.mean(ex22)) / np.std(ex22)
print(ex22)

ex23 = np.dtype([("r", np.ubyte),
                ("g", np.ubyte),
                ("b", np.ubyte),
                ("a", np.ubyte)])

mat1 = np.ones((5,3))
mat2 = np.ones((3,2))
ex24 = np.matmul(mat1, mat2)
print(ex24)

ex25 = np.arange(10)
ex25[(3<ex25) & (ex25<8)] *=-1
print(ex25)

# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
# print(np.array([np.nan]).astype(int).astype(float))

ex29 = np.random.uniform(-5, 5, 10)
print(ex29)
print(np.copysign(np.ceil(np.abs(ex29)), ex29))

mat1 = np.random.randint(0 ,11, 10)
mat2 = np.random.randint(0 ,11, 10)
ex30 = np.intersect1d(mat1, mat2)
print(mat1)
print(mat2)
print(ex30)

ex33_today = np.datetime64('today')
ex33_yesterday = np.datetime64('today') - np.timedelta64(1)
ex33_tomorrow = np.datetime64('today') + np.timedelta64(1)
print(ex33_today)
print(ex33_yesterday)
print(ex33_tomorrow)

ex34 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(ex34)

a = np.ones(3) * 4
b = np.ones(3) * 2
print('a=',a)
print('b=',b)
np.add(a, b, out=b)
np.divide(a, 2, out=a)
np.negative(a, out=a)
np.multiply(a,b, out=b)
print(b)

ex36 = np.random.uniform(0,10,10)

print(ex36)
print(ex36 // 1)
print(np.floor(ex36))
print(ex36.astype(int))
print(np.trunc(ex36))

ex37 = np.tile(np.arange(5), (5, 1))
print(ex37)

def generate():
    for i in range(10):
        yield i+1

ex38 = np.fromiter(generate(), dtype=int)
print(ex38)

ex39 = np.linspace(0, 1, 11, endpoint=False)
print(ex39[1:])

ex40 = np.random.random(10)
ex40.sort()
print(ex40)

ex41 = np.random.random(10)
print(np.add.reduce(ex41))

mat1 = np.random.randint(0, 3, 5)
mat2 = np.random.randint(0, 3, 5)
print(mat1)
print(mat2)
print(np.allclose(mat1, mat2))

mat1.flags.writeable = False
# mat1[1] = 2

ex44 = np.random.random((10,2))
print(ex44)
x, y = ex44[:,0], ex44[:,1]
r = np.sqrt(x**2+y**2)
p = np.arctan(y, x)
print(r)
print(p)

ex45 = np.random.random(10)
print(ex45.argmax())
ex45[ex45.argmax()] = 0
print(ex45)

ex46 = np.zeros((5,5), [('x',float),('y',float)])
ex46['x'], ex46['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(ex46)

mat1 = np.arange(10)
mat2 = mat1 + 0.5
ex47 = 1 / np.subtract.outer(mat1, mat2)
print(np.linalg.det(ex47))

# for d in [np.byte, np.int32, np.int64]:
#     print(np.finfo(d).min)
#     print(np.finfo(d).max)
# for d in [np.float32, np.float64]:
#     print(np.finfo(d).min)
#     print(np.finfo(d).max)

np.set_printoptions(threshold=sys.maxsize)
ex49 = np.random.random((10, 10))
print(ex49)

mat1 = np.arange(10)
mat2 = np.random.uniform(0, 10)
print(mat1)
print(mat2)
ex50 = (np.abs(mat1-mat2)).argmin()
print(mat1[ex50])

ex51 = np.zeros(10, [('position', [('x', int, 1), 
                                ('y', int, 1)]),
                    ('color', [('r', int, 1), 
                    ('g', int, 1), 
                    ('b', int, 1)])])

print(ex51)

ex52 = np.random.random((100, 2))
a = ex52[10]
b = ex52[20]
print(a)
print(b)
print(np.linalg.norm(a-b))

ex53 = np.random.random(10)*100
print(ex53)
print(ex53.astype(int))

array_string = '''
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
'''
ex54 = np.genfromtxt(array_string.splitlines(), delimiter=',', dtype=int)
print(ex54)

ex55 = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(ex55):
    print(index, value)

x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
delta = np.sqrt(x**2 + y**2)
sigma, mu = 1, 0
ex56 = np.exp(-((delta-mu)**2 / (2*sigma**2)))
print(ex56)

ex57 = np.zeros((3, 3))
np.put(ex57, np.random.choice(range(9), 5, replace=False), 1)
print(ex57)

ex58 = np.random.rand(1, 10)
ex58 = ex58 - ex58.mean(axis=1, keepdims=True)
print(ex58)

ex59 = np.random.randint(0, 10, (3,3))
print(ex59)
print(ex59[ex59[:,1].argsort()])

ex60 = np.random.randint(1,3,(3,3))
print((~ex60.any(axis=0)).any())

ex61 = np.random.uniform(0,1,10)
print(ex61.flat[np.abs(ex61-0.4).argmin()])

mat1 = np.arange(3).reshape(3,1)
mat2 = np.arange(3).reshape(1,3)
it = np.nditer([mat1, mat2, None])
for x,y,z in it:
    z[...] = x+y
print(it.operands[2])

class NewArray(np.ndarray):
    def __new__(cls, array, name):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "unknown")

ex63 = NewArray(np.arange(10), "range_to_9")
print(ex63.name)

mat1 = np.random.randint(0, 10, 10)
print(mat1)
ex64 = np.ones(10)
np.add.at(ex64, mat1, 1)
print(ex64)

index = np.arange(7)[1:]
print(index)
x = np.random.randint(1, 10, len(index))
print(np.bincount(index, x))

w, h = 256, 256
pic = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
color = np.unique(pic.reshape(-1, 3), axis=0)
ex66 = len(color)
print(ex66)

mat1 = np.random.randint(0, 10, (3,3,3,3))
ex67 = mat1.sum(axis=(-2, -1))
print(ex67)

D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts
print(D_means)

mat1 = np.random.uniform(0,1,(5,5))
mat2 = np.random.uniform(0,1,(5,5))
ex69 = np.einsum("ij,ji->i",mat1,mat2)
print(ex69)

mat1 = np.arange(1, 6)
mat2 = np.zeros(len(mat1) + (len(mat1)-1)*3)
mat2[::3+1] = mat1
print(mat2)

mat1 = np.ones((5,5,3))
mat2 = np.ones((5,5))*2
ex71 = mat1*mat2[:,:,None]
print(ex71)

ex72 = np.arange(9).reshape(3,3)
ex72[[0,1]] = ex72[[1,0]]
print(ex72)

mat1 = np.random.randint(0,100, (10, 3))
mat2 = np.roll(mat1.repeat(2,axis=1),-1,axis=1)
mat2 = mat2.reshape(len(mat2)*3, 2)
mat2 = np.sort(mat2, axis=1)
ex73 = mat2.view(dtype=[('p0', mat2.dtype), ('p1', mat2.dtype)])
ex73 = np.unique(ex73)
print(ex73)

C = np.bincount([1,1,2,3,4,4,6])
ex74 = np.repeat(np.arange(len(C)), C)
print(C)
print(ex74)

ex75 = np.arange(10)
total = np.cumsum(ex75, dtype=float)
print(total)
total[3:] = total[3:] - total[:-3]
print(total)
print(total[2:]/3)

from numpy.lib import stride_tricks

def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
ex76 = rolling(np.arange(10), 3)
print(ex76)

mat1 = np.random.randint(0,2,10)
ex77 = np.logical_not(mat1)
print(mat1)
print(ex77)
mat2 = np.random.uniform(-1, 1, 10)
ex77 = np.negative(mat2)
print(mat2)
print(ex77)

def distance(p0, p1, p):
    t = p1-p0
    l=(t**2).sum(axis=1)
    u = -((p0[:,0]-p[...,0])*t[:,0] + (p0[:,1]-p[...,1])*t[:,1]) / l
    u = u.reshape(len(u), 1)
    d = p0 +u*t - p
    return np.sqrt((d**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))

mat1 = np.arange(15)[1:]
print(mat1)
# ex81 = stride_tricks.as_strided(mat1, (11,4), (4,4))
ex81 = rolling(mat1, 4)
print(ex81)

mat1 = np.random.uniform(0,1,(5,6))
u,s,v = np.linalg.svd(mat1)
ex82 = np.sum(s > 1e-10)
print(ex82)

ex83 = np.random.randint(0, 5, 20)
print(ex83)
print(np.bincount(ex83).argmax())

mat1 = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (mat1.shape[0]-n)
j = 1 + (mat1.shape[1]-n)
ex84 = stride_tricks.as_strided(mat1, shape=(i,j,n,n), strides= mat1.strides*2)
print(ex84)

class Symetric(np.ndarray):
    def __setitem__(self, key, value):
        i,j = key
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def sysmetric(matrix):
        return np.asarray(matrix + matrix.T - np.diag(matrix.diagonal())).view(Symetric)

ex85 = sysmetric(np.random.randint(0, 10, (4,4)))
ex85[0,1] = 100
print(ex85)

p, n = 5, 10
mat1 = np.ones((p, n, n))
mat2 = np.ones((p, n, 1))
ex86 = np.tensordot(mat1, mat2, axes=[[0,2], [0,1]])
print(ex86)

mat1 = np.ones((16, 16))
k = 4
ex87 = np.add.reduceat(
    np.add.reduceat(
        mat1, np.arange(0, mat1.shape[0], k), 
        axis=0), 
    np.arange(0, mat1.shape[1], k), axis=1)
print(ex87)

def game_of_life(mat):
    n = (mat[0:-2, 0:-2] + mat[0:-2, 1:-1] + mat[0:-2,2:] + mat[1:-1, 0:-2] + mat[1:-1, 2:] + mat[2:, 0:-2] + mat[2:, 1:-1] + mat[2:, 2:])

    birth = (n==3) & (mat[1:-1, 1:-1] == 0)
    survive = ((n==2) | (n==3)) & (mat[1:-1, 1:-1] == 1)
    mat[...] = 0
    mat[1:-1, 1:-1][birth | survive] = 1
    return mat

ex88 = np.random.randint(0,2,(10, 10))
print(ex88)
for i in range(20): ex88 = game_of_life(ex88)
print(ex88)

ex89 = np.arange(100)
np.random.shuffle(ex89)
n = 10
print(ex89[np.argpartition(-ex89, n)[:n]])

def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

ex90 = cartesian(([1,2,3], [4,5], [6,7]))
print(ex90)

mat1 = np.array([("ABC", 2.5, 3), ("DEF", 3.6, 2)])
ex91 = np.core.records.fromarrays(mat1.T, names='col1, col2, col3', formats='S8, f8, i8')
print(ex91)

ex92 = np.random.rand(100)

np.power(ex92,3)
ex92*ex92*ex92
np.einsum('i,i,i->i',ex92,ex92,ex92)

a = np.random.randint(0, 4, (8,3))
b = np.random.randint(0, 4, (2,2))
c = (a[..., np.newaxis, np.newaxis]==b)
ex93 = np.where(c.any((3,1)).all(1))[0]
print(ex93)

mat1 = np.random.randint(0, 4, (10,3))
print(mat1)
mat2 = np.all(mat1[:,1:] == mat1[:,:-1], axis=1)
ex94 = mat1[~mat2]
print(ex94)

mat1 = np.arange(10)
print(mat1)
ex95 = ((mat1.reshape(-1,1) & (2**np.arange(5))) != 0).astype(int)
print(ex95[:,::-1])

mat1 = np.random.randint(0,2,(5,5))
ex96 = np.unique(mat1, axis=0)
print(ex96)

mat1 = np.random.uniform(0, 1, 10)
mat2 = np.random.uniform(0, 1, 10)

print(np.einsum('i->', mat1)) # sum of mat1
print(np.einsum('i,i->i', mat1, mat2)) # mat1 * mat2
print(np.einsum('i,i', mat1, mat2)) # inner of mat1 and mat2
print(np.einsum('i,j->ij', mat1, mat2)) # outer of mat1 and mat2

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
print(M)
print(X.sum(axis=-1))
M &= (X.sum(axis=-1) == n)
print(M)
print(X[M])

mat1 = np.random.randn(101)
n = 1000
idx = np.random.randint(0, mat1.size, (n, mat1.size))
mean = mat1[idx].mean(axis=1)
ex100 = np.percentile(mean, [2.5, 97.5])
print(ex100)