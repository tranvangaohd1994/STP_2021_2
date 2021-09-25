import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data_linear.csv").values
N = data.shape[0]
x = data[:,0].reshape(-1, 1)
y = data[:,1].reshape(-1, 1)
# draw Oxy with points in data_linear
plt.scatter(x, y)
plt.xlabel("met vuong")
plt.ylabel("gia")

# convert x to matrix 
x = np.hstack((np.ones((N, 1)), x))
# print(x)

# init w0 = 0, w1 = 1
w = np.array([0., 1.]).reshape(-1, 1)

numOfIteration = 100
loss = np.zeros((numOfIteration, 1))
learning_rate = 0.000001
for i in range(1, numOfIteration):
    d_w0 = np.dot(x, w) - y
    loss[i] = 0.5*np.sum(d_w0*d_w0)
    w[0] -= learning_rate*np.sum(d_w0)
    w[1] -= learning_rate*np.sum(np.multiply(x[:, 1 ].reshape(-1, 1), d_w0))
    print(loss[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')
plt.show()

x1 = 50
y1 = w[0] + w[1]*x1
print('gia nha cho 50m^2 la: ', y1)