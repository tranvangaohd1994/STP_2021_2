import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# data csv
data = pd.read_csv("dataset.csv").values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

# Oxy
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='tu choi')
plt.legend(loc=1)
plt.xlabel("muc luong (trieu)")
plt.ylabel("kinh nghiem (nam)")

# them cot 1 vao du lieu x
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

numOfIteration = 1000
loss = np.zeros((numOfIteration, 1))
learning_rate = 0.01

for i in range(1, numOfIteration):
    # gia tri du doan
    y_predict = sigmoid(np.dot(x, w))
    loss[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    # gradient descent
        # dao ham f(X)
    d_f = np.dot(x.T, y_predict - y)
    w -= learning_rate*d_f
    print(loss[i])

# ve duong phan cach w0 + w1*x + w2*y = 0
# y_predict > t ==> w0 + w1*x1 + w2*x2 > -ln(1/t -1)
t = 0.5
plt.plot((4, 10), (-(w[0] + 4*w[1] + np.log(1/t-1))/w[2], -(w[0] +10*w[1] + np.log(1/t-1))/w[2]), 'g')
plt.show()