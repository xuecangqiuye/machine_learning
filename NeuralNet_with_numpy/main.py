import numpy as np
import matplotlib.pyplot as plt
import copy as cp



# Relu function
def rel(z):
    return (abs(z) + z) / 2


# Show data classified result
def show_ans(x_s, y_s, m_s, subplt):
    for k in range(0, m_s):
        if y_s[0, k] == 1:
            subplt.plot(x_s[0, k], x_s[1, k], 'ro')
        else:
            subplt.plot(x_s[0, k], x_s[1, k], 'go')


# Show err's curve
def show_err(subplt, err_data):
    subplt.cla()
    subplt.plot(err_data[0], err_data[1])


# Show plane classified result
def show_class(W, b, n, g, gamma, max, min, subplt):
    X_t = np.arange(min[0], max[0], max[0] / 16)
    Y_t = np.arange(min[1], max[1], max[1] / 16)
    X_temp = [[], []]
    Z_t = []
    A_t = []
    miu_t = []
    sigmd_t = []
    Z_norm_t = []
    Z_cali_t = []
    for i in X_t:
        for j in Y_t:
            X_temp[0].append(i)
            X_temp[1].append(j)
    m_t = len(X_temp[0])
    X_t = np.array(X_temp).reshape(2, m_t)

    # Forward test
    Z_t.clear()
    A_t.clear()
    for i in range(len(n)):
        if i == 0:
            Z_t.append(np.dot(W[i], np.array(X_t).reshape(2, m_t)))
        else:
            Z_t.append(np.dot(W[i], A_t[i - 1]))
        # batch-normal
        miu_t.append(np.mean(Z[i], 1, None, None, True))
        sigmd_t.append(np.mean(np.square(Z_t[i] - miu_t[i]), 1, None, None, True))
        Z_norm_t.append((Z_t[i] - miu_t[i]) / np.sqrt(sigmd_t[i] + epsilon))
        Z_cali_t.append(gamma[i] * Z_norm_t[i] + b[i])
        if g[i] == 'RELU':
            A_t.append(rel(Z_cali_t[i]))
        elif g[i] == 'sigmoid':
            A_t.append(1.0 / (1.0 + np.exp(-Z_cali_t[i])))

    Y_tra_t = cp.deepcopy(A_t[len(n) - 1])
    Y_tra_t[Y_tra_t >= 0.5] = 1
    Y_tra_t[Y_tra_t < 0.5] = 0
    show_ans(X_t, Y_tra_t, m_t, subplt)


# start
fo = open("data1.txt")
x_temp = []
y_temp = []
for line in fo:
    temp = line.split(',')
    x_temp += [float(f) for f in temp[0:2]]
    y_temp += [float(temp[2])]

alpha = 0.07
# alpha = 5
beta = 0.9
X = np.array(x_temp).reshape(int(len(x_temp) / 2), 2).T
Y = np.array(y_temp).reshape(1, int(len(y_temp)))
# init par
v = len(X)
m = len(X[0])
n = [4, 4, 1]
g = ["RELU", "RELU", "sigmoid"]
W = []
b = []
gamma = []
miu = []
sigmd = []
Z = []
Z_norm = []
Z_cali = []
A = []
j = 1
epsilon = 0.00000000000001
loop = 10000
dJdZ = [0] * len(n)
dJdZ_cali = [0] * len(n)
dJdZ_norm = [0] * len(n)
dJdGamma = [0] * len(n)
dJdsigmd = [0] * len(n)
dJdmiu = [0] * len(n)
dJdW = [0] * len(n)
dJdb = [0] * len(n)
VdJdW = [0] * len(n)
VdJdb = [0] * len(n)
VdJdgamma = [0] * len(n)
SdJdW = [0] * len(n)
SdJdb = [0] * len(n)
err = [[], []]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

# initialize W，b
for i in n:
    if j == 1:
        W.append(np.random.randn(i, v))
        # W.append(np.ones(i * v).reshape(i, v))
        j += 1
    else:
        W.append(np.random.randn(i, last_i))
        # W.append(np.ones(i * last_i).reshape(i, last_i))
    b.append(np.zeros(i).reshape(i, 1))
    gamma.append(np.ones(i).reshape(i, 1))
    last_i = i
# normalize sets
mu = np.mean(X, 1).reshape(v, 1)
X = X - mu
lambd = np.mean(np.square(X), 1).reshape(v, 1)
X = X / lambd
X = X / np.max(X[0, :])
X_max = [1, 1]
X_min = [-1, -1]
# X_max = [np.ceil(np.max(X[0, :])), np.ceil(np.max(X[1, :]))]
# X_min = [np.floor(np.min(X[0, :])), np.floor(np.min(X[1, :]))]

# loop
j = 0
for j in range(loop):

    # Forward propagation
    Z.clear()
    A.clear()
    miu.clear()
    sigmd.clear()
    Z_norm.clear()
    Z_cali.clear()

    for i in range(len(n)):
        if i == 0:
            Z.append(np.dot(W[i], np.array(X).reshape(v, m)))
        else:
            Z.append(np.dot(W[i], A[i - 1]))
        # batch-normal
        miu.append(np.mean(Z[i], 1, None, None, True))
        sigmd.append(np.mean(np.square(Z[i] - miu[i]), 1, None, None, True))
        Z_norm.append((Z[i] - miu[i]) / np.sqrt(sigmd[i] + epsilon))
        Z_cali.append(gamma[i] * Z_norm[i] + b[i])
        if g[i] == 'RELU':
            A.append(rel(Z_cali[i]))
        elif g[i] == 'sigmoid':
            A.append(1.0 / (1.0 + np.exp(-Z_cali[i])))
    Y_tra = cp.deepcopy(A[len(n) - 1])
    Y_tra[Y_tra >= 0.5] = 1
    Y_tra[Y_tra < 0.5] = 0
    print("------------Forward propagation DONE-------------\n")
    J = -(np.dot(Y, np.log(A[len(n) - 1]).T) + np.dot((1 - Y), np.log(1 - A[len(n) - 1]).T)) / m
    print('the cost is {0}\n'.format(J[0][0]))

    if J[0][0] < 0.01:
        # stop term1
        break
    # times
    err[0].append(j + 1)
    # error rate
    err[1].append(J[0][0])

    # backward
    # dJdZ_cali[len(n) - 1] = (A[len(n) - 1] - Y)
    dJdZ_cali[len(n) - 1] = (A[len(n) - 1] - Y) / m

    for i in range(len(n)):
        # batch-normal backward
        dJdZ_norm[len(n) - 1 - i] = gamma[len(n) - 1 - i] * dJdZ_cali[len(n) - 1 - i]
        # a number
        dJdsigmd[len(n) - 1 - i] = np.sum(
            dJdZ_norm[len(n) - 1 - i] * (Z[len(n) - 1 - i] - miu[len(n) - 1 - i]) * (-0.5) * np.power((sigmd[len(n) - 1 - i] + epsilon), -1.5), 1, None, None, True
        )
        # a number
        dJdmiu[len(n) - 1 - i] = np.sum(-dJdZ_norm[len(n) - 1 - i] / np.sqrt(sigmd[len(n) - 1 - i] + epsilon), 1, None,None, True)

        dJdZ[len(n) - 1 - i] = dJdZ_norm[len(n) - 1 - i] * ((1 - dJdmiu[len(n) - 1 - i] * (1 / m)) / np.sqrt(sigmd[len(n) - 1 - i]+epsilon) + (-0.5 * (Z[len(n) - 1 - i] - miu[len(n) - 1 - i]) * np.power((sigmd[len(n) - 1 - i] + epsilon), -1.5) * dJdsigmd[len(n) - 1 - i] * 2 / m * (Z[len(n) - 1 - i] - miu[len(n) - 1 - i]) * (1-dJdmiu[len(n) - 1 - i] / m)))

        dJdGamma[len(n) - 1 - i] = np.sum(Z_norm[len(n) - 1 - i] * dJdZ_cali[len(n) - 1 - i], 1, None, None, True)

        dJdb[len(n) - 1 - i] = np.sum(dJdZ_cali[len(n) - 1 - i], 1, None, None, True)

        if len(n) - 1 - i == 0:
            dJdW[len(n) - 1 - i] = np.dot(dJdZ[len(n) - 1 - i], X.T)
        else:
            dJdW[len(n) - 1 - i] = np.dot(dJdZ[len(n) - 1 - i], A[len(n) - 1 - i - 1].T)

        if i <= len(n) - 2:
            temp_matrix = cp.deepcopy(Z_cali[len(n) - 1 - i - 1])
            if g[len(n) - 1 - i - 1] == 'RELU':
                temp_matrix[temp_matrix <= 0] = 0
                temp_matrix[temp_matrix > 0] = 1
            # elif g[len(n) - 1 - i - 1] == 'sigmoid':
            #     temp_matrix_s = np.exp(-temp_matrix) / np.square(1 + np.exp(-temp_matrix))
            #     temp_matrix = temp_matrix_s
            dJdZ_cali[len(n) - 1 - i - 1] = np.dot(W[len(n) - 1 - i].T, dJdZ[len(n) - 1 - i]) * temp_matrix

    if j == 99999:
        # stop term2
        break

    # Update parameter
    for ini_i in range(len(n)):
        # RMSprop gradient descent
        # SdJdW[len(n) - 1 - ini_i] = beta * SdJdW[len(n) - 1 - ini_i] + (1 - beta) * np.square(dJdW[len(n) - 1 - ini_i])
        # SdJdb[len(n) - 1 - ini_i] = beta * SdJdb[len(n) - 1 - ini_i] + (1 - beta) * np.square(dJdb[len(n) - 1 - ini_i])
        # W[len(n) - 1 - ini_i] = W[len(n) - 1 - ini_i] - alpha * (dJdW[len(n) - 1 - ini_i]/np.sqrt(SdJdW[len(n) - 1 - ini_i]+epsilon))
        # b[len(n) - 1 - ini_i] = b[len(n) - 1 - ini_i] - alpha * (dJdb[len(n) - 1 - ini_i]/np.sqrt(SdJdb[len(n) - 1 - ini_i]+epsilon))

        # Momentum gradient descent
        VdJdW[len(n) - 1 - ini_i] = beta * VdJdW[len(n) - 1 - ini_i] + (1 - beta) * dJdW[len(n) - 1 - ini_i]
        VdJdb[len(n) - 1 - ini_i] = beta * VdJdb[len(n) - 1 - ini_i] + (1 - beta) * dJdb[len(n) - 1 - ini_i]
        VdJdgamma[len(n) - 1 - ini_i] = beta * VdJdgamma[len(n) - 1 - ini_i] + (1 - beta) * dJdGamma[len(n) - 1 - ini_i]
        W[len(n) - 1 - ini_i] = W[len(n) - 1 - ini_i] - alpha * VdJdW[len(n) - 1 - ini_i]
        b[len(n) - 1 - ini_i] = b[len(n) - 1 - ini_i] - alpha * VdJdb[len(n) - 1 - ini_i]
        gamma[len(n) - 1 - ini_i] = gamma[len(n) - 1 - ini_i] - alpha * VdJdgamma[len(n) - 1 - ini_i]

        # General gradient descent
        # W[len(n) - 1 - ini_i] = W[len(n) - 1 - ini_i] - alpha * dJdW[len(n) - 1 - ini_i]
        # b[len(n) - 1 - ini_i] = b[len(n) - 1 - ini_i] - alpha * dJdb[len(n) - 1 - ini_i]
        # gamma[len(n) - 1 - ini_i] = gamma[len(n) - 1 - ini_i] - alpha * dJdGamma[len(n) - 1 - ini_i]
    j = j + 1
    # print(dJdW)
print("iteration times is:{0}".format(j))

show_ans(X, Y, m, ax1)

show_class(W, b, n, g, gamma, X_max, X_min, ax2)

show_ans(X, Y_tra, m, ax3)

show_err(ax4, err)

plt.show()
