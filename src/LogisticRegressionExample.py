
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:

## In case of Squared Error function


# In[3]:

def f(x):
    w = 5.0
    return 1 / (1 + math.pow(math.e, -w * x))

def g(w, x):
    return 1 / (1 + math.pow(math.e, -w * x))


def perturb(y):
    mu, sigma = y, 0.05
    return np.random.normal(mu, sigma)

def makeData():
    '''
    x_list = list(np.random.uniform(-1, 1, 100))
    x_list.sort()
    data = map(lambda x: (x, perturb(f(x))), x_list)
    return data
    '''
    data = []
    for i in range(-100, 101, 1):
        if i <= -50:
            data.append((i/100.0, 0.0))
        elif i >= 50:
            data.append((i/100.0, 1.0))
    return data


def gradE(data_list, w):
    # return sum(map(lambda (x, y): float((g(w, x) - y) * g(w, x) * (1 - g(w, x)) * x), data))
    return sum(map(lambda (x, y): float((g(w, x) - y) * g(w, x) * (1 - g(w, x)) * x), data_list))

def updateW(data_list, w, eta):
    return w - eta * gradE(data_list, w)

data_list = makeData()
def searchW():
    ## data = makeData()
    w = 0.0
    eta = 1.0e-2

    # iter = 10000
    iter = 10000
    for i in range(iter):
        w = updateW(data_list, w, eta)
        if i % 1000 == 0:
            print w
    return w
    
w = searchW()
data_list2 = []
for data in data_list:
    data_list2.append((data[0], g(w, data[0])))
    ## print data[1], g(w, data[0])


# In[4]:

x_list = np.array(map(lambda x: x[0], data_list))
y_list = np.array(map(lambda x: x[1], data_list))
plt.plot(x_list, y_list, 'bo')
x_list = np.array(map(lambda x: x[0], data_list2))
y_list = np.array(map(lambda x: x[1], data_list2))
plt.plot(x_list, y_list, 'yo')
plt.show()


# In[5]:

## In case of Log Loss


# In[6]:

def f(x):
    w = 5.0
    return 1 / (1 + math.pow(math.e, -w * x))

def g(w, x):
    return 1 / (1 + math.pow(math.e, -w * x))

def makeData():
    data = []
    for i in range(-100, 101, 1):
        if i <= -50:
            data.append((i/100.0, 0.0))
        elif i >= 50:
            data.append((i/100.0, 1.0))
    return data

def gradE(data, w):
    x, y = data
    return (g(w, x) - y) * x

def updateW(data, w, eta):
    return w - eta * gradE(data, w)

data_list = makeData()
##print data_list
def searchW():
    ## data = makeData()
    w = 0.0
    eta = 1.0e-2

    for i in range(100):
        for data in data_list:
            w = updateW(data, w, eta)
    return w
    
w = searchW()
print w

data_list2 = []
for data in data_list:
    data_list2.append((data[0], g(w, data[0])))
    ## print data[1], g(w, data[0])


# In[7]:

x_list = np.array(map(lambda x: x[0], data_list))
y_list = np.array(map(lambda x: x[1], data_list))
plt.plot(x_list, y_list, 'bo')
x_list = np.array(map(lambda x: x[0], data_list2))
y_list = np.array(map(lambda x: x[1], data_list2))
plt.plot(x_list, y_list, 'yo')
plt.show()


# In[ ]:




# In[8]:

def vector_inner_product(w_list, x_list):
    result = 0.0
    for i in range(len(w_list)):
        result += w_list[i] * x_list[i]
    return result

def vector_minus(x_list, y_list):
    result_list = []
    for i in range(len(x_list)):
        result_list.append(x_list[i] - y_list[i])
    return result_list

def vector_scala_multiplication(c, x_list):
    result_list = []
    for i in range(len(x_list)):
        result_list.append(c * x_list[i])
    return result_list

def vector_multiplication(w_list, x_list):
    result_list = []
    for i in range(len(w_list)):
        result_list.append(w_list[i] * x_list[i])
    return result_list

def f(x_list):
    w_list = [5.0, 1.0]
    return 1 / (1 + math.pow(math.e, -1.0 * vector_inner_product(w_list, x_list)))

def g(w_list, x_list):
    return 1 / (1 + math.pow(math.e, -1.0 * vector_inner_product(w_list, x_list)))

def makeData():
    data = []
    for i in range(0, 200, 1):
        if i <= 70:
            data.append((i/100.0, 0.0))
        elif i >= 130:
            data.append((i/100.0, 1.0))
    return data

def gradE(data, w_list):
    x, y = data
    x_list = [x, 1.0]
    c = (g(w_list, x_list) - y)# * g(w_list, x_list) * (1.0 - g(w_list, x_list))
    return vector_scala_multiplication(c, x_list)

def updateW(data, w_list, eta_list):
    return vector_minus(w_list, vector_multiplication(eta_list, gradE(data, w_list)))

data_list = makeData()
## print data_list
def searchW():
    w_list = [0.0, 0.0]
    eta_list = [1.0e-2, 1.0e-2]

    for i in range(1000):
        for data in data_list:
            w_list = updateW(data, w_list, eta_list)
    return w_list
    
w_list = searchW()
print w_list

data_list2 = []
for data in data_list:
    data_list2.append((data[0], g(w_list, (data[0], 1.0))))
    ## print data[1], g(w, data[0])


# In[9]:

x_list = np.array(map(lambda x: x[0], data_list))
y_list = np.array(map(lambda x: x[1], data_list))
plt.plot(x_list, y_list, 'bo')
x_list = np.array(map(lambda x: x[0], data_list2))
y_list = np.array(map(lambda x: x[1], data_list2))
plt.plot(x_list, y_list, 'yo')
plt.show()


# In[10]:

data_source = [[8.79236, 8.79636, 4.70997, 5.8211, 12.9699],
[8.79137, 8.79236, 4.70217, 5.82558, 12.9733],
[8.81486, 8.79137, 4.68944, 5.83112, 12.9774],
[8.81301, 8.81486, 4.68558, 5.84046, 12.9806],
[8.90751, 8.81301, 4.64019, 5.85036, 12.9831],
[8.93673, 8.90751, 4.62553, 5.86464, 12.9854],
[8.96161, 8.93673, 4.61991, 5.87769, 12.99],
[8.96044, 8.96161, 4.61654, 5.89763, 12.9943],
[9.00868, 8.96044, 4.61407, 5.92574, 12.9992],
[9.03049, 9.00868, 4.60766, 5.94232, 13.0033],
[9.06906, 9.03049, 4.60227, 5.95365, 13.0099],
[9.05871, 9.06906, 4.5896, 5.9612, 13.0159],
[9.10698, 9.05871, 4.57592, 5.97805, 13.0212],
[9.12685, 9.10698, 4.58661, 6.00377, 13.0265],
[9.17096, 9.12685, 4.57997, 6.02829, 13.0351],
[9.18665, 9.17096, 4.57176, 6.03475, 13.0429],
[9.23823, 9.18665, 4.56104, 6.03906, 13.0497],
[9.26487, 9.23823, 4.54906, 6.05046, 13.0551],
[9.28436, 9.26487, 4.53957, 6.05563, 13.0634],
[9.31378, 9.28436, 4.51018, 6.06093, 13.0693],
[9.35025, 9.31378, 4.50352, 6.07103, 13.0737],
[9.35835, 9.35025, 4.4936, 6.08018, 13.077],
[9.39767, 9.35835, 4.46505, 6.08858, 13.0849],
[9.4215, 9.39767, 4.44924, 6.10199, 13.0918],
[9.44223, 9.4215, 4.43966, 6.11207, 13.095],
[9.48721, 9.44223, 4.42025, 6.11596, 13.0984],
[9.52374, 9.48721, 4.4106, 6.12129, 13.1089],
[9.5398, 9.52374, 4.41151, 6.122, 13.1169],
[9.58123, 9.5398, 4.3981, 6.13119, 13.1222],
[9.60048, 9.58123, 4.38513, 6.14705, 13.1266],
[9.64496, 9.60048, 4.3732, 6.15336, 13.1356],
[9.6439, 9.64496, 4.3277, 6.15627, 13.1415],
[9.69405, 9.6439, 4.32023, 6.16274, 13.1444],
[9.69958, 9.69405, 4.30909, 6.17369, 13.1459],
[9.68683, 9.69958, 4.30909, 6.16135, 13.152],
[9.71774, 9.68683, 4.30552, 6.18231, 13.1593],
[9.74924, 9.71774, 4.29627, 6.18768, 13.1579],
[9.77536, 9.74924, 4.27839, 6.19377, 13.1625],
[9.79424, 9.77536, 4.27789, 6.2003, 13.1664]]


# In[12]:

def vector_inner_product(w_list, x_list):
    result = 0.0
    for i in range(len(w_list)):
        result += w_list[i] * x_list[i]
    return result

def vector_minus(x_list, y_list):
    result_list = []
    for i in range(len(x_list)):
        result_list.append(x_list[i] - y_list[i])
    return result_list

def vector_scala_multiplication(c, x_list):
    result_list = []
    for i in range(len(x_list)):
        result_list.append(c * x_list[i])
    return result_list

def vector_multiplication(w_list, x_list):
    result_list = []
    for i in range(len(w_list)):
        result_list.append(w_list[i] * x_list[i])
    return result_list

def g(w_list, x_list):
    return 1 / (1 + math.pow(math.e, -1.0 * vector_inner_product(w_list, x_list)))

def makeData():
    data_list = []
    for data in data_source:
        x_list = data[1:]
        y = data[0]/10.0
        data_list.append([x_list, y])
    return data_list
        

def gradE(data, w_list):
    x_list, y = data
    x_list2 = x_list + [1.0]
    c = (g(w_list, x_list2) - y) * g(w_list, x_list2) * (1.0 - g(w_list, x_list2))
    return vector_scala_multiplication(c, x_list2)

def updateW(data, w_list, eta_list):
    return vector_minus(w_list, vector_multiplication(eta_list, gradE(data, w_list)))


data_list = makeData()

## print data_list
def searchW():
    w_list = [0.0, 0.0, 0.0, 0.0, 0.0]
    eta_list = [1.0e-2] * 5 #, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3]

    for i in range(100000):
        for data in data_list:
            w_list = updateW(data, w_list, eta_list)
    return w_list
    
w_list = searchW()
print w_list


# In[ ]:




# In[13]:

data_list2 = []
for data in data_list:
    ## data_list2.append((data[0], g(w_list, data[0] + [1.0])))
    print data[1], g(w_list, data[0]+[1.0])


# In[ ]:



