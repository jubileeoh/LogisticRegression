
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
    iter = 100
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

    for i in range(10):
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



