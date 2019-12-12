import pandas as pd 
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt 
#import matplotlib.pyplot as plt

df = pd.read_csv("Advertising.csv", header=0)
X=np.array(df.values[0:150,0:3])
y=np.array(df.values[0:150,3])
X_test=np.array(df.values[150:,0:3])
y_test=np.array(df.values[150:,3])
one_array=np.ones((X.shape[0],1))
Xbar = np.concatenate((X,one_array),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)
w=np.dot(np.linalg.pinv(A),b)
def predict(X,w):
    result = w[3]
    for i in range(3):
        result+= X[i]*w[i]

    return print("Ket qua du doan la: ",result)
predict(X_test[30],w)
print("Ket qua thuc te la:", y_test[30])

#Nghiem lai bang sklearn
regr= linear_model.LinearRegression(fit_intercept=False).fit(Xbar,y)
print(regr.coef_)
print(w)