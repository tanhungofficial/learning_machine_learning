import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd 
from time import sleep
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

df_org = pd.read_csv("LogisticRegression.csv", header=0)
imp = Imputer(missing_values=np.nan, strategy= "mean").fit(df_org)
df = imp.transform(df_org)
x= np.array(df[0:200,:-2])
y= np.array(df[0:200,-1])
one=np.ones([x.shape[0],1])
X=np.concatenate((x,one),axis=1)
w_init=np.random.randn(X.shape[1],1)

x_test= np.array(df[200:350,:-2])
X_test= np.concatenate((x_test,np.ones([x_test.shape[0],1])), axis=1)
y_test= np.array(df[200:350,-1], dtype=np.int64)
def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(X,w):
    N= len(X)
    z=np.dot(X,w)
    print("Fx Values:\n", z.reshape(1,N), sep="")
    p=sigmoid(z)
    print("Probability\n",p.reshape(1,N), sep="")
    
    result=[]
    for i in range(N):
        if p[i]>=0.5:
            result.append(1)
        else:
            result.append(0)
    return result
def logistic_regression(X,y,w_init,eta,tol,limit_iter):
    w=[w_init]
    N= X.shape[0]
    d= X.shape[1]
    w_tmp=np.array([])
    iter_=0
    while iter_<limit_iter:
        w_tmp= np.zeros([d,1])
        for i in range(N):
            xi=X[i].reshape(d,1)
            yi=y[i]
            zi=sigmoid(np.dot(xi.T,w[-1]))
            eta*(zi-yi)*xi
            w_new=w[-1] + eta*(yi-zi)*xi
            w_tmp+=w_new
        w_tmp= w_tmp/N
        if np.linalg.norm(w[-1]-w_tmp)<tol:
            return w[-1], iter_
        w.append(w_tmp)
        iter_+=1
    return w[-1],iter_

w,iter_=logistic_regression(X,y,w_init,0.001,1e-4,10000)
result=np.array(predict(X_test,w))
print("Predicted result: \n",result.reshape(1,X_test.shape[0]),sep="")
print("Real result: \n", y_test.reshape(1,X_test.shape[0]),sep="")
print("\n Find out w: \n", w,"\nNumber of scan period: ", iter_)
print("\n Accuracy_score: ", accuracy_score(result,y_test)*100,"%", sep="")
















'''m __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
import  pandas as pd
np.random.seed(2)


df = pd.read_csv("LogisticRegression.csv", header=0)
X= np.array(df.values[0:300,1:7]).T
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

y= np.array(df.values[0:300,7]).T

'''
#X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
 #             2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
#y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
'''
# extended data 


def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
print(sigmoid(np.dot(w[-1].T, X)))
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 5)
plt.plot(X1, y1, 'bs', markersize = 5)

xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
'''