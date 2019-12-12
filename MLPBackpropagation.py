import numpy as np
import  matplotlib.pyplot as plt
import math
from sklearn.metrics import  accuracy_score
from scipy import sparse
N = 100 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels
for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j

# lets visualize the data:
# plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)

plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 5);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 5);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize =5);
# plt.axis('off')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
plt.show()

def ReLU(Z):
    return np.maximum(Z,0)
def softmax(Z):
    e_Z = np.exp(Z- np.max(Z, axis=0, keepdims=True))
    return  e_Z/np.sum(e_Z, axis=0)
def error(Y,Y_pred):
    return -np.sum(Y*np.log(Y_pred))
def convert_label(y):
    Y= sparse.coo_matrix((np.ones_like(y),(y,np.arange(len(y)))),shape=(C,len(y))).toarray()
    return Y
d1=100
d2=3
W1= np.random.randn(d0,d1)
b1= np.random.randn(d1,1)
W2=np.random.randn(d1,d2)
b2= np.random.randn(d2,1)
Y= convert_label(y)
eta=0.5
for i in range(10000):
    #feedforward
    Z1= np.dot(W1.T, X)+ b1
    A1= ReLU(Z1)
    Z2= np.dot(W2.T, A1)+b2
    A2=Y_pred= softmax(Z2)

    if i%1000==0:
        print("iter: %d, error: %f"%(i,error(Y,Y_pred)/N))
    #backpropagation
    E2= (Y_pred-Y)/N
    dW2= np.dot(A1,E2.T)
    db2= np.sum(E2, axis=1, keepdims=True)
    E1= np.dot(W2, E2)
    E1[Z1<=0]=0 # gradient ReLU
    dW1 = np.dot(X,E1.T)
    db1= np.sum(E1, axis=1, keepdims=True)
    #update w , b
    W2+=-eta*dW2
    W1+=-eta*dW1
    b2+=-eta*db2
    b1+=-eta*db1

Z1= np.dot(W1.T, X)+b1
A1= ReLU(Z1)
Z2= np.dot(W2.T, A1)+b2
A2= softmax(Z2)
result = np.argmax(A2, axis=0)
print("Accuracy: %5f"%(accuracy_score(result,y)*100),"%")

