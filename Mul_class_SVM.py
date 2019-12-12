import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

N = 100 # number of points per class
d = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d, N*C)) # data matrix (each row = single example)
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
def multi_class_svm_loss(W,X,y,reg):
    n = X.shape[1]
    Z= np.dot(W.T, X)
    margin_score = Z[y,np.arange(n)]
    margins = np.maximum(0, 1 - margin_score + Z)
    margins[y,np.arange(n)] = 0
    loss = np.sum(margins)
    loss /= n
    loss += 0.5*reg*np.sum(W*W)
    F =(margins>0).astype(int)
    F[y,np.arange(n)] = np.sum(-F, axis=0)
    dw = np.dot(X,F.T)/n + reg*np.sum(W)
    return loss,dw
W_init= np.random.randn(d,C)

def multi_class_svm(W_init,X,y,reg=0.01,eta=0.01,limit_iter=10000,batch=100):
    iter_ =0
    W=W_init
    W_tmp = 0
    while iter_ < limit_iter:
        batch_id = np.random.permutation(X.shape[1])[:batch].reshape(batch,)
        xi= X[:,batch_id]
        yi= y[batch_id]
        loss, dW = multi_class_svm_loss(W,xi,yi,reg)
        W-=eta*dW
        if np.linalg.norm(W-W_tmp) < 1e-100:
            return W,loss,iter_
        W_tmp = W
        if iter_ %1000==0:
            pass
        print('iter %d: '%iter_, loss)
        iter_ += 1
    return W,loss,iter_
def predict(X,W):
    Z= np.dot(W.T,X)
    return np.argmax(Z,axis=0)
W, loss, iter_=multi_class_svm(W_init,X,y,reg=0.01,batch=300,eta=0.001,limit_iter=10000)
y_pred = predict(X,W)
print(y_pred)
print(iter_)
print("Ti le nhan dang dung: ",accuracy_score(y,y_pred)*100,'%')
