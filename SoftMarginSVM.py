import  numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from scipy.spatial.distance import cdist
np.random.seed(2)
mean =[[2,2],[2.2,1]]
conv = [[.3,.2],[.2,.3]]
N=20
C=100
X0 = np.random.multivariate_normal(mean[0],conv, N)
X1= np.random.multivariate_normal(mean[1], conv, N)

X= np.concatenate((X0.T, X1.T), axis=1)
y= np.concatenate((np.ones((1,N)), -np.ones((1,N))), axis=1)
plt.scatter(X0[:,0], X0[:,1], 10, 'r', 'o')
plt.scatter(X1[:,0], X1[:,1], 10, 'g','s')

def sklearn(C=100):
    cls = SVC(C=C, kernel='linear')
    cls.fit(X.T, y.T)
    return cls.coef_, cls.intercept_
w_sk, b_sk = sklearn()

def draw_svm(w,b,color='b'):
    x1= np.arange(2,4,0.1)
    y1= -w[0,:]*x1/w[1,:] - b/w[1,:] 
    plt.plot(x1,y1, color, linewidth=1)
# Metohd duality
def duality():

    V= matrix(np.concatenate((X0.T, -X1.T),axis=1))
    K= matrix(np.dot(V.T,V))
    q = matrix(-np.ones((2*N,1)))
    G= matrix(np.concatenate((-np.eye(2*N),np.eye(2*N)), axis=0))
    h= matrix(np.concatenate((np.zeros((2*N,1)),C*np.ones((2*N,1))), axis=0))
    A= matrix(y)
    b= matrix(np.zeros((1,1)))
    solvers.options['show_progress']= False
    qp=  solvers.qp(K, q, G,h, A,b)
    lambda_ = np.array(qp['x'])
    S = np.where(lambda_ >1e-5)[0]
    M= np.array([i for i in S if lambda_[i,:]< 99.9 ])
    VS= np.array(V)[:,S]
    lambda_S= lambda_[S,:]

    XM= X[:,M]
    yM= y[:,M]
    w_dual = np.dot(VS, lambda_S)
    b_dual = np.mean(yM - np.dot(w_dual.T,XM))
    return  w_dual,b_dual, lambda_.reshape(5,int(0.4*N))

w_dual, b_dual, lambda_ = duality()

#Bai toan khong rang buoc
X_ext = np.concatenate((np.ones((1,X.shape[1])),X), axis=0)
Z= np.concatenate((X_ext[:,:N], -X_ext[:,N:]), axis=1)
w_init = np.random.randn(X_ext.shape[0],1)
lamda= 1./C
def cost(w):
    u= np.dot(w.T,Z)
    cost= np.sum(np.maximum(0,1-u)) + 0.5*lamda*np.sum(w*w)  - 0.5*lamda*w[-0]*w[-0] # no bias
    return cost
def grad_hinge_loss(w):
    u = np.dot(w.T, Z)
    H= np.where(u<1)[1]
    g= -np.sum(Z[:,H],axis=1,keepdims=True) + lamda*w
    g[0] -= lamda*w[0]
    return g
def gradient_descent(w_init,eta=0.01,limit_iter=100000):
    w=w_init
    iter_ = 0
    while iter_ < limit_iter:
        cost_= cost(w)
        if iter_%10000==0:
            print('iter %d: %5f'%(iter_, cost_))
        g= grad_hinge_loss(w)
        w-= eta*g
        if np.linalg.norm(g) < 1e-5:
            return w, iter_
        iter_+=1
    return w, iter_

w_ext,iter_= gradient_descent(w_init)
w_gd = w_ext[1:]
b_gd = w_ext[0]
print('Result by scikit-learn:')
print("W: ",w_sk)
print('b: ', b_sk)
print('Result by duality:')
print('W: ',w_dual.reshape(1,X.shape[0]))
print('b: ', b_dual)
print('Result by hinge loss:')
print('w: ', w_gd.reshape(1,2))
print('b: ', b_gd)
draw_svm(w_gd, b_gd)
draw_svm(w_sk.T,b_sk,color='r')
draw_svm(w_dual, b_dual, color='g')
plt.show()

