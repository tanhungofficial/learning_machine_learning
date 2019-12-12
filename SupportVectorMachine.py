import numpy as np
import matplotlib.pyplot as plt
from  cvxopt import matrix, solvers
np.random.seed(22)
means = [[2, 2], [3.5, 2]]
cov = [[.3, .2], [.2, .3]]
N=30

X0= np.random.multivariate_normal(means[0],cov,N)
X1= np.random.multivariate_normal(means[1],cov,N)

X= np.concatenate((X0,X1),axis=0).T
y= np.concatenate((np.ones((1,N)), -1*np.ones((1,N))),axis=1)
plt.scatter(X0[:,0],X0[:,1],20,"r",'o')
plt.scatter(X1[:,0],X1[:,1],20,'g','s')
# Build quaratic programming
V= np.concatenate((X0, -1*X1),axis=0).T
K= matrix(np.dot(V.T, V))
q = matrix(-np.ones((2*N,1)))
G= matrix(-1*np.eye(2*N))
h= matrix(np.zeros((2*N,1)))
A= matrix(y)
b= matrix(np.zeros((1,1)))
solvers.options['show_progress']=False
qp = solvers.qp(K,q,G,h,A,b)
lambda_= np.array(qp["x"])
print('Lambda: \n', lambda_.T)
S= np.where(lambda_>1e-6)[0]
XS= X[:,S]
lambda_S= lambda_[S,:]
VS= V[:,S]
yS=y[:,S]
w = np.dot(VS, lambda_S)
b = np.mean(yS.T-np.dot(w.T,XS))

print("Result by your code: ")
print('W:', w.T)
print('b:',b)

from sklearn.svm import  SVC
clf = SVC(C=1e5, kernel='linear')
clf.fit(X.T, y.reshape(2*N,))
print('Result by scikit-learn:')
print('w: ',clf.coef_)
print('b: ',clf.intercept_)

def draw_svm(w,b,color='b'):
    x= np.arange(0.5,5,0.1)
    y= -w[0,:]*x/w[1,:] - b/w[1,:]
    plt.plot(x,y, color, linewidth=3)
    for i in range(lambda_S.shape[0]):
        b_svm = -np.dot(w.T,XS[:,i])
        y_svm= -w[0,:]*x/w[1,:] - b_svm/w[1,:]
        plt.plot(x,y_svm,color, linewidth=1)

draw_svm(w,b)
plt.show()


