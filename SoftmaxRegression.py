import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from scipy import sparse
from sklearn.metrics import accuracy_score
mean=[[2,2], [4,2], [3,3]]
conv = [[.1,0],[0,.1]]
N=50
C=3
X0= np.random.multivariate_normal(mean[0],conv, N)
X1= np.random.multivariate_normal(mean[1],conv, N)
X2= np.random.multivariate_normal(mean[2],conv, N)
X= np.concatenate((X0,X1,X2), axis=0)
X= np.concatenate((np.ones([3*N,1]),X),axis=1).T
y= np.asarray([0]*N+[1]*N+[2]*N)
Y= sparse.coo_matrix((np.ones_like(y),(y,np.arange(len(y)))), shape=(C,len(y))).toarray()
d= X.shape[0]
X.
print(Y)
print(Y.shape)
W_init= np.random.randn(d,C)
e= 0.1*np.random.randn(d,3*N)
X_test = X-e
y_test = y
def softmax(X,W):
	Z= np.dot(W.T,X)
	e_Z= np.exp(Z)
	A=e_Z/e_Z.sum(axis=0,keepdims=True)
	return A
def predict(X,W):
	A= softmax(X,W)
	return np.argmax(A, axis= 0)
def display_softmax(X,result,mis_point):
	X0=X[:,result==0]
	X1=X[:,result==1]
	X2=X[:,result==2]
	mis_point=X[:, result!= y]
	plt.scatter(mis_point[1,:],mis_point[2,:],30,'r','s')
	plt.scatter(X0[1,:],X0[2,:],10,"r",'o')
	plt.scatter(X1[1,:],X1[2,:],10,"b",'s')
	plt.scatter(X2[1,:],X2[2,:],10,"g",'^')
	plt.axis("equal")
	plt.title('Softmax Regression')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.show()

def softmax_regression(X,Y,W_init,learning_rate=0.05, stop_condition= 1e-4, iter_limit=10000, check_w_after=10):
	W = [W_init]
	N = X.shape[1]
	d= X.shape[0]
	C= Y.shape[0]
	iter_=0
	while iter_ < iter_limit:
		mix_id = np.random.permutation(N)
		count=0
		for i in mix_id:
			Xi= X[:,i].reshape(d,1)
			Yi= Y[:,i].reshape(C,1)
			Ai=softmax(Xi,W[-1])
			W_new= W[-1]+ learning_rate*np.dot(Xi,(Yi- Ai).T)
			count+=1
			if count%check_w_after==0:
				if np.linalg.norm(W[-check_w_after]-W_new) < stop_condition:
					return W[-check_w_after], iter_
			W.append(W_new)
		iter_+=1
	return W[-1], iter_
W, iter_ = softmax_regression(X,Y,W_init)
result= predict(X_test,W)
mis_point=X[:, result!= y_test]
print("Find out W: \n", W)
print("Number of loop: ",iter_)
print("Predictial result: \n",result)
print("Number of missing: ", mis_point.shape[1])
print("Accuracy: ", accuracy_score(result,y_test)*100, "%", sep="")
display_softmax(X_test,result,mis_point)