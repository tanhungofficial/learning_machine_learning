from mnist import MNIST
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import sleep
mntrain = MNIST("D:\\Learing Document\\Machine Learning\\MNIST\\")
mntrain.load_training()
Xtrain_all = np.array(mntrain.train_images)
ytrain_all = np.array(mntrain.train_labels)

mntest = MNIST("D:\\Learing Document\\Machine Learning\\MNIST")
mntest.load_testing()
Xtest_all = np.array(mntest.test_images)
ytest_all = np.array(mntest.test_labels)
def extract_data(X, y, class_):
	y_id_reg=np.array([])
	for i in class_[0]:
		y_id_reg= np.hstack((y_id_reg, np.where(y==i)[0]))
	n0= len(y_id_reg)

	for i in class_[1]:
		y_id_reg=np.hstack((y_id_reg, np.where(y==i)[0]))
	n1= len(y_id_reg)-n0
	y_id_reg= y_id_reg.astype(int)
	Xtrain= np.array(X[y_id_reg])
	ytrain= np.asarray([0]*n0+[1]*n1)
	return Xtrain, ytrain

def display_error(error_id,Xtest):
 	X_error=np.array([])
 	plt.axis('off')
 	for i in error_id:
 		A = Xtest[i,:].reshape(28,28)
 		plt.imshow(A, interpolation='nearest' )
 		plt.gray()
 		plt.title(i)
 		plt.show()

class_=[[0],[1]]
Xtrain, ytrain= extract_data(Xtrain_all, ytrain_all,class_)
Xtest, ytest= extract_data(Xtest_all,ytest_all,class_)
logreg= linear_model.LogisticRegression().fit(Xtrain,ytrain)

result=logreg.predict(Xtest)
print(">>>Accurary: ", accuracy_score(result,ytest)*100, "%", sep="")
error_id = np.array(np.where(result!=ytest)).T
print(">>>Number of error: ", len(error_id))
print(">>>index of error: \n",error_id.T)
display_error(error_id,Xtest)
