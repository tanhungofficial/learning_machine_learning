import numpy as np
import  cv2
import scipy.io as sio
from sklearn.svm import SVC
from  sklearn.metrics import accuracy_score

img = cv2.imread('Chipu.png',0)
img = cv2.resize(img,(18,18))
cv2.imshow("Image test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
img = np.array(img).reshape(1,324)
img = img[0,:300].reshape(1,300)
data = sio.loadmat('myARgender.mat')
X_train = data['Y_train'].T
X_test = data['Y_test'].T
y_train = data['label_train'].T
y_test = data['label_test'].T

poly = SVC(kernel='poly',degree=4,gamma=4, coef0=1)
poly.fit(X_train,y_train)
poly_pred =poly.predict(X_test)

rbf = SVC(kernel='rbf',gamma=4)
rbf.fit(X_train,y_train)
rbf_pred= rbf.predict(X_test)

sigmoid = SVC(kernel='sigmoid', gamma=0.5, coef0=0.5)
sigmoid.fit(X_train,y_train)
sigmoid_pred = sigmoid.predict(X_test)

print('Ti le nhan dang dung su dung poly: %3f'%(accuracy_score(poly_pred,y_test)*100),"%")
print('Ti le nhan dang dung su dung rbf: %3f'%(accuracy_score(rbf_pred,y_test)*100),"%")
print('Ti le nhan dang dung su dung sigmoid: %3f'%(accuracy_score(sigmoid_pred,y_test)*100),"%")

print(poly.predict(img))
print(rbf.predict(img))
print(sigmoid.predict(img))

