import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import sparse
from time import  sleep
import cv2
from os import listdir
from sklearn.preprocessing import Imputer
path = 'D:\Learing Document\Machine Learning\VN-celeb\VN-celeb'
folder_names = np.array(listdir('D:\Learing Document\Machine Learning\VN-celeb\VN-celeb'),dtype=np.uint32)
folder_names.sort(axis=-1)
folder_names= folder_names[:10]
def create_path_labels(path,folder_names):
    out_path=[]
    out_labels=[]
    for i in folder_names:
        path_folder = path+'\\'+ str(i)
        images_name= listdir(path_folder)
        for j in images_name:
            path_image= path_folder +'\\'+ j
            out_path.append(path_image)
            out_labels.append(i-1)
    return out_path, np.array(out_labels)
images_path, image_labels = create_path_labels(path, folder_names)
def read_images_labels(images_path,image_labels,type=0):
    X= np.zeros((16384,1))
    y=[]
    for i in range(len(images_path)):
        img = cv2.imread(images_path[i],type)
        if len(img)==16384:
            img = np.array(img)
            img = img.reshape(img.size,1)

        else:
            img= cv2.resize(img,(128,128))
            img= np.array(img)
            img= img.reshape(img.size,1)
        X = np.concatenate((X, img), axis=1)
        y.append(image_labels[i])
    return np.array(X[:,1:]), np.array(y)
X, y = read_images_labels(images_path, image_labels)
imp= Imputer(missing_values=np.nan,strategy="mean").fit(X)
X= imp.transform(X)
#Khoi tao gia tri neural network
C=len(folder_names)
N=len(images_path)
eta=0.1
def create_d_b_W_A( n_sample, n_classification,n_feature=16384):
    d=np.zeros(n_classification,dtype=np.uint32)
    d[0] = n_feature
    d[1:-1] = n_sample
    d[-1] = n_classification
    W=[[],]
    b=[[],]
    A=[X,]
    for i in range(1,n_classification,1):
        Wi= np.random.randn(d[i-1],d[i])
        bi= np.random.randn(d[i],1)
        W.append(Wi)
        b.append(bi)
        A.append([])
    return d,b,W,A
def convert_labels(y,C):
    Y= sparse.coo_matrix((np.ones_like(y),(y,np.arange(len(y)))),shape=(C,len(y))).toarray()
    return Y
Y = convert_labels(y,C)
def softmax(Z):
    e_Z = np.exp(Z- np.max(Z, axis=0, keepdims=True))
    return e_Z/np.sum(e_Z, axis=0)
def ReLU(s):
    return np.maximum(s,0)
def create_emty_list(n):
    list_=[]
    for i in range(n):
        list_.append(np.random.randn(3,3))
    return list_
def error(Y,Y_pred):
    return -np.sum(Y*np.log(np.abs(Y_pred)))
def my_network(iter_limit=10000):
    d, b, W, A = create_d_b_W_A(N, C)
    Z=create_emty_list(C)
    E=create_emty_list(C)
    dW=create_emty_list(C)
    db= create_emty_list(C)
    for i in range(iter_limit):
        for k in range(1,C,1):
            Z[k] = np.dot(np.array(W[k]).T, np.array(A[k - 1])) - np.array(b[k]).reshape(d[k],1)
            if k<(C-1):
                A[k]= ReLU(np.array(Z[k]))
            else:
                A[k]= softmax(np.array(Z[k]))
        if i%1==0:
            cost= error(Y,A[-1])
            print("iter %d: %5f"%(i,cost))
        print(A[-1])
        print(Y)
        E[-1]= np.array((A[-1]-Y)/N)
        for k in range(1,C,1):
            if k>1:
                E[C-k]= np.array(np.dot(W[C-k+1],E[C-k+1]))
                E[C-k][Z[C-k]==0] = 0
            db[C-k]= np.array(np.sum(E[C-k], axis=1,keepdims=True))
            dW[C-k]=np.array(np.dot(A[C-k-1],E[C-k].T))

        for k in range(1,C,1):
            W[k] = np.array(W[k])- eta*dW[k]
            b[k] = np.array(b[k]) - eta*db[k]
my_network()













