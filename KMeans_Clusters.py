import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist #tinh khoang cach giua cac cap diem trong 2 tap hop
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0)
org_labels= np.asarray([0]*100+[1]*100+ [2]*100+[3]*100+ [4]*100)
K=3
N=150


def kmeans_init_center(X,K):
    return X[np.random.choice(X.shape[0],K)]

def kmeans_assign_lanels(X,centers):
    d= cdist(X,centers)
    return np.argmin(d,axis=1)

def kmeans_update_centers(X,labels, K):
    new_centers=[]
    for k in range(K):
        Xk= X[labels==k,:]
        new_centers.append(np.mean(Xk, axis=0))
    return new_centers
def has_convergred(centers, new_centers):
    return np.linalg.norm(centers)==np.linalg.norm(new_centers)
def kmeans_clusters(X,K):
    centers=[kmeans_init_center(X,K)]
    iter_=0
    while iter_<1000:
        labels=kmeans_assign_lanels(X,centers[-1])
        new_centers= kmeans_update_centers(X,labels,K)
        if has_convergred(centers[-1], new_centers):
            break
        iter_+=1
        centers.append(new_centers)
    return np.array(centers[-1]), np.array(labels), iter_
def show_centers(centers):
    n= np.shape(centers)[0]
    for k in range(n):
        plt.scatter(list(centers[k])[0],list(centers[k])[1],50,"y","s")

def kmeans_display(X, labels,centers):
    K = np.amax(labels) + 1
    X0 = X[labels == 0, :]
    X1 = X[labels == 1, :]
    X2 = X[labels == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 2, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 2, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 2, alpha = .8)

    plt.axis('equal')
    plt.title("Các điểm sau khi phân chia")
    plt.xlabel("Hoành độ x")
    plt.ylabel("Tung độ y")
    show_centers(centers)
    plt.show()

centers,pred_labels, iter_=kmeans_clusters(X,K)
kmeans_display(X,pred_labels,centers)
print("Tọa độ các centers: \n", centers.reshape(K,X.shape[1]))
print("Số lần lặp: ", iter_)

