from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import numpy as np
from numpy import linalg as LA

m, n = 2, 3
A = np.random.rand(m, n)

U, S, V = LA.svd(A)
print ('Frobenius norm of (UU^T - I) =',LA.norm(U.dot(U.T) - np.eye(m)))
print ('\n', S, '\n')
print ('Frobenius norm of (VV^T - I) =',LA.norm(V.dot(V.T) - np.eye(n)))


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

img = mpimg.imread('building2.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

# to gray
gray = 0.2125* img[:, :, 0] + 0.7154 *img[:, :, 1] + 0.0721 *img[:, :, 2]
plt.imshow(gray)
plt.axis('off')
plt.show()

from numpy import linalg as LA

U, S, V = LA.svd(gray)
from matplotlib.backends.backend_pdf import PdfPages

# percentage of preserving energy
with PdfPages('energy_preserved.pdf') as pdf:
    a = np.sum(S**2)
    b = np.zeros_like(S)
    for i in range(S.shape[0]):
        b[i] = np.sum(S[:i+1]**2, axis = 0)/a
## error
e =  1- b

def approx_rank_k(U, S, V, k):
    Uk = U[:, :k]
    Sk = S[:k]
    Vk = V[:k, :]
    return np.around(Uk.dot(np.diag(Sk)).dot(Vk))

# A = gray
# U, S, V = LA.svd(A)
A1 = []
for k in range(5, 100, 5):
    A1.append(approx_rank_k(U, S, V,k))
# show results
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
def update(i):
    ani = plt.cla()
    ani = plt.imshow(A1[i])
    label = '$k$ = %d: error = %.4f' %(5*i + 5, e[i])
    ax.set_xlabel(label)
    ani = plt.axis('off')
    ani = plt.title(label)

    return ani, ax

anim = FuncAnimation(fig, update, frames=np.arange(0, len(A1)), interval=500)
anim.save('a.gif', dpi = 300, writer = 'imagemagick')
plt.show()