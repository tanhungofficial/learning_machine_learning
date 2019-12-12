import numpy as np 
import matplotlib.pyplot as plt 
import math 

def fx(x):
	return x*x + 10*np.sin(x)

def grad(x):
	return 2*x + 10*np.cos(x)

x= np.arange(-10,10,0.1)
y= np.array(fx(x))
plt.plot(x,y,10,"r")
plt.axis([-6,6,-15,40])


def gradident_descent(x_init, grad, fx,eta):
	x=[x_init]
	x_new=0
	iter_=0
	while iter_ <1000:
		x_new= x[-1] - eta*grad(x[-1])
		if abs(fx(x[-1])-fx(x_new))< 1e-3:
			break
		iter_+=1
		x.append(x_new)
	return x,fx(np.array(x)) , iter_

def gd_momentum(x_init,grad,fx,eta, gama):
	x=[x_init]
	x_new=0
	v_bef=np.zeros_like(x_init)
	iter_=0
	while iter_ <1000:
		v_cur = gama*v_bef+ eta*grad(x[-1])
		x_new= x[-1] - v_cur
		if abs(fx(x[-1])-fx(x_new))< 1e-3:
			break
		iter_+=1
		x.append(x_new)
		v_bef= v_cur
	return x, fx(np.array(x)), iter_

def gd_nag(x_init,grad,fx,eta, gama):
	x=[x_init]
	x_new=0
	v_bef=np.zeros_like(x_init)
	iter_=0
	while iter_ <1000:
		v_cur = gama*v_bef+ eta*grad(x[-1]-gama*v_bef)
		x_new= x[-1] - v_cur
		if abs(fx(x[-1])-fx(x_new))< 1e-3:
			break
		iter_+=1
		x.append(x_new)
		v_bef= v_cur
	return x, fx(np.array(x)), iter_

x_gd,y_gd ,iter_gd = gradident_descent(6,grad,fx, 0.1)
x_gdm,y_gdm, iter_gdm = gd_momentum(-50,grad,fx,0.1,0.9)
x_nag,y_nag, iter_nag = gd_momentum(60,grad,fx,0.1,0.9)

#print(x_min, y_min , iter_ )
plt.scatter(x_gd[-1],y_gd[-1], 50,"r","o")
plt.scatter(x_gdm[-1], y_gdm[-1], 50,"g", "o")
plt.scatter(x_nag[-1], y_nag[-1], 20,"b", "o")
plt.show()
print("gradident_descent: ",x_gd[-1], y_gd[-1] , iter_gd)
print("gd_momentum: ",x_gdm[-1], y_gdm[-1], iter_gdm)
print("gd_momentum: ",x_nag[-1], y_nag[-1], iter_nag)