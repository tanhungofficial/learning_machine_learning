from cvxopt import matrix, solvers
import numpy as np
from numpy import array
from  math import log, exp

def linear_programming():

    c = matrix([-5.,-3.])
    G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
    h = matrix([10., 16., 32., 0., 0.])
    solvers.options['show_progress'] = False
    sol= solvers.lp(c,G,h)
    print('Solution of LP:\n', sol['x'],sep="")

def quadratic_programming():
    P = matrix([[1., 0.], [0., 1.]])
    q = matrix([-10., -10.])
    G = matrix([[1., 2., 1., -1., 0.], [1., 1., 4., 0., -1.]])
    h = matrix([10., 16., 32., 0., 0])

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    print('Solution of QP:')
    print(sol['x'])

def geometric_programming():
    K = [4]
    F = matrix([[-1., 1., 1., 0.],
                [-1., 1., 0., 1.],
                [-1., 0., 1., 1.]])
    g = matrix([log(40.), log(2.), log(2.), log(2.)])
    solvers.options['show_progress'] = False
    sol = solvers.gp(K, F, g)

    print('Solution of GP:')
    print(np.exp(np.array(sol['x'])))

    print('\nchecking sol^5')
    print(np.exp(np.array(sol['x']))**5)
linear_programming()
quadratic_programming()
geometric_programming()

