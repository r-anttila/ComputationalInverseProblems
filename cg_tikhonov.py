import pylab
import numpy as np
import math
from astra import add_noise_to_sino
import numeric.radon as r

n = 128
theta = np.linspace(0, np.pi, 30, False)
phantom = {32:r.phantom32, 64:r.phantom64, 128:r.phantom128, 256:r.phantom}

def A(f):
    global theta
    return r.radon(f, theta)


def At(g):
    global theta
    return r.iradon(g, theta, n)


def Q(f, alpha):
    return At(A(f)) + alpha*np.eye(n)*f


def grad_descent(data, alpha):
    '''
    This method solves the normal equation (A*A+alphaI)f=A*g iteratively using gradient descent.
    We need to know the shape of the desired output
    hence we take it as a parameter f_shape and leave computing it outside of
    this function. The matrix g is given by data.
    '''
    lambdak = 1e-3
    # We take the simple backprojection of the data as an initial guess.
    fk = At(data)
    grad = At(A(fk)-data)+2*alpha*fk

    for _ in range(100):
        fOld = fk
        gradOld = grad

        fk = fk - lambdak*grad
        grad = At(A(fk)-data)+2*alpha*fk

        lambdak = (np.dot(fk.flatten()-fOld.flatten(), fk.flatten()-fOld.flatten())) / \
            (np.dot(fk.flatten()-fOld.flatten(), grad.flatten()-gradOld.flatten()))

        print(lambdak)

        fk[fk < 0] = 0

    return fk


def conjugate_gradient_tikhonov(data, f_shape):
    '''
    This method solves the normal equation (A*A+alphaI)f=A*g iteratively using the
    conjugate gradient method. We need to know the shape of the desired output
    hence we take it as a parameter f_shape and leave computing it outside of
    this function. The matrix g is given by data.
    '''

    alpha = 0.01
    fk = np.zeros(f_shape)
    grad = Q(fk, alpha) - At(data)
    dk = -grad

    for i in range(math.ceil(1/alpha)):
        Qdk = Q(dk, alpha)
        ak = (-np.dot(grad.transpose().flatten(), dk.flatten())) / \
            (np.dot(dk.transpose().flatten(), Qdk.flatten()))
        fk = fk + ak*dk
        prev_grad = grad
        grad = grad + ak*Qdk
        bk = (np.dot(grad.transpose().flatten(), grad.flatten())) / \
            (np.dot(prev_grad.transpose().flatten(), prev_grad.flatten()))
        dk = bk*dk - grad

    return fk


data = r.radon(phantom[n], theta)
noise_level = 0.01
noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])
tik_rec = conjugate_gradient_tikhonov(noisy_data, (n, n))
#gd_rec = grad_descent(noisy_data, 5)
bp = r.iradon(noisy_data, theta, n)


pylab.gray()
pylab.figure(0)
pylab.plot(np.linspace(0,n,n), tik_rec[n//2,:], 'r', np.linspace(0,n,n), phantom[n][n//2,:], 'b')
pylab.figure(2)
pylab.imshow(tik_rec)
pylab.show()
