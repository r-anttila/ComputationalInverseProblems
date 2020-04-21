import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv

n = 256
theta = np.linspace(0, np.pi, 60, False)
phantom = {32:r.phantom32, 64:r.phantom64, 128:r.phantom128, 256:r.phantom}


def A(f):
    global theta
    return r.radon(f, theta)


def At(g):
    global theta
    return r.iradon(g, theta, n)


def Q(f, alpha):
    return At(A(f)) + alpha*np.eye(n)*f


def projected_grad_desc_TV(meas, f_size):
    '''
    Computes the Total Variation regularised reconstruction for the data using the
    projected gradient descent method.
    '''
    alpha = 1
    beta = 1e-8
    lambdak = 1e-4
    fk = At(meas)
    grad = (2 * At(A(fk)-meas)) - alpha*tv.smoothed_tv_grad(fk, beta)

    for i in range(1001):
        fOld = fk
        gradOld = grad

        fk = fk - lambdak*grad
        fk[fk < 0] = 0
        grad = (2 * At(A(fk)-meas)) - alpha*tv.smoothed_tv_grad(fk, beta)

        lambdak = (np.dot(fk.flatten()-fOld.flatten(), fk.flatten()-fOld.flatten())) / \
            (np.dot(fk.flatten()-fOld.flatten(), grad.flatten()-gradOld.flatten()))

        if i % 50 == 0:

            pylab.figure(44)
            pylab.gray()
            pylab.title("Iteration {}".format(i))
            pylab.imshow(fk)
            pylab.pause(0.1)

    return fk


data = r.radon(phantom[n], theta)
noise_level = 0.1
noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

rec = projected_grad_desc_TV(noisy_data, (n, n))

# pylab.gray()
pylab.figure(6)
pylab.imshow(At(data))
pylab.figure(0)
pylab.imshow(noisy_data)
# pylab.figure(2)
# pylab.imshow(rec)
pylab.figure(1)
pylab.plot(np.linspace(0,n,n), rec[n//2,:], 'r', np.linspace(0,n,n), phantom[n][n//2,:], 'b')

pylab.show()
