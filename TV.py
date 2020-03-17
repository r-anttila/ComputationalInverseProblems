import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv

theta = np.linspace(0, np.pi, 60, False)


def A(f):
    global theta
    return r.radon(f, theta)


def At(g):
    global theta
    return r.iradon(g, theta, 256)


def Q(f, alpha):
    return At(A(f)) + alpha*np.eye(256)*f


def projected_grad_desc_TV(meas, f_size):
    '''
    Computes the Total Variation regularised reconstruction for the data using the
    projected gradient descent method.
    '''
    alpha = 1
    beta = 1e-8
    lambdak = 1e-4
    fk = At(meas)
    grad = (2 * At(A(fk)-meas)) - alpha*tv.tv_grad(fk, beta)

    for i in range(301):
        fOld = fk
        gradOld = grad

        fk = fk - lambdak*grad
        fk[fk < 0] = 0
        grad = (2 * At(A(fk)-meas)) - alpha*tv.tv_grad(fk, beta)

        lambdak = (np.dot(fk.flatten()-fOld.flatten(), fk.flatten()-fOld.flatten())) / \
            (np.dot(fk.flatten()-fOld.flatten(), grad.flatten()-gradOld.flatten()))

        if i % 50 == 0:

            pylab.figure(44)
            pylab.gray()
            pylab.title("Iteration {}".format(i))
            pylab.imshow(fk)
            pylab.pause(0.1)

    return fk


data = r.radon(r.phantom, theta)
#noisy_data = add_noise_to_sino(data, 1e4)
noisy_data = data + 0.01*np.random.randn(data.shape[0], data.shape[1])

rec = projected_grad_desc_TV(noisy_data, (256, 256))

# pylab.gray()
pylab.figure(6)
pylab.imshow(At(data))
pylab.figure(0)
pylab.imshow(noisy_data)
# pylab.figure(2)
# pylab.imshow(rec)
pylab.figure(1)
pylab.imshow(r.phantom)

pylab.show()
