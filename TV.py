import pylab
import numpy as np
import math
from astra import add_noise_to_sino
import numeric.radon as r

theta = np.linspace(0, np.pi, 180, False)


def A(f):
    global theta
    return r.radon(f, theta)


def At(g):
    global theta
    return r.iradon(g, theta, 256)


def Q(f, alpha):
    return At(A(f)) + alpha*np.eye(256)*f


def grad_hor(f):
    return np.append(f[:, 1:]-f[:, :-1], np.zeros((256, 1)), axis=1)


def grad_vert(f):
    return np.append(f[1:, :]-f[:-1, :], np.zeros((1, 256)), axis=0)


def div_hor(f):
    return np.append(np.zeros((256, 1)), -f[:, :-1] + f[:, 1:], axis=1)


def div_vert(f):
    return np.append(np.zeros((1, 256)), -f[:-1, :] + f[1:, :], axis=0)


def tv_grad(f, beta):
    grad_hor_f = grad_hor(f)
    grad_vert_f = grad_vert(f)

    nomi_hor = grad_hor_f / np.sqrt(grad_hor_f**2 + grad_vert_f**2 + beta)
    nomi_vert = grad_vert_f / np.sqrt(grad_hor_f**2 + grad_vert_f**2 + beta)

    return div_hor(nomi_hor) + div_vert(nomi_vert)


def projected_grad_desc_TV(meas, f_size):
    '''
    Computes the Total Variation regularised reconstruction for the data using the
    projected gradient descent method.
    '''
    alpha = 10
    beta = 1e-6
    lambdak = 1e-4
    fk = At(meas)
    grad = (2 * At(A(fk)-meas)) - alpha*tv_grad(fk, beta)

    for i in range(200):
        fOld = fk
        gradOld = grad

        fk = fk - lambdak*grad
        fk[fk < 0] = 0
        grad = (2 * At(A(fk)-meas)) - alpha*tv_grad(fk, beta)

        lambdak = (np.dot(fk.flatten()-fOld.flatten(), fk.flatten()-fOld.flatten())) / \
            (np.dot(fk.flatten()-fOld.flatten(), grad.flatten()-gradOld.flatten()))

        if i % 10 == 0:

            pylab.figure(44)
            pylab.gray()
            pylab.title("Iteration {}".format(i))
            pylab.imshow(fk)
            pylab.pause(0.1)

    return fk


data = r.radon(r.phantom, theta)
noisy_data = add_noise_to_sino(data, 1e6)

rec = projected_grad_desc_TV(data, (256, 256))

# pylab.gray()
# pylab.figure(0)
# pylab.imshow(At(data))
# pylab.figure(1)
# pylab.imshow(noisy_data)
# pylab.figure(2)
# pylab.imshow(rec)
pylab.figure(0)
pylab.imshow(r.phantom)

pylab.show()
