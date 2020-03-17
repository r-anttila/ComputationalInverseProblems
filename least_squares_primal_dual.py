import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv

theta = np.linspace(0, np.pi, 60, False)


def A(f):
    # Define the forward projection with given number of angles

    global theta
    return r.radon(f, theta)


def At(g):
    # Define the backprojection with given number of angles

    global theta
    return r.iradon(g, theta, 256)


def radon_norm(f_shape):
    '''
    Computes the L2-norm of the radon transform. According to Sidky et al this should converge to the
    norm within 20 iterations so we stop there.
    '''
    xk = np.ones(f_shape)

    for i in range(21):
        xk = At(A(xk))
        xk = xk/np.linalg.norm(xk, 2)

        norm = np.linalg.norm(A(xk), 2)

    return norm


def lq_primal_dual(meas, f_shape):
    '''
    Computes the least squares minimizer with respect to f  to the problem 1/2 norm(Af - g)^2, where norm denotes 
    the L2-norm, using the primal-dual method. In this case A is the radon transform, f is the reconstruction we want to
    obtain and g is the measurement data given by the parameter meas.
    '''

    # We compute the L2-norm of the radon transform using the power method
    L = radon_norm((256, 256))
    sigma = 1/L
    tau = 1/L
    Theta = 1

    # Initialize the solution pk to the dual problem to zero
    pk = np.zeros(meas.shape)
    # Initialize the solution fk to the primal problem to zero
    fk = np.zeros(f_shape)
    ft = fk

    for i in range(301):
        pk = (pk+sigma*(A(ft) - meas))/(1+sigma)
        f_prev = fk
        fk = fk - tau*At(pk)
        fk[fk < 0] = 0  # Positivity constraint
        ft = fk + Theta*(fk - f_prev)

        if i % 50 == 0:
            gap = 1/2*np.linalg.norm(A(ft)-meas, 2)**2 + 1/2 * \
                np.linalg.norm(pk, 2)**2 + np.dot(pk.flatten(), meas.flatten())
            pylab.figure(1)
            pylab.title("Iteration {} | Dual gap: {}".format(i, round(gap, 1)))
            pylab.gray()
            pylab.imshow(ft)
            pylab.pause(0.1)

    return ft


# Creating measurement data
data = r.radon(r.phantom, theta)

# Adding noise
noise_level = 0.1
noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

# Compute the reconstruction
rec = lq_primal_dual(noisy_data, (256, 256))


pylab.gray()
pylab.figure(0)
pylab.imshow(r.phantom)
pylab.figure(2)
pylab.imshow(At(noisy_data))

pylab.show()
