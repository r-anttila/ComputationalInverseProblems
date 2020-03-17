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


def norm(f_shape):
    # TODO: Implement the algorithm computing the transform.
    '''
    Computes the L2-norm of the radon transform. According to Sidky et al this should converge to the
    norm within 20 iterations so we stop there.
    '''
    pass


def TV_primal_dual(meas, f_shape):
    # TODO: Implement the primal dual algorithm for TV reconstruction.
    '''
    Computes the least squares minimizer with respect to f  to the problem 1/2 norm(Af - g)^2, where norm denotes 
    the L2-norm, using the primal-dual method. In this case A is the radon transform, f is the reconstruction we want to
    obtain and g is the measurement data given by the parameter meas.
    '''

    pass


# Creating measurement data
data = r.radon(r.phantom, theta)

# Adding noise
noise_level = 0.1
noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

pylab.gray()
pylab.figure(0)
pylab.imshow(r.phantom)

pylab.show()
