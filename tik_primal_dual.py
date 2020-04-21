import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv

n_ang = 30
n = 64
phantom = {32:r.phantom32, 64:r.phantom64, 128:r.phantom128, 256:r.phantom}

theta = np.linspace(0, np.pi, n_ang, False)

def Af(f):
    # Define the forward projection with given number of angles

    global theta
    return r.radon(f, theta)


def Atf(g):
    # Define the backprojection with given number of angles

    global theta
    return r.iradon(g, theta, n)


def norm(f_shape):
    '''
    Computes the L2-norm of the radon transform. According to Sidky et al this should converge to the
    norm within 20 iterations so we stop there.
    '''
    xk = np.ones(f_shape)

    for i in range(21):
        xk = Atf(Af(xk))
        xk = xk/np.linalg.norm(xk)

    norm = np.linalg.norm(Af(xk))

    return norm


def norm_mat(A, At, f_shape):
    '''
    Computes the L2-norm of the radon transform. According to Sidky et al this should converge to the
    norm within 20 iterations so we stop there.
    '''
    xk = np.ones(f_shape).flatten()

    for i in range(21):
        xk = At.dot(A.dot(xk))
        xk = xk/np.linalg.norm(xk)

        norm = np.linalg.norm(A.dot(xk))

    return norm

def tik_primal_dual(meas, f_shape, alpha, show_prog=False):
    '''
    Computes the least squares minimizer with respect to f  to the problem 1/2 norm(Af - g)^2, where norm denotes 
    the L2-norm, using the primal-dual method. In this case A is the radon transform, f is the reconstruction we want to
    obtain and g is the measurement data given by the parameter meas.
    '''

    # We compute the L2-norm of the radon transform using the power method
    L = norm(f_shape)
    sigma = 1/L
    tau = 1/L
    Theta = 1

    # Initialize the solution pk to the dual problem to zero
    pk = np.zeros(meas.shape)
    # Initialize the solution fk to the primal problem to zero
    fk = np.zeros(f_shape)
    ft = fk

    for i in range(1001):
        pk = (2*(pk+sigma*(Af(ft) - meas)))/(sigma+2)
        f_prev = fk
        fk = (fk-tau*Atf(pk))/(2*alpha*tau+1)
        fk[fk<0] = 0 #Positivity constraint
        ft = fk + Theta*(fk-f_prev)

        if show_prog:
            if i % 50 == 0:
                gap = np.linalg.norm(Af(fk)-meas)**2 + alpha*np.linalg.norm(fk)**2 + 1/(4*alpha)*np.linalg.norm(Atf(pk))**2 + np.dot(pk.flatten(),meas.flatten()) +1/4*np.linalg.norm(pk)**2
                pylab.figure(1)
                pylab.title("Iteration {} | Dual gap: {:.2f}".format(i, gap))
                pylab.gray()
                pylab.imshow(fk)
                pylab.pause(0.1)

    return fk

def tik_primal_dual_mat(A, At, meas, f_shape, alpha, show_prog=False):
    '''
    Computes the least squares minimizer with respect to f  to the problem norm(Af - g)^2+alpha*norm(f)^2, where norm denotes 
    the L2-norm, using the primal-dual method. In this case A is the radon transform, f is the reconstruction we want to
    obtain and g is the measurement data given by the parameter meas.
    '''

    # We compute the L2-norm of the radon transform using the power method
    L = norm_mat(A,At,f_shape)
    sigma = 1/L
    tau = 1/L
    Theta = 1

    # Initialize the solution pk to the dual problem to zero
    pk = np.zeros(meas.shape).flatten()
    # Initialize the solution fk to the primal problem to zero
    fk = np.zeros(f_shape).flatten()
    ft = fk
    meas_v = meas.flatten()

    for i in range(2001):
        pk = (2*(pk+sigma*(A.dot(ft) - meas_v)))/(sigma+2)
        f_prev = fk
        fk = (fk-tau*At.dot(pk))/(2*alpha*tau+1)
        fk[fk < 0] = 0  # Positivity constraint
        ft = fk + Theta*(fk - f_prev)

        if show_prog:
            if i % 50 == 0:
                gap = np.linalg.norm(A.dot(ft)-meas_v)**2 + 1/(4*alpha)*np.linalg.norm(-At.dot(pk))**2 + np.dot(pk,meas_v) +1/4*np.linalg.norm(pk)**2
                pylab.figure(2)
                pylab.title("Iteration {} | Dual gap: {:.2f}".format(i, gap))
                pylab.gray()
                pylab.imshow(np.reshape(ft,f_shape))
                pylab.pause(0.1)

    return np.reshape(ft, f_shape)


if __name__ == "__main__":

    # Creating measurement data
    data = r.radon(phantom[n], theta)

    #A = r.radon_matrix(n_ang, n)
    #At = np.transpose(A)

    # Adding noise
    noise_level = 0.01
    noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

    # Compute the reconstruction
    rec = tik_primal_dual(noisy_data, (n, n), 5, True)
    #rec = tik_primal_dual_mat(A, At, noisy_data, (n,n), 0.01, True)

    pylab.gray()
    pylab.figure(0)
    pylab.imshow(phantom[n])
    pylab.figure(1)
    pylab.imshow(rec)
    pylab.figure(3)
    pylab.plot(np.linspace(0,n,n), rec[n//2,:], 'r', np.linspace(0,n,n), phantom[n][n//2,:], 'b')


    pylab.show()