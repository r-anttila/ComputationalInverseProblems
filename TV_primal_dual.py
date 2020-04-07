import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv

n_ang = 180
n = 256

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

    xk = np.ones(f_shape)

    for i in range(21):
        xk = Atf(Af(xk)) - tv.divgrad(xk)
        xk = xk/np.linalg.norm(xk)

        norm = np.sqrt(np.linalg.norm(Af(xk))**2+np.linalg.norm(np.append(tv.grad_hor(xk), tv.grad_vert(xk),axis=0))**2)

    return norm

def norm_mat(A, At, f_shape):

    xk = np.ones(f_shape)

    for i in range(21):
        xk = At.dot(A.dot(xk.flatten())) - tv.divgrad(np.reshape(xk, f_shape)).flatten()
        xk = xk/np.linalg.norm(xk)

        norm = np.sqrt(np.linalg.norm(A.dot(xk))**2+np.linalg.norm(np.append(tv.grad_hor(np.reshape(xk, f_shape)), tv.grad_vert(np.reshape(xk, f_shape)),axis=0))**2)

    return norm

def TV_primal_dual(meas, f_shape, alpha, show_prog=False):
    '''
    Computes the Total Variation regularized reconstruction to the problem. i.e the minimizer to 1/2*L2norm(Af-g)^2+lambda*L1norm(|grad(u)|)
    '''

    L = norm(f_shape)
    sigma = 1/L
    tau = 1/L
    Theta = 1
    # Initialize the solution pk to the dual problem to zero
    pk = np.zeros(meas.shape)
    # Initialize the solution fk to the primal problem to zero
    fk = np.zeros(f_shape)
    #Initialize the horizontal component of the q to zero
    qk_hor = np.zeros(f_shape)
    #Initialize the vertical component of the q to zero
    qk_vert = np.zeros(f_shape)
    ft = np.reshape(fk, f_shape)
    
    for i in range(2001):
        pk = (pk+sigma*(Af(ft)-meas))/(1+sigma)
        f_prev = fk
        grad, qk_hor, qk_vert = tv.tv_grad(ft, qk_hor, qk_vert, alpha, sigma)
        fk = fk-tau*Atf(pk) + tau*grad
        fk[fk<0] = 0
        ft = np.reshape(fk + Theta*(fk-f_prev), f_shape)
        if show_prog:
            if i % 50 == 0:
                gap = 1/2*np.linalg.norm(Af(ft)-meas)**2 +alpha*np.sum(np.sqrt(tv.grad_hor(ft)**2+tv.grad_vert(ft)**2)) + 1/2*np.linalg.norm(pk)**2+np.dot(pk.flatten(),meas.flatten())
                at = np.linalg.norm(Atf(pk).flatten()-grad.flatten(), np.inf)
                pylab.figure(2)
                pylab.title("Iteration {} | Dual gap: {:.2f} | A^tp: {:.2f}".format(i, gap, at))
                pylab.gray()
                pylab.imshow(ft)
                pylab.pause(0.1)

    return ft


def TV_primal_dual_mat(A, At, meas, f_shape, alpha, show_prog=False):
    '''
    Computes the Total Variation regularized reconstruction to the problem. i.e the minimizer to 1/2*L2norm(Af-g)^2+lambda*L1norm(|grad(u)|)
    '''

    L = norm_mat(A,At,f_shape)
    sigma = 1/L
    tau = 1/L
    Theta = 1
    # Initialize the solution pk to the dual problem to zero
    pk = np.zeros(meas.shape).flatten()
    # Initialize the solution fk to the primal problem to zero
    fk = np.zeros(f_shape).flatten()
    #Initialize the horizontal component of the q to zero
    qk_hor = np.zeros(f_shape)
    #Initialize the vertical component of the q to zero
    qk_vert = np.zeros(f_shape)
    ft = np.reshape(fk, f_shape)
    meas_v = meas.flatten()
    
    for i in range(1001):
        pk = (pk+sigma*(A.dot(ft.flatten())-meas_v))/(1+sigma)
        f_prev = fk
        grad, qk_hor, qk_vert = tv.tv_grad(ft, qk_hor, qk_vert, alpha, sigma)
        fk = fk-tau*At.dot(pk.flatten()) + tau*grad.flatten()
        fk[fk<0] = 0
        ft = np.reshape(fk + Theta*(fk-f_prev), f_shape)
        if show_prog:
            if i % 50 == 0:
                gap = 1/2*np.linalg.norm(A.dot(ft.flatten())-meas_v)**2 +alpha*np.sum(np.sqrt(tv.grad_hor(ft)**2+tv.grad_vert(ft)**2)) + 1/2*np.linalg.norm(pk)**2+np.dot(pk,meas_v)
                at = np.linalg.norm(At.dot(pk)-grad.flatten(), np.inf)
                pylab.figure(2)
                pylab.title("Iteration {} | Dual gap: {:.2f} | A^tp: {:.2f}".format(i, gap, at))
                pylab.gray()
                pylab.imshow(ft)
                pylab.pause(0.1)

    return ft

if __name__ == "__main__":

    # Creating measurement data
    data = r.radon(phantom[n], theta)

    #A = r.radon_matrix(n_ang, n)
    #At = np.transpose(A)

    # Adding noise
    noise_level = 0.05
    noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

    #rec = TV_primal_dual_mat(A, At, noisy_data, (n,n), 0.05, True)

    rec = TV_primal_dual(noisy_data, (n,n), 0.1, True)

    pylab.gray()
    pylab.figure(0)
    pylab.imshow(phantom[n])
    pylab.figure(3)
    pylab.plot(np.linspace(0,n,n), rec[n//2,:], 'r', np.linspace(0,n,n), phantom[n][n//2,:], 'b')

    pylab.show()