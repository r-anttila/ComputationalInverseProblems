import pylab
import numpy as np
import math
from astra import add_noise_to_sino
from numeric import radon as r, tv_grad as tv
from tik_primal_dual import *
from TV_primal_dual import *
from LS_primal_dual import *

def mat_comp():
    tik_alpha = 1
    tv_alpha = 0.1

    # Creating measurement data
    data = r.radon(phantom[n], theta)

    A = r.radon_matrix(n_ang, n)
    At = np.transpose(A)

    # Adding noise
    noise_level = 0.05
    noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

    # Compute the reconstruction
    rec_tik = tik_primal_dual_mat_alt(A, At, noisy_data, (n, n), tik_alpha)
    rec_ls = LS_primal_dual_mat(A, At, noisy_data, (n,n))
    rec_tv = TV_primal_dual_mat(A,At,noisy_data,(n,n),tv_alpha)

    pylab.gray()
    pylab.figure(0)
    pylab.title("Least-Squares reconstruction")
    pylab.imshow(rec_ls)
    pylab.figure(1)
    pylab.title("Tikhonov reconstruction")
    pylab.imshow(rec_tik)
    pylab.figure(2)
    pylab.title("Total Variation reconstruction")
    pylab.imshow(rec_tv)

    pylab.show()

def comp():
    tik_alpha = 1

    # Creating measurement data
    data = r.radon(phantom[n], theta)

    # Adding noise
    noise_level = 0.05
    noisy_data = data + noise_level*np.random.randn(data.shape[0], data.shape[1])

    # Compute the reconstruction
    rec_tik = tik_primal_dual_alt(noisy_data, (n, n), tik_alpha)
    rec_ls = LS_primal_dual(noisy_data, (n,n))

    pylab.gray()
    pylab.figure(0)
    pylab.title("Least-Squares reconstruction")
    pylab.imshow(rec_ls)
    pylab.figure(1)
    pylab.title("Tikhonov reconstruction")
    pylab.imshow(rec_tik)


    pylab.show()

if __name__ == "__main__":
    #mat_comp()
    comp()