import numpy as np


def grad_hor(f):
    return np.append(f[:, 1:]-f[:, :-1], np.zeros((f.shape[0], 1)), axis=1)


def grad_vert(f):
    return np.append(f[1:, :]-f[:-1, :], np.zeros((1, f.shape[0])), axis=0)


def div_hor(f):
    return np.append(np.zeros((f.shape[0], 1)), -f[:, :-1] + f[:, 1:], axis=1)


def div_vert(f):
    return np.append(np.zeros((1, f.shape[0])), -f[:-1, :] + f[1:, :], axis=0)


def smoothed_tv_grad(f, beta):
    grad_hor_f = grad_hor(f)
    grad_vert_f = grad_vert(f)

    nomi_hor = grad_hor_f / np.sqrt(grad_hor_f**2 + grad_vert_f**2 + beta)
    nomi_vert = grad_vert_f / np.sqrt(grad_hor_f**2 + grad_vert_f**2 + beta)

    return div_hor(nomi_hor) + div_vert(nomi_vert)

def tv_grad(f,q_hor, q_vert, alpha, sigma):
    grad_hor_f = grad_hor(f)
    grad_vert_f = grad_vert(f)

    denomi = np.sqrt((q_hor+sigma*grad_hor_f)**2 + (q_vert+sigma*grad_vert_f)**2)
    denomi[denomi < alpha] = alpha
    nomi_hor = alpha*(q_hor+sigma*grad_hor_f) / denomi
    nomi_vert = alpha*(q_vert+sigma*grad_vert_f) / denomi

    return div_hor(nomi_hor) + div_vert(nomi_vert), nomi_hor, nomi_vert

def div(f):

    return div_hor(f) + div_vert(f)

def divgrad(f):

    return div_hor(grad_hor(f)) + div_vert(grad_vert(f))