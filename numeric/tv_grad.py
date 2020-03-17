import numpy as np


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
