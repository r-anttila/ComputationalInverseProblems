import pylab
import scipy.io
import astra
import numpy as np
import numpy.linalg as nl

# Load a 256x256 phantom image
phantom = scipy.io.loadmat('phantom.mat')['phantom256']


def radon(f, theta):
    '''
    Takes a numpy.ndarray containing the discretisation of the data and theta
    containing the angles that the radon transform will be taken with 
    and returns a numpy.ndarray containing the radon transformed data i.e a
    sinogram of f.
    '''

    vol_geom = astra.create_vol_geom(f.shape[0], f.shape[1])
    proj_geom = astra.create_proj_geom('parallel', 1.0, 367, theta)
    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    f_id = astra.data2d.create('-vol', vol_geom, f)
    sin_id = astra.data2d.create('-sino', proj_geom)

    cfg = astra.astra_dict("FP")
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = sin_id
    cfg["VolumeDataId"] = f_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    sin_data = astra.data2d.get(sin_id)
    astra.algorithm.delete(alg_id)

    #sin_id, sin_data = astra.create_sino(f, proj_id)
    astra.data2d.delete(f_id)
    astra.data2d.delete(sin_id)
    astra.projector.delete(proj_id)

    return sin_data


def iradon(g, theta, n):
    '''
    Takes a numpy.ndarray g that was created with theta angles and applies the 
    inverse radon transform with no filters to it. Returns the backprojected data.
    '''
    vol_geom = astra.create_vol_geom(n)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 367, theta)
    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    f_id = astra.data2d.create('-vol', vol_geom)
    g_id = astra.data2d.create('-sino', proj_geom, g)

    cfg = astra.astra_dict("BP")
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = g_id
    cfg["ReconstructionDataId"] = f_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec_data = astra.data2d.get(f_id)
    astra.algorithm.delete(alg_id)

    #f_id, rec_data = astra.create_backprojection(g, proj_id)
    astra.data2d.delete(f_id)
    astra.data2d.delete(g_id)
    astra.projector.delete(proj_id)

    return rec_data


def fbp(g, theta, n, filter='ram-lak'):
    '''
    Computes the filtered backprojection of the given numpy.ndarray g with the
    desired filter.
    '''
    vol_geom = astra.create_vol_geom(n)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 367, theta)
    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    f_id = astra.data2d.create('-vol', vol_geom)
    g_id = astra.data2d.create('-sino', proj_geom, g)

    cfg = astra.astra_dict("FBP")
    cfg["ProjectorId"] = proj_id
    cfg["ProjectionDataId"] = g_id
    cfg["ReconstructionDataId"] = f_id
    cfg['option'] = {}
    cfg['option']["FilterType"] = filter

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    rec_data = astra.data2d.get(f_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(f_id)
    astra.data2d.delete(g_id)
    astra.projector.delete(proj_id)

    return rec_data
