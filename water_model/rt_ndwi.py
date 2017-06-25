# Title: rt_ndwi.py
# Description: Functions regarding NDWI calculation
#               1. def ndwi: Calculate NDWI scores
#               2. def ndvi: Calculate NDVI scores
#               3. def n_greater_threshold: create a feature vector for the given image
#                   (# of pixels with value > threshold)
#               4. def all_n_greater_threshold: create feature vectors for a list of images
#               5. def rescale: Rescale each pixel value x of img to alpha*(x+gamma)+beta (visualization purpose only)


import numpy as np

def ndwi(img_tif):
    """ Calculate NDWI for a single image """
    # Extract green band and NIR band
    g = img_tif[:, :, 1]
    nir = img_tif[:, :, 3]
    # Change from int to float
    g = g.astype('float')
    nir = nir.astype('float')
    # Return NDWI, ignore zero dividend error
    np.seterr(divide='ignore')
    return (g-nir)/(g+nir)


def ndvi(img_tif):
    """ Calculate NDVI for a single image """
    # Extract green band and NIR band
    r = img_tif[:, :, 2]
    nir = img_tif[:, :, 3]
    # Change from int to float
    r = r.astype('float')
    nir = nir.astype('float')
    # Return NDWI, ignore zero dividend error
    np.seterr(divide='ignore')
    return (nir-r)/(nir+r)


def n_greater_threshold(img, thred=[0]):
    ''' calculate number of pixels > thredshold '''
    vec = []
    for t in thred:
        vec.append((1*(img > t)).sum())
    return vec


def all_n_greater_threshold(imglist, thred=[0]):
    results = np.empty((len(imglist), len(thred)))
    for _ind_ in range(0, len(imglist)):
        _tmp_img  = imglist[_ind_]
        # _tmp_n = n_greater_threshold(_tmp_img, thred)
        results[_ind_, ] = n_greater_threshold(_tmp_img, thred)
    return results


def img_rescale(img, alpha=1, beta=0, gamma=0):
    """ Rescale each pixel value x of img to alpha*(x+gamma)+beta """
    return alpha*(img+gamma)+beta

