import numpy as np
import cv2
from utils import *

def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[3]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)

# ==== convert RGB image in black and white (# of samples, 256,256,1)
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[3]==3)
    bn_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1],rgb.shape[2],1))
    return bn_imgs

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[3]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
