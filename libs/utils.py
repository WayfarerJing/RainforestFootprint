from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#load images to images (# of samples, 256,256,3)
def load_rgb(infile):
  all_file_paths = [f for f in listdir(infile) if isfile(join(infile,f))]
  images = np.empty((len(all_file_paths) - 1, 256, 256, 3))
  for n in range(1, len(all_file_paths)):
    images[n - 1,:,:,:] = cv2.imread(join(infile,all_file_paths[n]))
  return images

#convert RGB image in black and white (# of samples, 256,256,1)
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[3]==3)
    bn_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1],rgb.shape[2],1))
    return bn_imgs
