from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

def load_rgb(infile):
  all_file_paths = [f for f in listdir(infile) if isfile(join(infile,f))]
  images = np.empty((len(all_file_paths) - 1, 256, 256, 3))
  for n in range(1, len(all_file_paths)):
    images[n - 1,:,:,:] = cv2.imread(join(infile,all_file_paths[n]))
  return images
