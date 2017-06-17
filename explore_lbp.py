# 0. Import sample images and tag list
# 1. Change to gray scale
# 2. Enhancement
# 3. Extract LBP feature

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import exposure as ep
from skimage import feature as ft


# 0. Import
DataDir = 'G:\Kaggle-amazon'
# Import JPG images
SmpNameList = os.listdir(os.path.join(DataDir, 'train-jpg-sample'))
SampleImg = []
for fl in SmpNameList:
    SampleImg.append(io.imread(os.path.join(DataDir, 'train-jpg-sample', fl)))

# Import tag lists
train_tags = pd.read_csv('G://Kaggle-amazon//train_v2.csv//train_v2.csv')

# Build list with unique labels
label_list = []
for tag_str in train_tags.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

# Add onehot features for every label
for label in label_list:
    train_tags[label] = train_tags['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Labels for sample images
condition = False
for img_nm in SmpNameList:
    condition = np.logical_or(condition, train_tags['image_name']+'.jpg' == img_nm)
SampleTag = train_tags.loc[condition, ]

# ---------------------------------------------------------------------------------------
# Try on one sample
img = SampleImg[SmpNameList.index('train_1.jpg')]
lbl = train_tags.loc[train_tags['image_name'] == 'train_1', 'tags']

# plt.imshow(img)
# lbl

# 1. Change to grayscale
img_gray = rgb2gray(img)
# plt.imshow(img_gray)

# 2. Enhancement
# Gamma adjusted
img_gamma =ep.adjust_gamma(img_gray)
# plt.imshow(img_gamma)

# Histogram Equalization (Use overall training set to build a filter?)
# Contrast stretching
p2, p98 = np.percentile(img_gray, (2, 98))
img_rescale = ep.rescale_intensity(img_gray, in_range=(p2, p98))
# plt.imshow(img_rescale)

# Equalization
img_eq = ep.equalize_hist(img_gray)
# plt.imshow(img_eq)

# Adaptive Equalization
img_adapteq = ep.equalize_adapthist(img_gray, clip_limit=0.03)
# plt.imshow(img_adapteq)

# Divide into patches
def sub_patches(img, p_size):
    '''return a list of patches with size p_size * p_size from img '''
    patches = []
    for loc_x in range(0, 256, p_size):
        for loc_y in range(0, 256, p_size):
            patches.append(img[loc_x:(loc_x+p_size), loc_y:(loc_y+p_size)])
    return(patches)
patches = sub_patches(img_gamma, 64)

# 3. LBP features (use histogram as vector)
# lbp = ft.local_binary_pattern(img_gamma, 8, 4)
# lbp_hist = np.histogram(lbp)
# plt.hist(lbp.ravel(), bins=256, range=(0,256))

# plt.imshow(lbp)
# f, axarr = plt.subplots(4, 4)
lbp = []
for ind in range(0,16):
    _tmp_img_lbp = ft.local_binary_pattern(patches[ind], 8, 4)
    # xar = ((ind + 1) // 4) - 1
    # yar = ((ind + 1) % 4) - 1
    # axarr[xar, yar].hist(_tmp_img_lbp.ravel(), bins=256, range= (0,256))
    lbp.append(np.histogram(_tmp_img_lbp, bins=256, range = (0, 256))[0])
