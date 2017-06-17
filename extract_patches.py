import numpy as np
import random
import ConfigParser

from utils import load_rgb
from utils import group_images

from pre_processing import my_PreProc

import keras

#Load the original data and return the extracted patches for training/testing
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV):
    train_imgs_original = load_rgb(DRIVE_train_imgs_original)
    #TODO: load y lables
    train_masks = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=2)

    train_imgs = my_PreProc(train_imgs_original)
    #train_masks = train_masks/255.

    print "\ntrain images/masks shape:"
    print train_imgs.shape
    print "train images range (min-max): " +str(np.min(train_imgs)) +' - '+str(np.max(train_imgs))

    #extract the TRAINING patches from the full images
    patches_imgs_train, patches_masks_train = extract_random(train_imgs,train_masks,patch_height,patch_width,N_subimgs,inside_FOV)
    data_consistency_check(patches_imgs_train, patches_masks_train)

    print "\ntrain PATCHES images/masks shape:"
    print patches_imgs_train.shape
    print "train PATCHES images range (min-max): " +str(np.min(patches_imgs_train)) +' - '+str(np.max(patches_imgs_train))

    return patches_imgs_train, patches_masks_train#, patches_imgs_test, patches_masks_test


#extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print "N_patches: please enter a multiple of number of samples"
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==2)  #4D array and 1D label
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    assert (full_imgs.shape[0] == full_masks.shape[0])
    patches = np.empty((N_patches,patch_h,patch_w,full_imgs.shape[3]))
    patches_masks = np.empty((N_patches,full_masks.shape[1]))
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print "patches per full image: " +str(patch_per_img)
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            if inside==True:
                if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
                    continue
            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2),:]
            patch_mask = full_masks[i]
            patches[iter_tot]=patch
            patches_masks[iter_tot,:]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks
