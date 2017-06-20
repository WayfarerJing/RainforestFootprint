import numpy as np
import random
import ConfigParser

from utils import load_rgb

from pre_processing import my_PreProc

def get_y_label_sample(path_dir, train_sample, train_label):
    train_sample_path, train_tag_path =  path_dir + train_sample, path_dir + train_label
    y_sample_train_label = load_sample_y_lable(train_sample_path, train_tag_path)
    return y_sample_train_label

#Load the original data
def get_data_training_sample(path_dir, train_sample, train_label):
    train_imgs_original_sample = load_rgb(path_dir + train_sample)
    train_masks = keras.utils.to_categorical(get_y_label_sample(path_dir, train_sample, train_label)['primary'], 2)

    train_imgs = my_PreProc(train_imgs_original_sample)

    print "\ntrain images shape:"
    print train_imgs.shape
    print "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))

    print "\ntrain masks shape:"
    print train_masks.shape

    return train_imgs_original_sample, train_masks


def get_data_training(path_dir, train_full_img, train_label):
    train_imgs_original = load_rgb(path_dir + train_full_img)
    train_masks = keras.utils.to_categorical(load_y_lable(path_dir, train_label)['primary'], 2)

    train_imgs = my_PreProc(train_imgs_original)

    print "\ntrain images shape:"
    print train_imgs.shape
    print "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))

    print "\ntrain masks shape:"
    print train_masks.shape

    return train_imgs_original, train_masks  

