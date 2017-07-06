import numpy as np
import random
import ConfigParser
from keras.utils import to_categorical
from utils import *

from pre_processing import my_PreProc

def get_y_label_sample(path_dir, train_sample, train_label):
    train_sample_path, train_tag_path =  path_dir + train_sample, path_dir + train_label
    y_sample_train_label = load_sample_y_lable(train_sample_path, train_tag_path)
    return y_sample_train_label

def load_y_lable(train_tag_path):  # Import tag lists
  train_tags = get_train_tags_from_file(train_tag_path)
  label_list = get_lable_list(train_tags)
  for label in label_list:
      train_tags[label] = train_tags['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
  return train_tags

#Load the original data
def get_data_training_sample(path_dir, train_sample):
    train_imgs_original_sample = load_rgb(path_dir + train_sample, image_type)
    train_masks = to_categorical(load_sample_y_lable(path_dir, train_sample)['primary'], 2)
    train_imgs = my_PreProc(train_imgs_original_sample)
    print "\ntrain images shape:"
    print train_imgs.shape
    print "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))
    print "\ntrain masks shape:"
    print train_masks.shape
    return train_imgs_original_sample, train_masks


def get_data_training(path_dir, train_full_img, train_tag, image_type):
    train_imgs_original = load_rgb(path_dir + train_full_img, image_type)
    train_masks = to_categorical(load_y_lable(path_dir + train_tag)['primary'], 2)
    train_imgs = my_PreProc(train_imgs_original)
    print "\ntrain images shape:"
    print train_imgs.shape
    print "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))
    print "\ntrain masks shape:"
    print train_masks.shape
    return train_imgs, train_masks  

def get_data_random_training(path_dir, sample_size, image_type):
    train_imgs_original, train_masks = load_random_rgb(path_dir, sample_size,train_tag, image_type)
    train_imgs = my_PreProc(train_imgs_original)
    print "\ntrain images shape:"
    print train_imgs.shape
    print "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs))
    print "\ntrain masks shape:"
    print train_masks.shape
    return train_imgs, train_masks
