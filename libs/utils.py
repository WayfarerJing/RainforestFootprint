from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from keras import backend as K

#load images to images (# of samples, 256,256,3)
def load_rgb_sample(train_path, image_type):
    all_file_paths = [f for f in listdir(train_path) if isfile(join(train_path,f))]
    n_ch = 3 if image_type == 'jpg' else 4 
    images = np.empty((len(all_file_paths), 256, 256, n_ch))
    print(len(all_file_paths))
    print("#####")
    for n in range(1, len(all_file_paths)):
        images[n - 1,:,:,:] = cv2.imread(join(train_path,all_file_paths[n]))
    print(images.shape)    
    return images

def load_rgb(train_path, image_type):
  all_file_paths = [f for f in listdir(train_path) if isfile(join(train_path,f))]
  images = np.empty((len(all_file_paths) - 1, 256, 256, 3))
  for n in range(1, len(all_file_paths)):
    images[n - 1,:,:,:] = cv2.imread(join(train_path,all_file_paths[n]))
  return images

def get_train_tags_from_file(train_tag_path):
  return pd.read_csv(train_tag_path)

def get_lable_list(train_tags):
  label_list = []
  for tag_str in train_tags.tags.values:
      labels = tag_str.split(' ')
      for label in labels:
          if label not in label_list:
              label_list.append(label)
  return label_list


def load_y_lable(train_tag_path): 
  train_tags = get_train_tags_from_file(train_tag_path)
  label_list = get_lable_list(train_tags)
  for label in label_list:
      train_tags[label] = train_tags['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
  return train_tags

################ batch ####################

def load_batch_rgb(train_tags, path_dir, start_idx, batch_size, imgtype):
    batch_train_tags, batch_train_filename = batch_from_label(train_tags, start_idx, batch_size, imgtype)
    batch_train_img = load_image_lst(mydir='train_jpg', filename_lst=batch_train_filename, imgtype=imgtype)
    return batch_train_img, batch_train_tags

def batch_from_label(df_label, start_idx, n, suffix):
    batch = df_label[start_idx:start_idx + n]
    batch_img_name = [x + suffix for x in batch['image_name']]
    return batch, batch_img_name

################ random ###################

def load_labels(dir, filename):
    '''Load labels from the given path, return a data frame containing  '''
    # Import tag lists
    train_tags = pd.read_csv(dir + "/" + filename)

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

    return train_tags


def sample_from_label(df_label, n, suffix):
    ''' Randomly sample n from the label datafrom, return sampled image names (type: list)
        If n is None, then sample is the whole training set'''
    if n is not None :
        smp = df_label.sample(n)
    else:
        smp = df_label
    smp_img_name = [x + suffix for x in smp['image_name']]
    return smp, smp_img_name

def load_image(mydir, filename, imgtype):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    path = os.path.abspath(os.path.join(rootpath, mydir, filename))
    # path = os.path.join(rootpath, mydir, filename)
    if os.path.exists(path):
        print('Found image {}'.format(path))
        if imgtype == 'tif':
            return tiff.imread(path)
        else:
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))

def load_image_lst(mydir, filename_lst, imgtype):
    '''Load images given a list of filenames, return a list of image array '''
    img_lst = []
    for smp_nm in filename_lst:
        img_lst.append(load_image(mydir, smp_nm, imgtype))
    return img_lst

def load_random_rgb(path_dir, sample_size, imgtype):
    train_tags = load_labels(dir = path_dir, filename='train_v2.csv')
    smp_train_tags, smp_train_filename = sample_from_label(df_label=train_tags, n=sample_size, suffix=imgtype)
    smp_train_img = load_image_lst(mydir='train_jpg', filename_lst=smp_train_filename, imgtype=imgtype)
    return smp_train_img, smp_train_tags

def sample_from_label(df_label, n, suffix):
    ''' Randomly sample n from the label datafrom, return sampled image names (type: list)
        If n is None, then sample is the whole training set'''
    if n is not None :
        smp = df_label.sample(n)
    else:
        smp = df_label
    smp_img_name = [x + suffix for x in smp['image_name']]
    return smp, smp_img_name

def load_sample_y_lable(train_sample_path, train_tag_path):
    train_tags = get_train_tags_from_file(train_tag_path)
    smpNameList = listdir(train_sample_path)
    condition = False
    for img_nm in smpNameList:
        condition = np.logical_or(condition, train_tags['image_name'] + '.jpg' == img_nm)
    sampleTag = train_tags.loc[condition,]
    label_list = get_lable_list(sampleTag)
    for label in label_list:
        sampleTag[label] = sampleTag['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    return sampleTag

def load_random_sample_y_lable(image_name_list, train_tag_path):
    train_tags = get_train_tags_from_file(train_tag_path)
    condition = False
    condition = np.logical_or(condition, train_tags['image_name'] + '.jpg' == image_name_list)
    sampleTag = train_tags.loc[condition,]
    label_list = get_lable_list(sampleTag)
    for label in label_list:
        sampleTag[label] = sampleTag['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
    return sampleTag

  
def true_pos(y_true, y_pred):
    return K.sum(y_true * K.round(y_pred))

def false_pos(y_true, y_pred):
    return K.sum(y_true * (1. - K.round(y_pred)))

def false_neg(y_true, y_pred):
    return K.sum((1. - y_true) * K.round(y_pred))

def precision(y_true, y_pred):
    return true_pos(y_true, y_pred) / \
           (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))

def recall(y_true, y_pred):
    return true_pos(y_true, y_pred) / \
           (true_pos(y_true, y_pred) + false_neg(y_true, y_pred))

def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))
