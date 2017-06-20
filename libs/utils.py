from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

#load images to images (# of samples, 256,256,3)
def load_rgb(train_path):
  all_file_paths = [f for f in listdir(train_path) if isfile(join(train_path,f))]
  images = np.empty((len(all_file_paths) - 1, 256, 256, 3))
  for n in range(1, len(all_file_paths)):
    images[n - 1,:,:,:] = cv2.imread(join(train_path,all_file_paths[n]))
  return images

def get_train_tags_from_file(train_tag_path):
  return pd.read_csv (train_tag_path)

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
