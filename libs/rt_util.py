# Title: rt_util.py
# Description: This is a util file storing useful functions
#               1. def load_image: load image from specified path and file
#               2. def load_image_lst: load image from specified path and file list
#               3. def load_labels: load tags, create 0/1 label for each tag
#               4. def sample_from_label: Randomly select n samples from the label dataframe,
#                                         return a list of sample image file names
#               5. def precision_recall: calculate precision and recall
#               6. def f_score: calculate f_score, by default it's f2


import os
from skimage import io
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

rootpath = 'G://Kaggle-amazon'

# def sample_images(tags, n=None):
#     """Randomly sample n images with the specified tags."""
#     condition = True
#     if isinstance(tags, string_types):
#         raise ValueError("Pass a list of tags, not a single tag.")
#     for tag in tags:
#         condition = condition & labels_df[tag] == 1
#     if n is not None:
#         return labels_df[condition].sample(n)
#     else:
#         return labels_df[condition]
#
#


def load_image(mydir, filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    path = os.path.abspath(os.path.join(rootpath, mydir, filename))
    if os.path.exists(path):
        print('Found image {}'.format(path))
        return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))


def load_image_lst(mydir, filename_lst):
    '''Load images given a list of filenames, return a list of image array '''
    img_lst = []
    for smp_nm in filename_lst:
        img_lst.append(load_image(mydir, smp_nm))
    return img_lst


#
#
# def sample_to_fname(sample_df, row_idx, suffix='tif'):
#     '''Given a dataframe of sampled images, get the
#     corresponding filename.'''
#     fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
#     return '{}.{}'.format(fname, suffix)


def load_labels(dir, filename):
    '''Load labels from the given path, return a data frame containing  '''
    # Import tag lists
    train_tags = pd.read_csv(os.path.join(rootpath, dir, filename))

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


def sample_from_label(df_label, n=None, suffix = '.tif'):
    ''' Randomly sample n from the label datafrom, return sampled image names (type: list)
        If n is None, then sample is the whole training set'''
    if n is not None :
        smp = df_label.sample(n)
    else:
        smp = df_label
    smp_img_name = [x + suffix for x in smp['image_name']]
    return smp, smp_img_name


def precision_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float')
    return (cm[0][0]/np.sum(cm[0])), (cm[0][0]/(cm[0][0] + cm[1][0]))


def f_score(y_true, y_pred, beta=2.):
    p, r = precision_recall(y_true, y_pred)
    return (1 + beta**2) * p*r / (beta**2 * p + r)


