#========================================================================
# Title: DDWI_water_model.py
# Description: A model for 'water' classification
#               1. Read in image and label, randomly select 1000 sample
#               2. Calculate NDWI scores, create features
#               3. Model: random forest
#=========================================================================

import rt_util # RT's util functions
import rt_ndwi # RT's NDWI data processing functions
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# import matplotlib.pyplot as plt

## set seed
seed = 7
np.random.seed(seed)

### Read labels, image (randomly sample)
##  Import labels
train_tags = rt_util.load_labels(dir = 'train_v2.csv', filename='train_v2.csv')

## Randomly select 1000 sample from the label df
smp_train_tags, smp_train_filename = rt_util.sample_from_label(df_label=train_tags, n=1000)
# print('Type of smp_train_tags: ' + str(type(smp_train_tags)))
# print(smp_train_filename[:11])

## Load image from the sampled list (in the order of the given list)
smp_train_img = rt_util.load_image_lst(mydir='train_tif_v2', filename_lst=smp_train_filename)
# print('Type of output'+str(type(smp_train_img)))
# print('Length of output' + str(len(smp_train_img)))

## Data processing
smp_train_img_ndwi = []
for img in smp_train_img:
    smp_train_img_ndwi.append(rt_ndwi.ndwi(img))

# parameters
n_thred = 10
test_size = 0.2

# Create thresholds list
cutoff_shred = np.linspace(0, 1, n_thred)
# Create x dataset (number of pixels with ndwi value > thresholds)
x_set = rt_ndwi.all_n_greater_threshold(smp_train_img_ndwi, thred=cutoff_shred)
# Create y dataset
y_set = np.array(smp_train_tags['water'])
# Split training set and test set
x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=test_size, random_state=seed)

## Model 1: Random forest
clf = RandomForestClassifier(n_estimators=25)
clf.fit(x_train, y_train)
clf_pred = clf.predict(x_test)
score = rt_util.f_score(y_test, clf_pred)
print(score)

# TODO: install xgboost package (tutorial page not found)
# Installation tutorial: https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/13043
# Xgboost example for binary classification problem: http://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/