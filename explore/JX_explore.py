from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt

###############################
########import images##########

dir_name = '../train-jpg-sample/'

# all file paths has 101 elements which includes 100 paths, the first one is not an image file
# images size: [100, 3 * [256,256]]
all_file_paths = [f for f in listdir(dir_name) if isfile(join(dir_name,f))]
images = np.empty(len(all_file_paths) - 1, dtype=object)
for n in range(1, len(all_file_paths)):
    images[n - 1] = cv2.imread(join(dir_name,all_file_paths[n]))

# print(len(all_file_paths))
# print(len(images))
# print(type(images))
# print(len(images[0]))
# for i in range(0, len(all_file_paths)):
#     print(i)
#     print(all_file_paths[i])


img = cv2.imread('../train_10.jpg')

##########################################
#########detector, descriptor#############

orb = cv2.ORB()

kp, des1 = orb.detectAndCompute(img, None)

img2 = cv2.drawKeypoints(img, kp, color = (0, 255, 0 ), flags = 0)

plt.imshow(img2), plt.show()
print(des1)
print(len(des1[0]))

###########################################
##########contrast#########################

lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)

#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
plt.imshow(final),plt.show()

############################################
########linear transformation###############

imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

beta = -0.8
alpha = 3

imghsv[:,:,2] = [[pixel * beta + alpha for pixel in row] for row in imghsv[:,:,2]]
cv2.imshow('contrast', cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR))
cv2.waitKey(1000)
