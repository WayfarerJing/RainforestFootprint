import ConfigParser
from keras.models import Model
from keras.layers import Input, MaxPooling2D, core, Convolution2D, merge, UpSampling2D, Dense
from keras import backend as K
from utils import f1_score

K.set_image_data_format('channels_last')

#Define the neural network
def get_unet(n_ch, patch_height, patch_width):
    inputs = Input((patch_height, patch_width, n_ch))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    conv11 = core.Reshape((1, patch_width * patch_height))(conv10)
    
    conv12 = Dense(2, activation='sigmoid', name='main_output')(conv11)
    conv12 = core.Reshape((2, ))(conv12)
    
    model = Model(input=inputs, output=conv12)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', f1_score])
    
    return model

def get_unet_2(n_ch, patch_height, patch_width):
    inputs = Input((patch_height, patch_width, n_ch))
    
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    
    conv1 = BatchNormalization(input_shape = ((patch_height, patch_height, n_ch)))(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = BatchNormalization()(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BatchNormalization()(pool2)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)

    conv4 = BatchNormalization()(pool3)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BatchNormalization()(pool4)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = BatchNormalization()(pool5)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv6)
 
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=3)
    conv7 = Dropout(0.25)(up7)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv7)
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv4], mode='concat', concat_axis=3)
    conv8 = BatchNormalization()(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv8)
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=3)
    conv9 = Dropout(0.25)(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv9)
    
    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=3)
    conv10 = BatchNormalization()(up10)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv10)
    conv10 = BatchNormalization()(up10)
    conv10 = Conv2DTranspose(64, 3, 3, activation='relu', border_mode='same')(conv10)
  
    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=3)
    conv11 = BatchNormalization()(up11)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv11)
    conv11 = BatchNormalization()(conv11)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv11)
    conv11 = BatchNormalization()(up11)
    conv11 = Conv2DTranspose(32, 3, 3, activation='relu', border_mode='same')(conv11)

    conv11 = Convolution2D(1, 1, 1, activation='relu', border_mode='same')(conv11)
    conv11 = Flatten()(conv11)
    predictions = Dense(2, activation='sigmoid')(conv11)
    model = Model(input=inputs, output=predictions)
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', f1_score])
    
    return model
