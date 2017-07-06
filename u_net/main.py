import ConfigParser
from train_u_net import get_unet
from extract_files import get_data_training_sample
from extract_files import get_data_training
from extract_files import get_data_random_training
from utils import f1_score
from keras.utils import to_categorical

#=============== read input from config =====================
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

#Paths
path_dir = config.get('data paths', 'path_dir')
train_sample = config.get('data paths', 'train_sample')
train_full = config.get('data paths', 'train_all_jpg')
image_type = 'jpg'
train_tag = config.get('data paths', 'train_tag')

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))



#============ Load the data and divided in patches ========

#patches_train_img, patches_train_tag = get_data_random_training(
#                       path_dir, train_full, train_tag, image_type)

#print(patches_train_img.shape)
#print(patches_train_tag.shape)

#=========== Construct and save the model arcitecture =====
n_ch = patches_train_img.shape[3]
patch_height = patches_train_img.shape[1]
patch_width = patches_train_img.shape[2]

model = get_unet(n_ch, patch_height, patch_width)  #the U-net model

print "Check: final output of the network:"
print model.output_shape

#=========== Train in batches ============================
rootpath = '/home/ubuntu/data'

start_idx = 0
batch_size = 8000
total_size = len(train_tags)
rest = total_size - batch_size

while(rest > 0):
    x, y = load_batch_rgb(train_tags,path_dir,start_idx,batch_size,'.jpg')
    patches_train_img = np.asarray(x)
    patches_train_tag = to_categorical(y)
    model.fit(patches_train_img, patches_train_tag, batch_size=32, epochs=N_epochs, verbose=1, shuffle=True,
              validation_split=0.1)
    rest -= batch_size
    start_idx += batch_size

score = model.evaluate(patches_train_img, patches_train_tag, batch_size=32)

print score

model.save('./output/model/' + '_batch_size_' + batch_size + '_epochs_' + N_epochs)

from keras.utils import plot_model
plot_model(model, to_file='./output/result_plot/momodel.png')
