import ConfigParser
from train_u_net import

#=============== read input from config =====================
config = ConfigParser.RawConfigParser()
config.read('config_file_path')

#Paths
path_dir = config.get('data paths', 'path_dir')
train_sample = config.get('data paths', 'train_sample')
train_tag = config.get('data paths', 'train_tag')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#============ Load the data and divided in patches ========
patches_train_img_sample, patches_train_tag_sample = get_data_training_sample(
                       path_dir, train_sample, train_tag)

#=========== Construct and save the model arcitecture =====
n_ch = patches_train_img_sample.shape[3]
patch_height = patches_train_img_sample.shape[1]
patch_width = patches_train_img_sample.shape[2]
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model
print "Check: final output of the network:"
print model.output_shape
#TODO: add check pointer

model.fit(patches_train_img_sample, patches_train_tag_sample, batch_size=32, epochs=N_epochs, verbose=1, shuffle=True,
              validation_split=0.2)
score = model.evaluate(patches_train_img_sample, patches_train_tag_sample, batch_size=16)
#TODO: save model
