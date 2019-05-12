from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image                  
#from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True 

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50


from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    #list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def extract_VGG19(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg19(tensors)
    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def extract_Resnet50(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def input_branch(input_shape=None):
    
    size = int(input_shape[2] / 4)
    
    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input

train_files, train_targets = load_dataset('highresdata/train')
valid_files, valid_targets = load_dataset('highresdata/valid')
test_files, test_targets = load_dataset('highresdata/test')
#train_files, train_targets = load_dataset('train')
#test_files, test_targets = load_dataset('test')

#dog_names = [item[20:-1] for item in sorted(glob("dog/assets/images/train/*/"))]
dog_names = [item[10:-1] for item in sorted(glob("highresdata/train/*/"))]
#dog_names = [item[10:-1] for item in sorted(glob("train/*/"))]
#print(dog_names)
#input()
print('There are %d total dog categories.' % len(dog_names))
#np.hstack stacks array in sequence horizontally
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
#input()
# pre-process the data for Keras
# rescale every pixel in every image by 255
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255



#Extracting feature
print("Extracting feature")

train_vgg19 = extract_VGG19(train_files)
valid_vgg19 = extract_VGG19(valid_files)
test_vgg19 = extract_VGG19(test_files)
print("VGG19 shape", train_vgg19.shape[1:])

train_resnet50 = extract_Resnet50(train_files)
valid_resnet50 = extract_Resnet50(valid_files)
test_resnet50 = extract_Resnet50(test_files)
print("Resnet50 shape", train_resnet50.shape[1:])



vgg19_branch, vgg19_input = input_branch(input_shape=(7, 7, 512))
resnet50_branch, resnet50_input = input_branch(input_shape=(1, 1, 2048))
concatenate_branches = Concatenate()([vgg19_branch, resnet50_branch])
net = Dropout(0.3)(concatenate_branches)
net = Dense(640, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.3)(net)
net = Dense(133, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[vgg19_input, resnet50_input], outputs=[net])
model.summary()


model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
#checkpointer = ModelCheckpoint(filepath='saved_models/bestmodel.hdf5', verbose=1, save_best_only=True)
history = model.fit([train_vgg19, train_resnet50], train_targets, validation_data=([valid_vgg19, valid_resnet50], valid_targets),epochs=30, batch_size=4, verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.savefig('Model_accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.savefig('Model_loss.png')

#model.load_weights('saved_models/bestmodel.hdf5')



predictions = model.predict([test_vgg19, test_resnet50])
breed_predictions = [np.argmax(prediction) for prediction in predictions]
breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))
