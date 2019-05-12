# Super Resolution Classification

This is a project aimed at using classification on super resolution image.

## Super Resolution GAN Model

This model aims to generate super resolution images from low resolution images.

### Weights and images

Pretrained weights and images are actually needed for the model. However, they are too large to upload to github. (to update: upload the weights and images to google drive and provide download link here) 

### Run inference on images
Environment:
    python 3.6
    tensorflow 1.10
Run:
```
python infer2.py
```
will generate the high resolution images.

```
python trytest.py
```
will test how the gan performs.


## Classfication Model

This is a deep nerual network which combines pretrained VGG19 and ResNet50. The model aims to classify dog breeds from dog images.

### Input images

In order to see the results of classification accuracy due to the resolution of images, run the model first with raw image input as the low resolution images. Next on, run the model with the input of Super Resolution GAN model. Please be award that the images should be stored in labeled files respectively, and the model requires train, test and valid set.

### Run Model
Environment:
    python 3.6.1
    cuda 8.0
    cudnn 6.0
    tensorflow r1.2
    (load h5py library if needed)
Run:
```
  python main.py
```
will run the model and returns the accuracy and hisory
