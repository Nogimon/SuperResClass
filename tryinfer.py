from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time
import numpy as np
from glob import glob
import scipy.misc as sic
#import imageio
import collections

def infer(inputdir, outputdir):

    Flags = tf.app.flags

    # The system parameter
    Flags.DEFINE_string('output_dir', './result/', 'The output directory of the checkpoint')
    Flags.DEFINE_string('summary_dir', './result/log/', 'The dirctory to output the summary')
    Flags.DEFINE_string('mode', 'inference', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', './SRGAN_pre-trained/model-200000', 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_model', True, 'If set True, the weight will be loaded but the global_step will still '
                                                     'be 0. If set False, you are going to continue the training. That is, '
                                                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'The type of pretrained model (SRGAN or SRResnet)')
    Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
    Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
    Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')
    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
    Flags.DEFINE_string('input_dir_LR', './infer/sample/', 'The directory of the input resolution input data')
    Flags.DEFINE_string('input_dir_HR', './data/infer_HR', 'The directory of the high resolution input data')
    Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
    Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
    Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
    Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                      'enough random shuffle.')
    Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                       'enough random shuffle')
    Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
    # Generator configuration
    Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
    # The content loss parameter
    Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
    Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
    Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
    Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
    Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
    Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')


    FLAGS = Flags.FLAGS
    FLAGS.input_dir_LR = inputdir
    FLAGS.output_dir = outputdir

    # Print the configuration of the model
    print_configuration_op(FLAGS)

    # Check the output_dir is given
    if FLAGS.output_dir is None:
        raise ValueError('The output directory is needed')

    # Check the output directory to save the checkpoint
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    # Check the summary directory to save the event
    if not os.path.exists(FLAGS.summary_dir):
        os.mkdir(FLAGS.summary_dir)


    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None

    # Declare the test data reader
    inference_data = inference_data_loader(FLAGS)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        max_iter = len(inference_data.inputs)
        print('Evaluation starts!!')
        for i in range(max_iter):
            input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
            path_lr = inference_data.paths_LR[i]
            results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
            filesets = save_images(results, FLAGS)
            for i, f in enumerate(filesets):
                print('evaluate image', f['name'])

    delflags(FLAGS)


def delflags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def inference_data_loader2(FLAGS, input_dir_LR):
    # Get the image name list
    if (input_dir_LR == 'None'):
        raise ValueError('Input directory is not provided')

    if not os.path.exists(input_dir_LR):
        raise ValueError('Input directory not found')

    image_list_LR_temp = os.listdir(input_dir_LR)
    image_list_LR = [os.path.join(input_dir_LR, _) for _ in image_list_LR_temp if _.split('.')[-1] == 'png']

    # Read in and preprocess the images
    def preprocess_test(name):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        # check grayscale image
        if im.shape[-1] != 3:
            h, w = im.shape
            temp = np.empty((h, w, 3), dtype=np.uint8)
            temp[:, :, :] = im[:, :, np.newaxis]
            im = temp.copy()
        im = im / np.max(im)

        return im

    image_LR = [preprocess_test(_) for _ in image_list_LR]

    # Push path and image into a list
    Data = collections.namedtuple('Data', 'paths_LR, inputs')

    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )


def save_images2(output_dir, fetches, FLAGS, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    in_path = fetches['path_LR']
    name, _ = os.path.splitext(os.path.basename(str(in_path)))
    fileset = {"name": name, "step": step}

    if FLAGS.mode == 'inference':
        kind = "outputs"
        filename = name + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][0]
        with open(out_path, "wb") as f:
            f.write(contents)
        filesets.append(fileset)
    else:
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][0]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def infer2():#inputdir, outputdir):
    
    Flags = tf.app.flags

    # The system parameter
    Flags.DEFINE_string('output_dir', './result/', 'The output directory of the checkpoint')
    Flags.DEFINE_string('summary_dir', './result/log/', 'The dirctory to output the summary')
    Flags.DEFINE_string('mode', 'inference', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', './SRGAN_pre-trained/model-200000', 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_model', True, 'If set True, the weight will be loaded but the global_step will still '
                                                     'be 0. If set False, you are going to continue the training. That is, '
                                                     'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'The type of pretrained model (SRGAN or SRResnet)')
    Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
    Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
    Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')
    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
    Flags.DEFINE_string('input_dir_LR', './infer/sample/', 'The directory of the input resolution input data')
    Flags.DEFINE_string('input_dir_HR', './data/infer_HR', 'The directory of the high resolution input data')
    Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
    Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
    Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
    Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                      'enough random shuffle.')
    Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                       'enough random shuffle')
    Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
    # Generator configuration
    Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
    # The content loss parameter
    Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
    Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
    Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
    Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
    Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
    Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')

    
    FLAGS = Flags.FLAGS
    #FLAGS.input_dir_LR = inputdir
    #FLAGS.output_dir = outputdir

    # Print the configuration of the model
    print_configuration_op(FLAGS)

    # Check the output_dir is given
    if FLAGS.output_dir is None:
        raise ValueError('The output directory is needed')

    # Check the output directory to save the checkpoint
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    # Check the summary directory to save the event
    if not os.path.exists(FLAGS.summary_dir):
        os.mkdir(FLAGS.summary_dir)


    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    # In the testing time, no flip and crop is needed
    if FLAGS.flip == True:
        FLAGS.flip = False

    if FLAGS.crop_size is not None:
        FLAGS.crop_size = None

    # Declare the test data reader
    #inference_data = inference_data_loader2(FLAGS, inputdir)

    inputs_raw = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='inputs_raw')
    path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

    with tf.variable_scope('generator'):
        if FLAGS.task == 'SRGAN' or FLAGS.task == 'SRResnet':
            gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    with tf.name_scope('convert_image'):
        # Deprocess the images outputed from the model
        inputs = deprocessLR(inputs_raw)
        outputs = deprocess(gen_output)

        # Convert back to uint8
        converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
        converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

    with tf.name_scope('encode_image'):
        save_fetch = {
            "path_LR": path_LR,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
        }

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)




        dire0 = "./lowresdata/train/"
        folder = sorted(glob(dire0 + "*/"))
        folders = folder[:]

        num = 7
        for inputf in folders:
            outputdir = inputf[0:num] + '2' + inputf[num:]
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            inference_data = inference_data_loader2(FLAGS, inputf)


            max_iter = len(inference_data.inputs)
            print('Evaluation starts for ', inputf)
            for i in range(max_iter):
                input_im = np.array([inference_data.inputs[i]]).astype(np.float32)
                path_lr = inference_data.paths_LR[i]
                results = sess.run(save_fetch, feed_dict={inputs_raw: input_im, path_LR: path_lr})
                filesets = save_images2(outputdir ,results, FLAGS)
                for i, f in enumerate(filesets):
                    print('evaluate image', f['name'])

    delflags(FLAGS)


if __name__ == "__main__":
    '''
    dire0 = "./lowresdata/train/"
    folder = sorted(glob(dire0 + "*/"))
    folders = folder[:]

    num = 7
    for inputf in folders:
        outputf = inputf[0:num] + '2' + inputf[num:]
        if not os.path.exists(outputf):
            os.makedirs(outputf)
        infer(inputf, outputf)
    '''
    infer2()


