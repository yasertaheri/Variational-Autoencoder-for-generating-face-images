# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 03:07:43 2017

@author: y_moh
"""

import matplotlib.pyplot as plt
import tensorflow as tf

def load_data():

    filename_queue = tf.train.string_input_producer( tf.train.match_filenames_once("./img_align_celeba/*.jpg"), shuffle = False, num_epochs=None)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image_query = tf.image.decode_image(image_file, channels=3) 

    image =tf.image.resize_image_with_crop_or_pad (image_query, 160, 160)
    image =tf.image.resize_images (image, [80, 80])

    image = tf.image.rgb_to_grayscale(image)

    image = tf.cast(image,tf.uint8)


    image.set_shape((80, 80, 1))

    images = tf.train.batch([image], batch_size=10, capacity=30, num_threads=1,allow_smaller_final_batch=True)

    images = tf.reshape(images,[-1,80,80])

    return images 


# Start a new session to show example output.
