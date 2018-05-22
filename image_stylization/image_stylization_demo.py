"""
A small Jupyter demo of the fast image stylization. To use, install jupyter and from this
directory run 'jupyter notebook'
"""
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt


import ast
import numpy as np
import random
import tensorflow as tf
import urllib2
import sys
import os

# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/ubuntu/eclipse-workspace/Github/magenta/magenta/models/image_stylization"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)

from image_stylization import image_utils
from image_stylization import model

def DownloadCheckpointFiles(checkpoint_dir='checkpoints'):
    """Download checkpoint files if necessary."""
    url_prefix = 'http://download.magenta.tensorflow.org/models/' 
    checkpoints = ['multistyle-pastiche-generator-monet.ckpt', 'multistyle-pastiche-generator-varied.ckpt']
    for checkpoint in checkpoints:
        full_checkpoint = os.path.join(checkpoint_dir, checkpoint)
        if not os.path.exists(full_checkpoint):
            print 'Downloading', full_checkpoint
            response = urllib2.urlopen(url_prefix + checkpoint)
            data = response.read()
            with open(full_checkpoint, 'wb') as fh:
                fh.write(data)

# Select an image (any jpg or png).
input_image = 'evaluation_images/guerrillero_heroico.jpg'

# Select a demo ('varied' or 'monet')
demo = 'mine'

# DownloadCheckpointFiles()
image = np.expand_dims(image_utils.load_np_image(
          os.path.expanduser(input_image)), 0)
if demo == 'monet':
    checkpoint = 'checkpoints/multistyle-pastiche-generator-monet.ckpt'
    num_styles = 10  # Number of images in checkpoint file. Do not change.
elif demo == 'varied':
    checkpoint = 'checkpoints/multistyle-pastiche-generator-varied.ckpt'
    num_styles = 32  # Number of images in checkpoint file. Do not change.
else:
    checkpoint = 'tmp/image_stylization/run1/train/model.ckpt-49458'
    num_styles = 7  # Number of images in checkpoint file. Do not change.
    
    
# Styles from checkpoint file to render. They are done in batch, so the more 
# rendered, the longer it will take and the more memory will be used.
# These can be modified as you like. Here we randomly select six styles.
styles = range(num_styles)
# random.shuffle(styles)
which_styles = styles[0:6]
num_rendered = len(which_styles)  
print(styles,which_styles)

with tf.Graph().as_default(), tf.Session() as sess:
    stylized_images = model.transform(
        tf.concat([image for _ in range(len(which_styles))], 0),
        normalizer_params={
            'labels': tf.constant(which_styles),
            'num_categories': num_styles,
            'center': True,
            'scale': True})
    model_saver = tf.train.Saver(tf.global_variables())
    model_saver.restore(sess, checkpoint)
    stylized_images = stylized_images.eval()
    
    # Plot the images.
    counter = 0
    num_cols = 3
    f, axarr = plt.subplots(num_rendered // num_cols, num_cols, figsize=(25, 25))
    for col in range(num_cols):
        for row in range( num_rendered // num_cols):
            axarr[row, col].imshow(stylized_images[counter])
            axarr[row, col].set_xlabel('Style %i' % which_styles[counter])
            counter += 1
            
    plt.show()
        