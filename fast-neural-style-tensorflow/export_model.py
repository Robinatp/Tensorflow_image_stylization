# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import utils


tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models/mosaic/fast-style-model.ckpt-40000", "an image transformation network")
tf.app.flags.DEFINE_string("image_file", "img/test.jpg", "the test mage")

FLAGS = tf.app.flags.FLAGS

def save_graph_to_file(sess, graph, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph, ["output_image"])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return

def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)
    # Get image's height and width.
    height = 0
    width = 0
    with tf.gfile.GFile(FLAGS.image_file, 'rb') as f:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(f.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(f.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
            print(image)
            plt.subplot(121)
            np_image = sess.run(image)
            plt.imshow(np_image)
            
            input_shape = (None, None, 3)
            input_tensor = tf.placeholder(dtype=tf.uint8, shape=input_shape, name='image_tensor')
            print(input_tensor)
            with tf.variable_scope("input_process"):   
                processed_image = utils.mean_image_subtraction(
                    input_tensor, [123.68, 116.779, 103.939])                    # Preprocessing image
                batched_image = tf.expand_dims(processed_image, 0)               # Add batch dimension

            generated = model.net(batched_image, training=False)
            
            generated = tf.cast(generated, tf.uint8)
            # Remove batch dimension
            generated = tf.squeeze(generated, [0],name='output_image')
            

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)
            
            summary_writer = tf.summary.FileWriter("logs",sess.graph)
            save_graph_to_file(sess,sess.graph_def ,"models/new_freeze_graph.pb") 
            
             
            # Make sure 'generated' directory exists.
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with tf.gfile.GFile(generated_file, 'wb') as f:
                feed_dict={input_tensor:np_image}
                plt.subplot(122)
                plt.imshow(sess.run(generated,feed_dict))
                plt.show()
                start_time = time.time()
                f.write(sess.run(tf.image.encode_jpeg(generated),feed_dict))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
