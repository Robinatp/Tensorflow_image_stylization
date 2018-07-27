import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave, imresize
import utils
import os
import cv2

import matplotlib.pyplot as plt 
from encoder import Encoder
from decoder import Decoder
from adain_norm import AdaIN
slim = tf.contrib.slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import vgg
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


VGG_PATH = 'models/vgg19_normalised.npz'
DECODER_PATH = 'models/decoder.npy'


content_path = 'images/content/'
style_path ='images/style/'
output_path = './output/'



# print all op names
def _print_tensor_name(chkpt_fname):
    reader = pywrap_tensorflow.NewCheckpointReader(chkpt_fname)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("all tensor name:")
    for key in var_to_shape_map:
        print("tensor_name: ", key )
#         if("conv1_1" in key):
#             print(reader.get_tensor(key)) # Remove this is you want to print only variable names

def _load_ckpts(weights_file):
    _print_tensor_name(weights_file)
    saver = tf.train.Saver() #initial
    saver.restore(sess, weights_file) # saver.restore/saver.save


def init_vgg19_decoder(weights_file):
    reader = pywrap_tensorflow.NewCheckpointReader(weights_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    
#     for key in var_to_shape_map:
#         print("tensor_name: ", key)
#         #print(reader.get_tensor(key))
        
    variable_retore={}
    encoder_vars = [var for var in slim.get_variables('encoder')
                            if 'encoder' in var.name]

    
    vgg19_encoder_vars = [var for var in slim.get_variables('vgg_19')
                            if 'vgg_19' in var.name]

    
    for k ,v in zip(vgg19_encoder_vars,encoder_vars):
        variable_retore[k.op.name]=v.op.name
        
        
    with tf.variable_scope('', reuse = True):
        for k,v in variable_retore.items():
#             if("conv4_1" in v):
#                 print(v,reader.get_tensor(v)) # Remove this is you want to print only variable names
            sess.run(tf.get_variable(k).assign(reader.get_tensor(v)))
#             if("conv4_1" in k):
#                 print(k,sess.run(tf.get_variable(k)))

def save_graph_to_file(sess, graph_def, graph_file_name):
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph_def, ["output"])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return     



if __name__ == '__main__':
    CONVERT_WEIGHT=False

    content_images = os.listdir(content_path)
    style_images = os.listdir(style_path)


    with tf.Graph().as_default(), tf.Session() as sess:

        if CONVERT_WEIGHT:
            #init class
            encoder = Encoder(VGG_PATH)# convert the tf.nn.conv2d to slim format of vgg
        decoder = Decoder(mode='test', weights_path=DECODER_PATH)

        content_input = tf.placeholder(tf.float32, shape=(1,None,None,3), name='content_input')
        style_input =   tf.placeholder(tf.float32, shape=(1,None,None,3), name='style_input')

        # switch RGB to BGR
        content = tf.reverse(content_input, axis=[-1])
        style   = tf.reverse(style_input, axis=[-1])

        # preprocess image
        content = vgg.preprocess(content)
        style   = vgg.preprocess(style)


        encoder_content, encoder_content_points= vgg.vgg_19(content,reuse=False, final_endpoint="conv4_1")
        encoder_style, encoder_style_points= vgg.vgg_19(style,reuse=True, final_endpoint="conv4_1")


        # pass the encoded images to AdaIN
        target_features = AdaIN(encoder_content, encoder_style)

        # decode target features back to image
        with tf.variable_scope("decoder_target"):
            #alpha = 0.8
            #target_features=(1-alpha)*encoder_content+alpha*target_features #content-style trade-off
            generated_img = decoder.decode(target_features)

            # deprocess image
            generated_img = vgg.deprocess(generated_img)
    
            # switch BGR back to RGB
            generated_img = tf.reverse(generated_img, axis=[-1])
    
            # clip to 0..255
            generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
        generated_img = tf.to_int32(generated_img,name ="output")

        sess.run(tf.global_variables_initializer())
        
        writer =tf.summary.FileWriter("logs/",graph = sess.graph)
        writer.close()
        
        
#         saver = tf.train.Saver()
#         saver.save(sess,"weight/AdaIN.ckpt")
#         _load_ckpts("weight/AdaIN.ckpt")
#         _print_tensor_name("weight/AdaIN.ckpt")
         
        if CONVERT_WEIGHT:
            print("encoder")
            for v in slim.get_variables("encoder"):
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))
             
            print("decoder")
            for v in slim.get_variables("decoder"):
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))
                
            print("vgg_19")
            for v in slim.get_variables("vgg_19"):
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))

        # Function to restore encoder parameters.
        def init_fn_encoder(session):
            saver_encoder = tf.train.Saver(slim.get_variables('encoder'))
            print("encoder:", slim.get_variables('encoder'))
            saver_encoder.restore(session, "weight/AdaIN.ckpt")
            
        # Function to restore decoder parameters.
        def init_fn_decoder(session):
            saver_decoder = tf.train.Saver(slim.get_variables('decoder'))
            print("decoder", slim.get_variables('decoder'))
            saver_decoder.restore(session, "weight/AdaIN.ckpt")
            
        def init_fn(session):
            init_fn_encoder(session)
            init_fn_decoder(session)

        if CONVERT_WEIGHT:
            init_fn(sess)#restore encoder and decoder separately
            init_vgg19_decoder("weight/AdaIN.ckpt")# convert the tf.nn.conv2d to slim format of vgg
           
            print("parameter")
            for v in slim.get_variables():
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))
                
            varaibles_to_save = [var for var in slim.get_variables()
                            if 'encoder' not in var.name]
            print(varaibles_to_save)
            saver = tf.train.Saver(varaibles_to_save)
            saver.save(sess,"weight/AdaIN-slim.ckpt")
        else:
            print("parameter")
            for v in slim.get_variables():
                print('name = {}, shape = {}'.format(v.name, v.get_shape()))
                
            _load_ckpts("weight/AdaIN-slim.ckpt")
            save_graph_to_file(sess, sess.graph_def, "weight/AdaIN.pb")
        
        
        
        
        for s in style_images:     
            style_image   = imread(os.path.join(style_path,s), mode='RGB')
            style_tensor = np.expand_dims(style_image, axis=0)
 
            f, axarr = plt.subplots(2, 3, figsize=(25, 25))
            axarr[0,0].imshow(style_image)
            axarr[0,0].set_xlabel(s)
            counter = 1 
            for c in content_images:
                # Load image from path and add one extra diamension to it.
                content_image = imread(os.path.join(content_path,c), mode='RGB')
                content_tensor = np.expand_dims(content_image, axis=0)
                

                result = sess.run(generated_img, 
                                  feed_dict={content_input: content_tensor,
                                             style_input: style_tensor})
                
                result_name = os.path.join(output_path,s.split('.')[0]+'_'+c.split('.')[0]+'.jpg')
                print(result_name,' is generated')
                
                axarr[counter/3,counter%3].imshow(result[0])
                axarr[counter/3,counter%3].set_xlabel(result_name)
                counter += 1
#             plt.savefig("output/"+s.split('.')[0]+".jpg", bbox_inches='tight')
            plt.show() 



