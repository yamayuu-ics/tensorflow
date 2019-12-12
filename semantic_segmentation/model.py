
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model


class Unet(Model):
    
    def __init__(self):
        super().__init__()

        self.conv1_1 = layers.Conv2D(64, 3,padding='same', activation='relu')
        self.conv1_2 = layers.Conv2D(64, 3,padding='same', activation='relu')
        self.max_pool1 = layers.MaxPooling2D()

        self.conv2_1 = layers.Conv2D(128, 3,padding='same', activation='relu')
        self.conv2_2 = layers.Conv2D(128, 3,padding='same', activation='relu')
        self.max_pool2 = layers.MaxPooling2D()

        self.conv3_1 = layers.Conv2D(256, 3,padding='same', activation='relu')
        self.conv3_2 = layers.Conv2D(256, 3,padding='same', activation='relu')
        self.max_pool3 = layers.MaxPooling2D()

        self.conv4_1 = layers.Conv2D(512, 3,padding='same', activation='relu')
        self.conv4_2 = layers.Conv2D(512, 3,padding='same', activation='relu')
        self.max_pool4 = layers.MaxPooling2D()

        self.conv5_1 = layers.Conv2D(1024, 3,padding='same', activation='relu')
        self.conv5_2 = layers.Conv2D(512, 3,padding='same', activation='relu')
        self.deconv5 = layers.Conv2DTranspose(512,kernel_size=[2, 2],padding='same',strides=[2, 2],activation='relu')
        
        self.conv_up_4_1 = layers.Conv2D(512, 3,padding='same', activation='relu')
        self.conv_up_4_2 = layers.Conv2D(512, 3,padding='same', activation='relu')
        self.deconv4 = layers.Conv2DTranspose(256,kernel_size=[2, 2],padding='same',strides=[2, 2],activation='relu')
        
        self.conv_up_3_1 = layers.Conv2D(256, 3,padding='same', activation='relu')
        self.conv_up_3_2 = layers.Conv2D(256, 3,padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(128,kernel_size=[2, 2],padding='same',strides=[2, 2],activation='relu')

        self.conv_up_2_1 = layers.Conv2D(128, 3,padding='same', activation='relu')
        self.conv_up_2_2 = layers.Conv2D(128, 3,padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(128,kernel_size=[2, 2],padding='same',strides=[2, 2],activation='relu')

        self.conv_up_1_1 = layers.Conv2D(64, 3,padding='same', activation='relu')
        self.conv_up_1_2 = layers.Conv2D(64, 3,padding='same', activation='relu')

        self.out_conv = layers.Conv2D(20, [1,2],padding='same', activation=None)
        #self.deconv3 = layers.Conv2DTranspose(128,kernel_size=[2, 2],padding='same',strides=[2, 2],activation='relu')
        self.sofmax = layers.Softmax() # batch is chanel last format
    @tf.function
    def call(self,input):
        with tf.name_scope("U-net"):
            with tf.name_scope("encoder"):
                x = self.conv1_1(input)
                result1 = self.conv1_2(x)
                x = self.max_pool1(result1)

                x = self.conv2_1(x)
                result2 = self.conv2_2(x)
                x = self.max_pool2(result2)

                x = self.conv3_1(x)
                result3 = self.conv3_2(x)
                x = self.max_pool3(result3)

                x = self.conv4_1(x)
                result4 = self.conv4_2(x)
                x = self.max_pool4(result4)

                x = self.conv5_1(x)
                x = self.conv5_2(x)

            with tf.name_scope("decoder"):
                x = self.deconv5(x)
                x = tf.concat([x,result4],axis=3)

                x = self.conv_up_4_1(x)
                x = self.conv_up_4_2(x)
                x = self.deconv4(x)

                x = tf.concat([x,result3],axis=3)
                x = self.conv_up_3_1(x)
                x = self.conv_up_3_2(x)
                x = self.deconv3(x)

                x = tf.concat([x,result2],axis=3)
                x = self.conv_up_2_1(x)
                x = self.conv_up_2_2(x)
                x = self.deconv2(x)

                x = tf.concat([x,result1],axis=3)
                x = self.conv_up_1_1(x)
                x = self.conv_up_1_2(x)
            
                x = self.out_conv(x)
            
            out = self.sofmax(x)
            return out
    

        

if __name__=='__main__':
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(stamp)
    logdir = "log/%s"%stamp
    writer = tf.summary.create_file_writer(logdir)

    input_tensor = np.random.rand(1, 256,256,3)
    print(input_tensor.shape)
    input_tensor = tf.convert_to_tensor(input_tensor,dtype=tf.float32)
    print(input_tensor.shape)
    with tf.device('/CPU:0'):
        model = Unet()
    
    tf.summary.trace_on(graph=True)
    
    out = model(input_tensor)

    with writer.as_default():
        tf.summary.trace_export(name="my_func_trace",step=0)

    
    print(out.shape)
    sum = tf.math.reduce_sum(out,axis=-1)
    print(sum.shape)
    print(sum)