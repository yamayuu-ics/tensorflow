
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model

IMAGE_W = 256
IMAGE_H = 256

SEED = 20160923

test = Image.open('./palette.png')
palette = np.array(test.getpalette(), dtype=np.uint8).reshape(-1, 3) # パレットの取得

print("tf ver : %s"%tf.__version__)


#lines_dataset = tf.data.TextLineDataset('/mnt/hdd1/ml_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
#data_set = lines_dataset.map(read_images)
#data_set = data_set.map(da_image)
#data_set = data_set.batch(4)

#train_summary_writer = tf.summary.create_file_writer('./log/train')


class VOC_Loader():
    def __init__(self):
        self.train_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        self.valid_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        #print(self.AUTOTUNE)

        train_lines_dataset = tf.data.TextLineDataset(self.train_path)
        self.train_data = train_lines_dataset.map(self.read_images, num_parallel_calls=self.AUTOTUNE)
        self.train_data = self.train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.train_data = self.train_data.map(self.da_image, num_parallel_calls=self.AUTOTUNE)
        self.train_data = self.train_data.shuffle(1464)
        self.train_data = self.train_data.batch(10)
        

        valid_lines_dataset = tf.data.TextLineDataset(self.valid_path)        
        self.valid_data = valid_lines_dataset.map(self.read_images, num_parallel_calls=self.AUTOTUNE)
        self.valid_data = self.valid_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.valid_data = self.valid_data.map(self.pre_valid, num_parallel_calls=self.AUTOTUNE)
        self.valid_data = self.valid_data.batch(2)
        
    
    def read_images(self,path):
        img_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/JPEGImages/' + path + '.jpg'
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image / 255
    
        label_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/SegmentationClass_pre/' + path + '.png'
        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label,1)

        return image,label

    def da_image(self,image,label):
        image = tf.image.resize(image,(IMAGE_H,IMAGE_W))
        label = tf.image.resize(label,(IMAGE_H,IMAGE_W),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        r = tf.random.uniform([1],maxval=1.0)
        if r > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
    
        return image,label
    
    def pre_valid(self,image,label):
        image = tf.image.resize(image,(IMAGE_H,IMAGE_W))
        label = tf.image.resize(label,(IMAGE_H,IMAGE_W),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
        return image,label


if __name__=='__main__':
    import time
    Loader = VOC_Loader()

    start = time.time()


    tbar = tqdm(Loader.train_data)
    for image,label in tbar:
        a = 1+1
        #print(image.shape)
        #print(label.shape)

    for image,label in Loader.valid_data:
        pass
        #print(image.shape)
        #print(label.shape)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    sys.exit()


