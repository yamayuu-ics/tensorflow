
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image

import tensorflow as tf

IMAGE_W = 256
IMAGE_H = 256

SEED = 20160923

test = Image.open('/mnt/hdd1/ml_data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png')
palette = np.array(test.getpalette(), dtype=np.uint8).reshape(-1, 3) # パレットの取得

def convert_to_rgb(index_colored_numpy, palette, n_colors=None):
    assert index_colored_numpy.dtype == np.uint8 and palette.dtype == np.uint8
    assert index_colored_numpy.ndim == 2 and palette.ndim == 2
    assert palette.shape[1] == 3
    if n_colors is None:
        n_colors = palette.shape[0]
    reduced = index_colored_numpy.copy()
    reduced[index_colored_numpy > n_colors] = 0 # 不要なクラスを0とする
    expanded_img = np.eye(n_colors, dtype=np.int32)[reduced]  # [H, W, n_colors] int32
    use_pallete = palette[:n_colors].astype(np.int32)  # [n_colors, 3] int32
    return np.dot(expanded_img, use_pallete).astype(np.uint8)


def dbg_convert_tensr_to_idx_color_img(tensor,palette):
    numpy_img = tensor.numpy()
    num_img = numpy_img.shape[0]
    out_img = []
    for i in range(num_img):
        color_img = convert_to_rgb(np.reshape(numpy_img[i],(IMAGE_H,IMAGE_W)),palette)
        out_img.append(color_img)
    
    out_img = np.array(out_img)
    return out_img




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

    for image,label in Loader.train_data:
        pass
        #print(image.shape)
        #print(label.shape)

    for image,label in Loader.valid_data:
        pass
        #print(image.shape)
        #print(label.shape)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    sys.exit()


