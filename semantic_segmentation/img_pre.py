import os
import sys
import numpy as np
from PIL import Image

path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt'

with open(path) as f:
    lines = f.readlines()
    for l in lines:
        #print(l.replace('\n', ''))
        l = l.replace('\n', '')
        img_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/SegmentationClass/'+l+'.png'

        image = Image.open(img_path)
        image = np.asarray(image)
        #print(image.shape)
        #print(np.unique(image))

        out = Image.fromarray(image)
        out_path = '/mnt/hdd1/ml_data/VOCdevkit/VOC2012/SegmentationClass_pre/'+l+'.png'
        out.save(out_path)

        image = Image.open(out_path)
        image = np.asarray(image)
        #print(image.shape)
        #print(np.unique(image))

        #sys.exit()
