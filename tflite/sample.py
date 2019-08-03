# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Reshape, Dense, Add, Conv2D, ZeroPadding2D

# Model parameters
MODEL_WIDTH = 256
MODEL_HEIGHT = 256
MODEL_CHANNEL = 3

# Create model structure
input0 = Input(shape=(MODEL_WIDTH,MODEL_HEIGHT,MODEL_CHANNEL))
conv0 = Conv2D(
      filters=1,
      kernel_size=(1,1),
      strides=(MODEL_WIDTH,MODEL_HEIGHT),
      padding='valid',
      activation='relu'
)(input0)
model = Model(inputs=[input0], outputs=[conv0])

# Save model
model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_name = 'model_stride/model_stride_' + str(MODEL_WIDTH) +'x' + str(MODEL_HEIGHT) + 'x' + str(MODEL_CHANNEL)
model.save(model_name + '.h5')

# Convert to quantized tflite
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5')
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.default_ranges_stats = (0, 6)
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (128., 127.)}  # mean, std_dev
tflite_model = converter.convert()
open(model_name + '.tflite', "wb").write(tflite_model)