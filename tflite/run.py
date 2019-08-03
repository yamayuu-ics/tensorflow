import time
from PIL import Image, ImageDraw, ImageFont
import numpy
from edgetpu.basic.basic_engine import BasicEngine

MODEL_NAME = "model_stride/model_stride/model_stride_256x256x3_edgetpu.tflite"


### Load model and prepare TPU engine
engine = BasicEngine(MODEL_NAME)
width = engine.get_input_tensor_shape()[1]
height = engine.get_input_tensor_shape()[2]

### prepara input tensor
img = Image.new('RGB', (width, height), (128, 128, 128))
draw = ImageDraw.Draw(img)
input_tensor = numpy.asarray(img).flatten()

### Run inference
start = time.time()
num_measurement = 10000
for i in range(num_measurement):
    _, raw_result = engine.RunInference(input_tensor)
    # time.sleep(2)
elapsed_time = time.time() - start
print ("elapsed_time: {0} ".format(1000 * elapsed_time / num_measurement) + "[msec]")
