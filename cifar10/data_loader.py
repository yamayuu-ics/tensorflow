
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import numpy as np


_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3

NUM_CLASSES = 10

_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS

# The record is the image plus a one-byte label



_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1



class Dataloader(object):
	
	def __init__(self,batch_size,mode):
		if mode is 'train':
			self.filepath = [	'/data/ml_data/cifar-10-batches-bin/data_batch_1.bin',
								'/data/ml_data/cifar-10-batches-bin/data_batch_2.bin',
								'/data/ml_data/cifar-10-batches-bin/data_batch_3.bin',
								'/data/ml_data/cifar-10-batches-bin/data_batch_4.bin',
								'/data/ml_data/cifar-10-batches-bin/data_batch_5.bin']
		elif mode is 'test':
			self.filepath = ['/data/ml_data/cifar-10-batches-bin/test_batch.bin']
		else:
			return
		print(self.filepath)
		

		#self.dataset = tf.data.TextLineDataset(self.filepath)

		#self.dataset = self.dataset.map(self.path)

		self.dataset = tf.data.FixedLengthRecordDataset(self.filepath, _RECORD_BYTES)
		self.dataset = self.dataset.map(self.parse_record,num_parallel_calls=2)

		#self.dataset = self.dataset.map(self.read_pickle)

		self.batch = batch_size

		self.dataset = self.dataset.batch(self.batch)

		iter = self.dataset.make_initializable_iterator()
		self.init_op = iter.initializer
		self.data_batch,self.label_batch = iter.get_next()
		#self.label_batch = tf.one_hot(indices=tf.cast(self.label_batch, tf.int32), depth=10)
		#self.file = iter.get_next()



	def parse_record(self,raw_record):
		record_vector = tf.decode_raw(raw_record, tf.uint8)
		label = tf.cast(record_vector[0], tf.int32)
		depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],[_NUM_CHANNELS, _HEIGHT, _WIDTH])
		image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
		#image = preprocess_image(image, is_training)
		image = tf.cast(image, tf.float32)
		
		return image, label

def main(_):
	dataloader = Dataloader(100,'test')
	print(dataloader.dataset)
	input_tensor = tf.constant(np.random.rand(2,32, 32,3), dtype=tf.float32)
	print(input_tensor)

	init = tf.initializers.global_variables()

	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)

	sess.run(init)
	sess.run(dataloader.init_op)

	print(type(sess.run(input_tensor)))

	while True:
		try:
			img,label = sess.run([dataloader.data_batch,dataloader.label_batch])
			print(label)
			#print(img)
			#data,label = sess.run([dataloader.batch])
			#print(label)

		except tf.errors.OutOfRangeError:
			break

if __name__ == '__main__':
	tf.app.run()
