
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import numpy as np


def create_val(scope,name,shape):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        val = tf.get_variable(name,shape=shape)
    return val


class Model(object):

    def __init__(self,batch_size): 
        self.dummy = 0
        self.batch = batch_size
    #def all(self,x,label):
        #self.logits = self.model(x)
        


    def fc(self,scope,x,num_in,num_output):
        sh_w = [num_in,num_output]
        weight = create_val(scope,'fc_w',sh_w)
        bias = create_val(scope,'fc_b',[num_output])
        fc_result = tf.add(tf.matmul(x, weight) , bias)

        #relu = tf.cond(do_relu,lambda: tf.nn.relu(fc_result),lambda: fc_result)
        relu = tf.nn.relu(fc_result)
        return relu
    def fc_out(self,scope,x,num_in,num_output):
        sh_w = [num_in,num_output]
        weight = create_val(scope,'fc_o_w',sh_w)
        bias = create_val(scope,'fc_o_b',[num_output])
        fc_result = tf.add(tf.matmul(x, weight) , bias)

        #relu = tf.cond(do_relu,lambda: tf.nn.relu(fc_result),lambda: fc_result)
        #relu = tf.nn.relu(fc_result)
        return fc_result

    def conv_relu(self,scope,x,kernel_size,in_ch,num_layer):
        #in_ch = tf.shape(x)[-1]
        sh = [kernel_size,kernel_size,in_ch,num_layer]
        filter = create_val(scope,'conv',sh)
        conv = tf.nn.conv2d(x,filter,strides=[1,1,1,1], padding='SAME')
        relu = tf.nn.relu(conv)

        return relu
    def model(self,x,label):
        
        conv0_0 = self.conv_relu('conv0_0',x,3,3,64)        
        conv0_1 = self.conv_relu('conv0_1',conv0_0,3,64,64)       
        max_pool0 = tf.nn.max_pool(conv0_1,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')        
        
        conv1_0 = self.conv_relu('conv1_0',max_pool0,3,64,128)
        conv1_1 = self.conv_relu('conv1_1',conv1_0,3,128,128)
        max_pool1 = tf.nn.max_pool(conv1_1,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')        
        
        conv2_0 = self.conv_relu('conv2_0',max_pool1,3,128,256)
        conv2_1 = self.conv_relu('conv2_1',conv2_0,3,256,256)
        conv2_2 = self.conv_relu('conv2_2',conv2_1,3,256,256)
        max_pool2 = tf.nn.max_pool(conv2_2,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')

        conv3_0 = self.conv_relu('conv3_0',max_pool2,3,256,512)
        conv3_1 = self.conv_relu('conv3_1',conv3_0,3,512,512)
        conv3_2 = self.conv_relu('conv3_2',conv3_1,3,512,512)
        max_pool3 = tf.nn.max_pool(conv3_2,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')

        conv4_0 = self.conv_relu('conv4_0',max_pool3,3,512,512)
        conv4_1 = self.conv_relu('conv4_1',conv4_0,3,512,512)
        conv4_2 = self.conv_relu('conv4_2',conv4_1,3,512,512)
        max_pool4 = tf.nn.max_pool(conv4_2,ksize=[1,2, 2,1], strides=[1,2, 2,1], padding='SAME')

        fc_shape = max_pool4.get_shape().as_list()
        nodes = fc_shape[1]*fc_shape[2]*fc_shape[3]
        fc_reshape = tf.reshape(max_pool4, (self.batch, nodes), name='fc_reshape')

        fc0 = self.fc('fc0',fc_reshape,512,4096)
        fc1 = self.fc('fc1',fc0,4096,4096)
        fc_o = self.fc_out('fc_o',fc1,4096,10)

        #sm = tf.nn.softmax(fc2)
        self.loss_val = self.loss(fc_o,label)
        self.acc = tf.reduce_mean((tf.cast(tf.equal(tf.math.argmax(fc_o,1),tf.cast(label,tf.int64)),tf.float32)))
        return fc_o
    def loss(self,logits,label):
        labels = tf.cast(label, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean
    
    def calc_accuracy(self):
        self.dummy = 0

        

def main(_):

    

    input_tensor = tf.constant(np.random.rand(2,32, 32,3), dtype=tf.float32)
    #logits = tf.constant(np.random.rand(2,10), dtype=tf.float32)
    label = tf.constant([1,2], dtype=tf.int32)

    
    
    

    net = Model(2)

    #net.all(input_tensor,label)
    res = net.model(input_tensor,label)

    
    init = tf.initializers.global_variables()
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)   

    sess.run(init)

    print(sess.run(res))
    print(sess.run(net.acc))
    print(sess.run(net.loss_val))
    #print(sess.run(   tf.equal(tf.math.argmax(res,1),tf.cast(label,tf.int64))   ))
    #print(sess.run(tf.reduce_mean((tf.cast(tf.equal(tf.math.argmax(res,1),label),tf.float32)))))
    #print(sess.run(net.acc))
    #print(sess.run(tf.shape(input_tensor)))
    #print(sess.run(net.loss_val))

    

if __name__ == '__main__':
	tf.app.run()