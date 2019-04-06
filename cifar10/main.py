
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import numpy as np

import data_loader as DL
import model as M


epoch = 3


def test(sess,loader,model,cnt):
    loss = 0
    acc = 0
    num = 0
    sess.run(loader.init_op)
    while True:
        try:
            _ = sess.run(model.model(loader.data_batch,loader.label_batch) )
                        
            loss += sess.run(model.loss_val)
            acc += sess.run(model.acc)
            num += 1
            print('test')
        except tf.errors.OutOfRangeError:
            break
    ## batch loop
    return loss/num,acc/num

def main(_):

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        dataloader = DL.Dataloader(100,'train')
        test_loader = DL.Dataloader(100,'test')
        #input_tensor = tf.constant(np.random.rand(10,32, 32,3), dtype=tf.float32)
        
        opt = tf.train.AdamOptimizer()
        

        with tf.device('/gpu:0'):
            _model = M.Model(dataloader.batch)

            logits = _model.model(dataloader.data_batch,dataloader.label_batch)           

            #loss = _model.loss(logits,dataloader.label_batch)
            
            grads = opt.compute_gradients(_model.loss_val)
            
        apply_grad_op = opt.apply_gradients(grads)

        init = tf.initializers.global_variables()

        tf.summary.scalar('Loss', _model.loss_val, ['test'])
        tf.summary.scalar('Accuracy', _model.acc, ['test'])
        summary_op = tf.summary.merge_all('test')

        #onehot_labels = tf.one_hot(indices=tf.cast(dataloader.label_batch, tf.int32), depth=10)
        #accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        sess.run([dataloader.init_op,init])

        summary_writer = tf.summary.FileWriter('./tmp/', sess.graph)


        
        cnt = 0
        for ep in range(epoch):
            print('%d epoch'%ep)
            sess.run(dataloader.init_op)
            while True:
                try:
                    _ = sess.run(apply_grad_op)
                    
                    if (cnt%100) is 0:
                        
                        l = sess.run(_model.loss_val)
                        a = sess.run(_model.acc)
                        
                        print('Train Loss = %f , Train Acc = %f per' % (l,a))

                        #tl,ta = test(sess,test_loader,_model,cnt)
                        #print('Test Loss = %f , Test Acc = %f per' % (tl,ta))

                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, global_step=cnt)
                except tf.errors.OutOfRangeError:
                    break
                cnt += 1
            ## batch loop
        ## epoch loop

if __name__ == '__main__':
	tf.app.run()