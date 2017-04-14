import nn_utils
import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist_dir=os.path.join(os.path.expanduser('~'), "workspace/mnist")

def get_mnist_train(epochs, batch_size, mnist_dir=mnist_dir, validation_size=10000, has_label=True):
    mnist = input_data.read_data_sets(mnist_dir, validation_size=validation_size)
    epoch_step=nn_utils.epoch_step(mnist.train.images.shape[0], batch_size)
    if has_label:
        return epoch_step, nn_utils.data_in([mnist.train.images.reshape(-1, 28, 28, 1), mnist.train.labels], epochs, batch_size)
    else:
        return epoch_step, nn_utils.data_in(mnist.train.images.reshape(-1, 28, 28, 1), epochs, batch_size, has_label=has_label)

def test():
    with tf.Session() as sess:
        batch=get_mnist_train(sess, 1, 1, validation_size=0, has_label=False)
        batch=tf.reshape(batch, [-1, 28, 28])
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init)
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess, coord=coord)
        count=0
        i=0
        try:
            while not coord.should_stop():
                a, b=sess.run([batch, batch])
                i+=1
                break
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    test()