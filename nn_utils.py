import prettytensor as pt
import tensorflow as tf
import os, shutil
import numpy as np

home_dir=os.path.expanduser('~')
log_root=os.path.join(home_dir, "workspace/tb/")
save_dir='./save'

g_step=tf.Variable(0, name='global_step', trainable=False)

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_to_var(data):
    data_tpl = tf.placeholder(dtype=data.dtype, shape=data.shape)
    var = tf.Variable(data_tpl, trainable=False, collections=[])
    return data_tpl, var

def data_in(dataset, epochs, batch_size, batch_buffer=8, has_label=True):
    if has_label:
        data, label=dataset
        data_tpl, data_var=data_to_var(data)
        label_tpl, label_var=data_to_var(label)
        inputs=[data_var, label_var]
        run=[data_var.initializer, label_var.initializer]
        feed_dict={data_tpl:data, label_tpl:label}
    else:
        data_tpl, data_var = data_to_var(dataset)
        input=[data_var]
        run=[data_var.initializer]
        feed_dict = {data_tpl: dataset}

    def var_init(sess):
        sess.run(run, feed_dict=feed_dict)

    producers=tf.train.slice_input_producer(input, num_epochs=epochs, capacity=batch_size*batch_buffer)
    return var_init, tf.train.batch(producers, batch_size=batch_size, capacity=batch_size*batch_buffer, num_threads=2)

def read_and_decode(filename_queue, attribs, func):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features=attribs)

    return func(features)

def tfr_batch(files, epochs, batch_size, attribs, func, batch_buffer=8):
    if not isinstance(files, list):
        files=[files]
    filename_queue = tf.train.string_input_producer(files, num_epochs=epochs)
    data=read_and_decode(filename_queue, attribs, func)
    data_batch = tf.train.shuffle_batch(data, batch_size=batch_size, capacity=batch_buffer*batch_size,
                                        min_after_dequeue=(batch_buffer-3)*batch_size, num_threads=2)
    return data_batch


def kronecker(x, y):
    #x is a tensor, and y is a matrix
    input_shape = tf.shape(x)
    mat_shape=tf.shape(y)

    # perform a tensor-matrix kronecker product
    fx = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1])
    fx = tf.expand_dims(fx, -1)  # (bchw)x1
    mat = tf.expand_dims(tf.reshape(y, [-1]), 0)  # 1x(sh x sw)
    prod = tf.matmul(fx, mat)  # (bchw) x(sh x sw)
    prod = tf.reshape(prod, [-1, input_shape[3], input_shape[1], input_shape[2], mat_shape[0], mat_shape[1]])
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, [-1, input_shape[1] * mat_shape[0], input_shape[2] * mat_shape[1], input_shape[3]])
    return prod

def batch_l2_loss(input, recon):
    return tf.reduce_sum(tf.square(input-recon), reduction_indices=[1, 2, 3])

def train_init(sess):
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

def get_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def init_log(exp, sess, log_root=log_root, max_to_keep=5, summary_list=None):
    dir=os.path.join(log_root, exp)
    if not os.path.exists(dir):
        os.makedirs(dir)
    i=1
    try:
        i=max([int(run[3:]) for run in os.listdir(dir) if run.startswith('run')])+1
    except:
        pass
    if i>max_to_keep:
        for j in xrange(1, i):
            shutil.rmtree(os.path.join(dir, 'run%d' % j))
        i=1
    target=os.path.join(dir, 'run%d' % i)
    os.mkdir(target)
    if summary_list is None:
        summary_op = tf.summary.merge_all()
    else:
        summary_op=tf.summary.merge(summary_list)
    return summary_op, tf.summary.FileWriter(target, sess.graph)

def get_saver(old_scope, new_scope=None):
    t_vars=get_vars(old_scope)
    if new_scope:
        i=len(old_scope)
        new_var_list={}
        for var in t_vars:
            new_var_list[new_scope + var.op.name[i:]] = var
        t_vars=new_var_list
        max_to_keep=1
    else:
        max_to_keep=3
    return tf.train.Saver(var_list=t_vars, max_to_keep=max_to_keep)

def save_var(sess, saver, file, step=0):
    saver.save(sess, os.path.join(save_dir, file), global_step=step)

def export_graph(file):
    tf.train.export_meta_graph(os.path.join(save_dir, file))

def epoch_step(size, batch_size):
    if size % batch_size==0:
        return size/batch_size
    return size/batch_size+1

def load(sess, file, vars):
    if isinstance(vars, str):
        t_vars=get_vars(vars)
    else:
        t_vars=vars
    target=os.path.join(save_dir, file)
    saver=tf.train.Saver(var_list=t_vars)
    saver.restore(sess, target)

class TrainObj:
    def eval(self, step, result):
        pass

def train_wrap(sess, ops, train_obj=TrainObj(), tb_log=None, saver=None, filename=None, save_interval=10000, summary_list=None):
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if tb_log:
        s_op, s_writer=init_log(tb_log, sess, summary_list=summary_list)
        try:
            print('graph running')
            while not coord.should_stop():
                step, s_log, result=sess.run([g_step, s_op, ops])
                train_obj.eval(step, result)
                s_writer.add_summary(s_log, step)
                if saver and step % save_interval==0:
                    save_var(sess, saver, filename, step)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)
        s_writer.close()
    else:
        try:
            print('graph running')
            while not coord.should_stop():
                step, result=sess.run([g_step, ops])
                train_obj.eval(step, result)
                if saver and step % save_interval==0:
                    save_var(sess, saver, filename, step)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
        coord.join(threads)
    print('total number of steps trained:%d' % step)
    if saver:
        save_var(sess, saver, filename, step)


def make_grid(imgs, nrow=8, padding=2):
    import math
    # make the mini-batch of images into a grid
    shape=imgs.shape
    nmaps = shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(shape[1] + padding), int(shape[2] + padding)
    grid = np.zeros([height * ymaps, width * xmaps, shape[3]]).astype(np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            y1=y*height+1+padding//2
            x1=x*width+1+padding//2
            grid[y1:y1+height-padding, x1:x1+width-padding]=imgs[k]
            k = k + 1
    return np.squeeze(grid)


def save_image(imgs, filename, grayscale=False, nrow=8, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    if imgs.dtype != np.uint8:
        if grayscale:
            imgs=imgs*255
        else:
            imgs = (imgs * 0.5 + 0.5) * 255
        imgs = imgs.astype('uint8')
    grid = make_grid(imgs, nrow=nrow, padding=padding)
    im = Image.fromarray(grid)
    im.save(filename)
