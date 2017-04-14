import prettytensor as pt
import tensorflow as tf
import deconv2d_release
import nn_utils

from adamax import AdamaxOptimizer

batch_size=64
log_dir='./tb'
mnist_dir='./mnist'


def conv_ae(scope, filter_no, img_length=64, bottleneck=4, channel=3, act_fn=tf.nn.relu, last_act=tf.tanh):
    with tf.variable_scope(scope):
        with pt.defaults_scope(activation_fn=act_fn):
            layer = pt.template('batch').conv2d(4, filter_no, stride=2, name='conv1')
            img_length>>=1
            i=0
            while img_length>bottleneck:
                filter_no<<=1
                img_length>>=1
                layer=layer.conv2d(4, filter_no, stride=2, name='conv%d' % (i+2))
                i+=1

            for j in range(i):
                filter_no>>=1
                img_length<<=1
                layer=layer.deconv2d(4, filter_no, [-1, img_length, img_length, filter_no], stride=2, name='deconv%d' % (j+1))

            img_length<<=1
            return layer.deconv2d(4, channel, [-1, img_length, img_length, channel], stride=2, name='deconv%d' % (i+1),
                                  activation_fn=last_act)
    return cae_tpl


def conv_gen_bn(scope, filter_no, z_dim, img_size=64, bottleneck=4, channel=3, bn_arg=False, act_fn=tf.nn.relu, last_act=tf.tanh):
    if not bn_arg:
        bias=tf.zeros_initializer()
    else:
        bias=None
    with tf.variable_scope(scope):
        with pt.defaults_scope(activation_fn=act_fn, batch_normalize=bn_arg):
            layer = pt.template('batch').reshape((-1, 1, 1, z_dim)) \
                .deconv2d(bottleneck, filter_no, [-1, bottleneck, bottleneck, filter_no], stride=1,
                          edges=pt.pretty_tensor_class.PAD_VALID, name='deconv1', bias=bias)

            img_length = bottleneck
            i=2
            while img_length<img_size/2:
                filter_no >>= 1
                img_length <<= 1
                layer = layer.deconv2d(4, filter_no, [-1, img_length, img_length, filter_no], stride=2,
                                       name='deconv%d' % i,
                                       bias=bias)
                i+=1

            img_length <<= 1
            return layer.deconv2d(4, channel, [-1, img_length, img_length, channel], stride=2, activation_fn=last_act,
                                  name='deconv%d' % i, batch_normalize=False)

def ae_def(scope):
    with tf.variable_scope(scope):
        with pt.defaults_scope(activation_fn=tf.nn.relu):
            cae_tpl=(pt.template('batch').flatten()
                    .fully_connected(256, name='fc1')
                    .fully_connected(256)
                    .fully_connected(28*28, name='fc2', activation_fn=tf.nn.sigmoid)
                     .reshape([-1, 28, 28, 1]))
    return cae_tpl

def magan_train(exp_name, disc_tpl, disc_scope, gen, gen_scope, data_batch, max_epochs, epoch_step,
                optim=AdamaxOptimizer, var_init=None, beta1=0.5, lr=0.0005, grayscale=False):
    margin = tf.Variable(initial_value=0, dtype=tf.float32)
    with tf.Session() as sess:
        real_recon = disc_tpl.construct(batch=data_batch)
        real_energy = nn_utils.batch_l2_loss(data_batch, real_recon)

        fake_recon = disc_tpl.construct(batch=gen)
        fake_energy = nn_utils.batch_l2_loss(fake_recon, gen)

        fake_loss = fake_energy
        real_loss = real_energy + tf.maximum(margin - fake_loss, 0)

        g_step = nn_utils.g_step
        real_train_op = optim(beta1=beta1, learning_rate=lr).minimize(real_loss, var_list=nn_utils.get_vars(disc_scope),
                                                          global_step=g_step)
        fake_train_op = optim(beta1=beta1, learning_rate=lr).minimize(fake_loss, var_list=nn_utils.get_vars(gen_scope))

        real_e=tf.reduce_mean(real_energy)
        fake_e=tf.reduce_mean(fake_energy)
        real_summary = tf.summary.scalar('real_energy', real_e)
        fake_summary = tf.summary.scalar('fake_energy', fake_e)

        real_sum = tf.Variable(initial_value=0, dtype=tf.float32)
        real_add = real_sum.assign_add(real_e)

        fake_sum = tf.Variable(initial_value=0, dtype=tf.float32)
        fake_add = fake_sum.assign_add(fake_e)

        margin_update = tf.cond(tf.logical_and(tf.less(real_sum, fake_sum), tf.less(real_sum / epoch_step, margin)),
                                lambda: margin.assign(real_sum / epoch_step), lambda: margin.assign_add(0))

        if var_init:
            var_init(sess)
        nn_utils.train_init(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        s_op, s_writer = nn_utils.init_log(exp_name, sess, log_root=log_dir, summary_list=[real_summary, fake_summary])
        for i in range(2):
            for j in range(epoch_step):
                sess.run([real_train_op])
        for j in range(epoch_step):
            sess.run([real_add])

        sess.run(margin.assign(real_sum/epoch_step))
        pre_fake_sum = 0
        k=0
        print 'initial estimate of margin:%f' % sess.run(margin)
        for i in range(max_epochs):
            sess.run([real_sum.assign(0), fake_sum.assign(0)])
            for j in range(epoch_step):
                _, step, s_log, _=sess.run([real_train_op, g_step, s_op, real_add])
                sess.run([fake_train_op, fake_add])
                s_writer.add_summary(s_log, step)

            if pre_fake_sum==0:
                pre_fake_sum=sess.run(fake_sum)
            else:
                t= sess.run([fake_sum])
                if t>pre_fake_sum:
                    sess.run(margin_update)
                    print "margin updated to %f" % sess.run(margin)
                pre_fake_sum=t
            imgs, fake_energy_py = sess.run([gen, fake_energy])
            sort_tmp=fake_energy_py.tolist()
            sort_tmp=[x[0] for x in sorted(enumerate(sort_tmp), key=lambda x:x[1])]
            imgs_sorted=imgs[sort_tmp]
            nn_utils.save_image(imgs_sorted, 'save/%s-%d.jpg' % (exp_name, k), nrow=8, grayscale=grayscale)
            k += 1
            print "E(x):%f, E(z):%f" % tuple(sess.run([real_sum / epoch_step, fake_sum / epoch_step]))

        s_writer.close()

        g_saver = nn_utils.get_saver(gen_scope)
        nn_utils.save_var(sess, g_saver, gen_scope)
        d_saver = nn_utils.get_saver(disc_scope)
        nn_utils.save_var(sess, d_saver, disc_scope)

def celebA_train(data_dir):
    import celebA_in
    exp_name='celebA'
    gen_scope = '%s-gen' % exp_name
    disc_scope = '%s-disc' % exp_name

    z_dim=350
    max_epochs=50

    disc_tpl = conv_ae(disc_scope, 64, act_fn=nn_utils.lrelu)
    gen_tpl = conv_gen_bn(gen_scope, 512, z_dim)
    z = tf.random_normal([batch_size, z_dim])
    gen = gen_tpl.construct(batch=z)

    epoch_step=nn_utils.epoch_step(celebA_in.max_no, batch_size)
    celeb_file = celebA_in.process_celebA(celeb_source=data_dir)
    data_batch = celebA_in.read_celebA(celeb_file, max_epochs, batch_size)
    magan_train(exp_name, disc_tpl, disc_scope, gen, gen_scope, data_batch,max_epochs, epoch_step)

def mnist_train():
    from mnist_in import get_mnist_train
    exp_name='mnist'
    gen_scope = '%s-gen' % exp_name
    disc_scope = '%s-disc' % exp_name
    z_dim=50
    max_epochs=200

    gen_tpl = conv_gen_bn(gen_scope, 128, z_dim, img_size=28, bottleneck=7, channel=1, last_act=tf.nn.sigmoid)
    disc_tpl=ae_def(disc_scope)

    z = tf.random_normal([batch_size, z_dim])
    gen = gen_tpl.construct(batch=z)

    epoch_step, data_tuple = get_mnist_train(max_epochs, batch_size, mnist_dir=mnist_dir, validation_size=0, has_label=False)
    var_init, data_batch = data_tuple
    magan_train('mnist', disc_tpl, disc_scope, gen, gen_scope, data_batch, max_epochs, epoch_step,
                var_init=var_init, grayscale=True)

def cifar10_train(data_dir):
    import cifar10_data, os
    imgs, _ = cifar10_data.load(data_dir)
    raw_batch = cifar10_data.cifar_preloaded(imgs, batch_size)
    data_batch = (tf.cast(tf.transpose(raw_batch, perm=[0, 2, 3, 1]), tf.float32) / 255 - 0.5) / 0.5

    z_dim=320
    max_epochs=200

    exp_name = 'cifar10'
    gen_scope = '%s-gen' % exp_name
    disc_scope = '%s-disc' % exp_name

    disc_tpl = conv_ae(disc_scope, 128, img_length=32, act_fn=nn_utils.lrelu)
    gen_tpl = conv_gen_bn(gen_scope, 512, z_dim, img_size=32)
    z = tf.random_normal([batch_size, z_dim])
    gen = gen_tpl.construct(batch=z)
    epoch_step=nn_utils.epoch_step(50000, batch_size)
    magan_train(exp_name, disc_tpl, disc_scope, gen, gen_scope, data_batch, max_epochs, epoch_step)

if __name__=='__main__':
    import sys
    exp_name=sys.argv[1]
    if exp_name=='cifar':
            assert len(sys.argv)>2, 'please specify data directory'
            cifar10_train(sys.argv[2])
    elif exp_name=='celebA':
        assert len(sys.argv)>2, 'please specify data directory'
        celebA_train(sys.argv[2])
    elif exp_name=='mnist':
        mnist_train()
    else:
        raise Exception('unknown experiment')