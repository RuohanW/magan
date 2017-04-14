import tensorflow as tf
import nn_utils
import os

from PIL import Image
import transforms

size=64
channel=3
max_no=202599
img_key='img_raw'
file_tpl='%6d.jpg'
home_dir=os.path.expanduser('~')
celeb_source=os.path.join(home_dir, "Pictures/img_align_celeba")

default_attribs={
    img_key:tf.FixedLenFeature([], tf.string)
}

default_transf=transforms.Compose([transforms.Scale(size), transforms.CenterCrop(size), transforms.ToFloat(), transforms.Normalize(0.5, 0.5)])

def process_celebA(dest='celebA', celeb_source=celeb_source, force=False, transform=default_transf, files=None):
    dest_file='%s.tfr' % dest
    if os.path.exists(dest_file) and not force:
        return dest_file

    print 'Processing celeb data into a Tensorflow Record file. It may take a while depending on your computer speed...'
    if files is None:
        files=xrange(1, max_no)
    file_iter=[os.path.join(celeb_source, '%s.jpg' % str(i).zfill(6)) for i in files]

    writer = tf.python_io.TFRecordWriter(dest_file)

    for pic in file_iter:
        img=transform(Image.open(pic))
        example=tf.train.Example(features=tf.train.Features(feature={img_key:nn_utils.bytes_feature(img.tostring())}))
        writer.write(example.SerializeToString())
    writer.close()
    print 'Preprocessing finished'
    return dest_file

def read_celebA(files, epochs, batch_size, attribs=default_attribs):
    def process(features):
        image = tf.decode_raw(features[img_key], tf.float32)
        image.set_shape([size * size * channel])
        image=tf.reshape(image, [size, size, channel])
        return [image]

    return nn_utils.tfr_batch(files, epochs, batch_size, attribs, process)

def test():
    data_file=process_celebA('celebA')
    data_batch=read_celebA(data_file, 1, 1)
    with tf.Session() as sess:
        nn_utils.train_init(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        print sess.run([data_batch])
        print sess.run([data_batch])


if __name__=='__main__':
    test()





