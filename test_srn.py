import numpy as np
from PIL import Image
import caffe
import sys

caffe.set_device(3)
caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
model_weights = 'snapshot/train_iter_10000.caffemodel'
test_image = sys.argv[1]

im = Image.open(test_image)
in_ = np.array(im, dtype=np.float32)
in_ = in_[np.newaxis, ...]

net = caffe.Net(model_def, model_weights, caffe.TEST)
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
net.forward()
out = net.blobs['fc8'].data[0].argmax(axis=0)

print out
