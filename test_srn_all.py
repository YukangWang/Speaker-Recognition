import numpy as np
from PIL import Image
import caffe
import sys

caffe.set_device(2)
caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
model_weights = 'snapshot/train_iter_10000.caffemodel'
#test_image = sys.argv[1]
net = caffe.Net(model_def, model_weights, caffe.TEST)

split_f = 'test.txt'
indices = open(split_f,'r').read().splitlines()
count = 0

for i in range(len(indices)):
	im = Image.open(indices[i].split()[0])
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[np.newaxis, ...]
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	net.forward()
	pred = net.blobs['fc8'].data[0].argmax(axis=0)
	img = indices[i].split()[0]
	gt = indices[i].split()[1]
	print 'img: '+img+'; '+'prediction: '+str(pred)+'; '+'ground truth: '+gt
	if pred == int(gt):
		count = count +1
acc = float(count)/len(indices)
print 'overall accuracy: '+str(acc)
