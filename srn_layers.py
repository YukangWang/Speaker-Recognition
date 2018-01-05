import caffe

import numpy as np
from PIL import Image

class TestDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.test_dir = params['test_dir']
        self.split = params['split']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")

        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}.txt'.format(self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):
        # load image + label pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        self.idx += 1
        if self.idx == len(self.indices):
            self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        im = Image.open(self.indices[self.idx].split()[0])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[np.newaxis, ...]
        return in_

    def load_label(self, idx):
	lb = np.zeros((1,1), dtype=np.uint8)
        lb[0][0] = self.indices[self.idx].split()[1]
	label = lb[np.newaxis, ...]
        return label

class TrainDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.train_dir = params['train_dir']
        self.split = params['split']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")

        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}.txt'.format(self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):
        # load image + label pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        self.idx += 1
        if self.idx == len(self.indices):
            self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        im = Image.open(self.indices[self.idx].split()[0])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[np.newaxis, ...]
        return in_

    def load_label(self, idx):
        lb = np.zeros((1,1), dtype=np.uint8)
        lb[0][0] = self.indices[self.idx].split()[1]
        label = lb[np.newaxis, ...]
        return label

