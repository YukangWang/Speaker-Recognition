#!/usr/bin/env python
import caffe
import numpy as np
import os
import sys

# init
caffe.set_device(int(sys.argv[1]))
#caffe.set_device(3)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
# solver.net.copy_from(weights)

solver.step(10000)
