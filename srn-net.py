# -*- coding: UTF-8 -*-
import caffe
def create_net(split, include_acc=False):

    net = caffe.NetSpec()

    pydata_params = dict(split=split)
    if split == 'train':
        pydata_params['train_dir'] = './'
        pylayer = 'TrainDataLayer'
    else:
        pydata_params['test_dir'] = './'
        pylayer = 'TestDataLayer'

    net.data, net.label = caffe.layers.Python(module='srn_layers', layer=pylayer, ntop=2, param_str=str(pydata_params))
    net.conv1 = caffe.layers.Convolution(net.data, kernel_size=11, stride=4, num_output=96, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.pool1 = caffe.layers.Pooling(net.conv1, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size=5, group=2, num_output=256, pad=2, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.pool2 = caffe.layers.Pooling(net.conv2, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    net.conv3 = caffe.layers.Convolution(net.pool2, kernel_size=3, num_output=384, pad=1, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu3 = caffe.layers.ReLU(net.conv3, in_place=True)
    net.conv4 = caffe.layers.Convolution(net.relu3, kernel_size=3, group=2, num_output=384, pad=1, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu4 = caffe.layers.ReLU(net.conv4, in_place=True)
    net.conv5 = caffe.layers.Convolution(net.relu4, kernel_size=3, group=2, num_output=256, pad=1, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu5 = caffe.layers.ReLU(net.conv5, in_place=True)
    net.pool5 = caffe.layers.Pooling(net.relu5, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    net.fc6 = caffe.layers.InnerProduct(net.pool5, num_output=256, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu6 = caffe.layers.ReLU(net.fc6, in_place=True)
    net.drop6 = caffe.layers.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
    net.fc7 = caffe.layers.InnerProduct(net.drop6, num_output=128, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.relu7 = caffe.layers.ReLU(net.fc7, in_place=True)
    net.drop7 = caffe.layers.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
    net.fc8 = caffe.layers.InnerProduct(net.drop7, num_output=24, weight_filler={"type": "xavier"},bias_filler={"type": "constant"}, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    net.loss = caffe.layers.SoftmaxWithLoss(net.fc8, net.label, loss_param=dict(normalize=True))
    net.acc = caffe.layers.Accuracy(net.fc8, net.label)
    return str(net.to_proto())

def write_net(train_proto, test_proto):

    with open(train_proto, 'w') as f:
        f.write(str(create_net('train')))

    with open(test_proto, 'w') as f:
        f.write(str(create_net('test', include_acc = True)))

def write_sovler(root, solver_proto, train_proto, test_proto):
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()
    sovler_string.train_net = train_proto
    sovler_string.test_net.append(test_proto)
    sovler_string.test_iter.append(108)
    sovler_string.test_interval = 999999
    sovler_string.base_lr = 1e-4
    sovler_string.momentum = 0.95
    sovler_string.weight_decay = 5e-4
    sovler_string.lr_policy = 'fixed'
    sovler_string.display = 20
    sovler_string.average_loss = 20
    sovler_string.iter_size = 32
    sovler_string.max_iter = 10000
    sovler_string.snapshot = 2000
    sovler_string.snapshot_prefix = root + 'snapshot/train'
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
    sovler_string.test_initialization = 0

    with open(solver_proto, 'w') as f:
        f.write(str(sovler_string))

if __name__ == '__main__':
    root = '/home/wangyukang/caffe/examples/Speaker-Recognition/'
    train_proto = root + 'train.prototxt'
    test_proto = root + 'test.prototxt'
    solver_proto = root + 'solver.prototxt'

    write_net(train_proto, test_proto)
    print "train.prototxt test.prototxt success"
    write_sovler(root, solver_proto, train_proto, test_proto)
    print "solver.prototxt success"

