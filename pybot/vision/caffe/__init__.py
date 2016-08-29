# Author: Sudeep Pillai <spillai@csail.mit.edu>
# License: MIT

def setup_rcnn(method, data_dir, net): 
    if method == 'fast_rcnn':
        print('=====> Fast RCNN')
        # fast_rcnn: ['vgg16', 'vgg_cnn_m_1024', 'caffenet']
        from pybot.vision.caffe.fast_rcnn_utils import FastRCNNDescription
        rcnn = FastRCNNDescription(data_dir, net=net)
    elif method == 'faster_rcnn':
        print('=====> Faster RCNN')
        # faster_rcnn: ['vgg16', 'zf']
        from pybot.vision.caffe.faster_rcnn_utils import FasterRCNNDescription
        rcnn = FasterRCNNDescription(
            data_dir, with_rpn=False, 
            net=net, opt_dir='fast_rcnn')
    else: 
        raise ValueError('Unknown rcnn method {}'.format(method))

    return rcnn
