import sys
sys.path.append('./')
import os
import numpy as np
from pathlib import Path
from easydict import EasyDict as edict
from modules.misc import Timer

import onnxruntime as ort

cfg = edict({
    'onnx_file': '/home/nvidia/FXKJ/onnx_test/data/deeplabv3plus_mobilenet_fix_sim.onnx',
    'inp_size': [1, 3, 640, 640]
})

if __name__ == '__main__':
    # read image
    timer = Timer('onnx-infer')
    random_image = np.random.random(cfg.inp_size).astype(np.float32)

    sess = ort.InferenceSession(cfg.onnx_file)
    input_name = sess.get_inputs()[0].name
    for i in range(10):
        if i == 0:
            onnx_out = sess.run(None, input_feed={input_name: random_image})
        else:
            timer.start()
            onnx_out = sess.run(None, input_feed={input_name: random_image})
            timer.pause()
    timer.print_time()
    pass