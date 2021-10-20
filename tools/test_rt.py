import sys
sys.path.append('./')
import os
import numpy as np
from easydict import EasyDict as edict

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from modules import rt_common as common
from modules.misc import Timer

cfg = edict({
    'onnx_file': '/home/nvidia/FXKJ/onnx_test/data/deeplabv3plus_mobilenet_fix_sim.onnx',
    'inp_size': [1, 3, 640, 640],
    'ENABLE_TR_CACHE': True
})


def main():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    engine = common.get_engine(cfg.onnx_file, TRT_LOGGER, cfg.ENABLE_TR_CACHE)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    shapes = list(engine.get_binding_shape(x) for x in engine)
    in_shapes = shapes[0]
    out_shapes = shapes[1:]
    ## read images
    random_image = np.random.random(cfg.inp_size).astype(np.float32)
    # start inference
    print(">>> start inference")
    timer_infer = Timer('inference')
    for i in range(11):
        if i > 0:
            timer_infer.start()
        
        inputs[0].host = random_image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        for j in range(len(trt_outputs)):
            trt_outputs[j] = trt_outputs[j].reshape(out_shapes[j])
        
        if i > 0:
            timer_infer.pause()
    timer_infer.print_time()
if __name__ == '__main__':
    main()