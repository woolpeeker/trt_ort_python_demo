```bash
virtualenv venv
source venv/bin/activate
pip install easydict
pip install six
pip install numpy==1.19.4
pip install pycuda
export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH
```

https://elinux.org/Jetson_Zoo  
jetson platform precompiled packages, including pytorch tensorflow tensorrt onnx ...  

numpy==1.19.5 cause "onnxruntime import error: Illegal instruction", replace it with v1.19.4