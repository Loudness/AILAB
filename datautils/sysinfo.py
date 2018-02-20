
import os
import platform 


def showSystemInfo():

    print("\n##### Installed System overview #####\n")

    try:
        import sys 
        print("- Python Version: {}".format(sys.version))
    except:
        print("Weird: Python script running without Python installed")

    try:
        import tensorflow
        print("- TensorFlow Version: {}".format(tensorflow.__version__))
    except:
        print("Warning: Tensorflow not installed")

    print()
    try:
        import keras
        print("\n- Keras Version: {}".format(keras.__version__))
    except:
        print("Warning: Keras not installed")

    try:
        import pandas
        print("- Pandas Version: {}".format(pandas.__version__))
    except:
        print("Warning: Pandas not installed")

    try:
        import sklearn
        print("- Scikit-Learn Version: {}".format(sklearn.__version__))
    except:
        print("Warning: Scikit-Learn not installed")

    print("\n#### System Specific ####\n")
    try:
        import platform 
        print("- Machine: {}".format(platform.machine()))
        print("- Platform: {}".format(platform.platform() + "(" +platform.version()+")"))
        print("- Uname: {}".format(platform.uname()))
        if (os.name!='nt'):
            print("- Distro: {}".format(platform.dist()))
    except:
        print("Weird: Python script running without Python installed")


    try:
        #import tensorflow as tf
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        from tensorflow.python.client import device_lib
        print("- Tensorflow GPU/CPU INFO:" + device_lib.list_local_devices())
    except:
        print("\nSkipped tensorflow CPU/GPU check")

    """  Save this for later. 
    try:
        #untested for now.
        import py3nvml
        py3nvml.nvmlInit()
        for i in range(py3nvml.nvmlDeviceGetCount()):
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
            py3nvml.vmlDeviceGetName(handle),
            py3nvml.meminfo.free/1024.**2, py3nvml.meminfo.used/1024.**2, py3nvml.meminfo.total/1024.**2))
            py3nvml.nvmlShutdown()
    except:
       print("NVIDIA py3NVML not installed.. Skipping.")
       print("If you have a NVIDIA GPU, install with >pip install py3nvml")
    """
