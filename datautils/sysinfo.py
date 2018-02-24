
import os
import platform 
 

def showSystemInfo():

    print("\n##### Installed System overview #####\n")

    apa = memoryCheck()
    if (os.name =='nt'):
        print("Installed Memory: " + apa.windowsRam())
    else:
        print("Installed Memory: " + apa.linuxRam())

    print()

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
        print("\nShowing CPU (and optional GPU info if properly set up) used by tensorflow")
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())
    except:
        print("\nSkipped tensorflow CPU/GPU check")





class memoryCheck():
    """Checks memory of a given system"""

     
    def windowsRamOLD(self):
        """Uses Windows API to check RAM in this OS"""
        kernel32 = windll.kernel32
        #kernel32 = ctypes.windll.kernel32
        #c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong)
            ]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(byref(memoryStatus))
 
        return int(memoryStatus.dwTotalPhys/1024**2)

    def windowsRam(self):
        import ctypes
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ('dwLength', c_ulong),
                ('dwMemoryLoad', c_ulong),
                ('dwTotalPhys', c_ulong),
                ('dwAvailPhys', c_ulong),
                ('dwTotalPageFile', c_ulong),
                ('dwAvailPageFile', c_ulong),
                ('dwTotalVirtual', c_ulong),
                ('dwAvailVirtual', c_ulong)
            ]
 
        
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
        #totalRam = memoryStatus.dwTotalPhys / (1024*1024)
        #availRam = memoryStatus.dwAvailPhys / (1024*1024)
        #There might be a bug here.. Please recheck.
        totalRam = (memoryStatus.dwTotalPhys * memoryStatus.dwLength) / (1024*1024)
        availRam = totalRam * (memoryStatus.dwMemoryLoad/100)   #in percent of used memory
        
        return ("\nTotal Ram:" + str(round(totalRam,2)) + " MiB" + "\nAvailable Ram:" +  str(round(availRam,2)) + "MiB " ) 
 
    def linuxRam(self):
        """Returns the RAM of a linux system"""
        totalMemory = os.popen("free -m").readlines()[1].split()[1]
        return (totalMemory + " MiB")

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
