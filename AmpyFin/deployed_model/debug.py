# tf_check.py
# Simple script to verify your TensorFlow installation
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow was imported successfully!")
    
    # Check if CUDA is available (for GPU support)
    if tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else tf.config.list_physical_devices('GPU'):
        print("GPU is available and detected by TensorFlow!")
    else:
        print("GPU is not available. TensorFlow will use CPU only.")
    
    # Try importing Keras modules
    from tensorflow import keras
    print(f"Keras version: {keras.__version__}")
    print("Keras was imported successfully!")
    
    # Try importing specific modules that give you trouble
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras import callbacks
    print("All specific Keras modules were imported successfully!")
    
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("\nInstallation suggestions:")
    
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        print("For Apple Silicon (M1/M2/M3) Macs, use:")
        print("pip install tensorflow-macos tensorflow-metal")
    else:
        print("For most systems, use:")
        print("pip install tensorflow")
    
    if sys.version_info.major == 3 and sys.version_info.minor >= 11:
        print(f"\nWARNING: You're using Python {sys.version_info.major}.{sys.version_info.minor}")
        print("TensorFlow is most stable on Python 3.7-3.10. Consider downgrading your Python version.")