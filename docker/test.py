# confirm tensorflow sees the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
from keras import backend
print(backend.tensorflow_backend._get_available_gpus())
