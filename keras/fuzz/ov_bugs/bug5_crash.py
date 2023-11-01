import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

import openvino as ov

layer = keras.layers.ReLU(threshold=1)
input_shape = [1, 2, 3]
input_data = np.random.random(input_shape)
print(input_data)
x = layers.Input(shape=input_shape[1:], dtype="int8")
y = layer(x)
model = models.Model(x, y)
model.summary()
res_keras = model(input_data)

tf2_model_path = f"_temp_model"
tf.saved_model.save(model, tf2_model_path)
ov_model = ov.convert_model(tf2_model_path,  input=input_shape)
