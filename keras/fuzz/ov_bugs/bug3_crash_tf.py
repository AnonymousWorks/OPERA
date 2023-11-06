import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, models
import numpy as np

layer = keras.layers.SpatialDropout2D(rate=0.1)
input_shape = [12, 10, 6, 18]
input_data = np.random.random(input_shape)
weights = layer.get_weights()
layer.set_weights(weights)

x = layers.Input(shape=input_shape[1:], dtype="float32")
y = layer(x)
model = models.Model(x, y)
model.summary()
res_keras = model(input_data)
tf.saved_model.save(model, "tf_model")
