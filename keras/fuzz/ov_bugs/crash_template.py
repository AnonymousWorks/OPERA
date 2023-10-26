from tensorflow import keras as keras

layer = keras.layers.CategoryEncoding(num_tokens=6, output_mode="multi_hot")
res = layer([1,2,3,4,5])
print(res)
