# import tensorflow as tf
# from tensorflow import keras as keras
# from tensorflow.keras import layers, models
# import numpy as np
#
# import openvino as ov
# a = {'return_sequences':True,'return_state':True,'go_backwards':False,'stateful':False,'unroll':False,'time_major':False,'units':2,'activation':"tanh",'recurrent_activation':"hard_sigmoid",'use_bias':True,'unit_forget_bias':True,'kernel_regularizer':None,'recurrent_regularizer':None,'bias_regularizer':None,'activity_regularizer':None,'kernel_constraint':None,'recurrent_constraint':None,'bias_constraint':None,'dropout':0.0,'recurrent_dropout':0.0,'implementation':1,}
# layer = keras.layers.LSTM(**a)
# input_shape = [1, 2, 3]
# input_data = np.random.random(input_shape)
# # weights = layer.get_weights()
# # layer.set_weights(weights)
#
# x = layers.Input(shape=input_shape[1:], dtype="float32")
# y = layer(x)
# model = models.Model(x, y)
# # model.summary()
# res_keras = model(input_data)
#
# tf2_model_path = f"_temp_model"
# tf.saved_model.save(model, tf2_model_path)
# ov_model = ov.convert_model(tf2_model_path,  input=input_shape)
#
# ir_path = f"_temp_OVIR.xml"
# ov.save_model(ov_model, ir_path, compress_to_fp16=False)
# core = ov.Core()
# model = core.read_model(ir_path)
# compiled_model = core.compile_model(model=model, device_name="CPU")
#
# output_key = compiled_model.outputs
#
# for i,  output in enumerate(output_key):
#     res_dlc = compiled_model(input_data)[output]
#     np.testing.assert_allclose(res_keras[i], res_dlc, atol=1e-3, rtol=1e-3)
