import openvino as ov

ov_model = ov.convert_model('../_temp_model/ConvTranspose.onnx')  # input=input_shapes
