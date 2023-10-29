import onnx
import onnxruntime
import openvino as ov

ov_model = ov.convert_model('../_temp_model/2.onnx')  # input=input_shapes
ir_path = f"temp_OVIR.xml"  # file must ends with 'xml'
ov.save_model(ov_model, ir_path, compress_to_fp16=False)
core = ov.Core()
model = core.read_model(ir_path)

compiled_model = core.compile_model(model=model, device_name="CPU")

# show the model structure
# input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
# network_input_shape = input_key.shape
print(output_key)

result = compiled_model(input_data)[output_key]
