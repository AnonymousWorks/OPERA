import torch
from torch.nn import Module
import openvino as ov


torch_model = torch.nn.ConstantPad2d(value=81866, padding=[],).eval()
input_data=[torch.randint(1, 100, [17, 16], dtype=torch.float32)]

trace = torch.jit.trace(torch_model, [input.clone() for input in input_data])
trace = torch.jit.freeze(trace)
input_shapes = list([inp.shape for inp in input_data])
print(input_shapes)

ov_model = ov.convert_model(trace, example_input=input_data)
