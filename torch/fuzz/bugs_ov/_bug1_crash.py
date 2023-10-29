import torch
from torch.nn import Module
import openvino as ov


class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], kernel_size=3)

torch_model = max_pool2d().float().eval()
input_data = torch.randn([7, 176, 3, 10], dtype=torch.float32)
input_data = [input_data]  # fix the issue-2
trace = torch.jit.trace(torch_model, [input.clone() for input in input_data])
trace = torch.jit.freeze(trace)  # fix the issue-1, it can convert to OV_IR, but lead to crash when compile_model()

input_shapes = list([inp.shape for inp in input_data])
print(input_shapes)

ov_model = ov.convert_model(trace, input=input_shapes, example_input=input_data)
print("convert to ov successfully...")
ir_path = f"/_temp_OVIR.xml"  # file must ends with 'xml'
ov.save_model(ov_model, ir_path)
core = ov.Core()
model = core.read_model(ir_path)

import ipywidgets as widgets
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

compiled_model = core.compile_model(model=model, device_name=device.value)
