import torch
from torch.nn import Module
import openvino as ov
import ipywidgets as widgets
import os
import numpy as np


def compile_torch(model, input_shapes, input_data):
    temp_model_dir = "_temp_model"
    if not os.path.exists(temp_model_dir):
        os.mkdir(temp_model_dir)

    ov_model = ov.convert_model(model, input=input_shapes)
    ir_path = f"{temp_model_dir}/_temp_OVIR.xml"
    ov.save_model(ov_model, ir_path)
    core = ov.Core()
    model = core.read_model(ir_path)

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )

    compiled_model = core.compile_model(model=model, device_name=device.value)

    output_key = compiled_model.output(0)

    result = compiled_model(input_data)[output_key]
    return [result]

input_shapes = [7, 176, 3, 10]
input_data = torch.randn(input_shapes, dtype=torch.float32)


class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], kernel_size=3)

baseline_model = max_pool2d().float().eval()

baseline_outputs = baseline_model(*[input.clone() for input in input_data])
baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
print(baseline_outputs)

trace = torch.jit.trace(baseline_model, [input.clone() for input in input_data])
print(type(input_shapes))
input_shapes = list([inp.shape for inp in input_data])
print(type(input_shapes))


res_dlc = compile_torch(trace, input_shapes, input_data)

for i, baseline_output in enumerate(baseline_outputs):
    output = res_dlc[i]
    np.testing.assert_allclose(baseline_output, output, rtol=1e-3, atol=1e-3)
