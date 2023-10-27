import torch
from torch.nn import Module
import openvino as ov
import ipywidgets as widgets
import os
import numpy as np


def compile_torch(model, input_data):
    temp_model_dir = "_temp_model"
    if not os.path.exists(temp_model_dir):
        os.mkdir(temp_model_dir)

    ov_model = ov.convert_model(model, example_input=input_data)
    ir_path = f"{temp_model_dir}/_temp_OVIR.xml"
    ov.save_model(ov_model, ir_path,  compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)

    # device = widgets.Dropdown(
    #     options=core.available_devices + ["AUTO"],
    #     value='AUTO',
    #     description='Device:',
    #     disabled=False,
    # )
    # print(type(device.value))

    compiled_model = core.compile_model(model=model, device_name='GPU')
    output_key = compiled_model.output(0)
    result = compiled_model(input_data)[output_key]
    return result


weight = torch.randn([2, 3, 3, 3], dtype=torch.float32)
input_data = torch.randn([2, 3, 16, 15], dtype=torch.float32)


class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], weight)


torch_model = conv2d().float().eval()
torch_outputs = torch_model(input_data).cpu().numpy()

trace = torch.jit.trace(torch_model, input_data)
trace = torch.jit.freeze(trace)

input_shapes = input_data.shape
res_ov = compile_torch(trace, input_data)
np.testing.assert_allclose(torch_outputs, res_ov, rtol=1e-3, atol=1e-3)
