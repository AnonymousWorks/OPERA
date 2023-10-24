import torch
from torch.nn import Module
import openvino as ov


class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], kernel_size=3)

torch_model = max_pool2d().float().eval()
input_data = torch.randn([7, 176, 3, 10], dtype=torch.float32)
trace = torch.jit.trace(torch_model, [input.clone() for input in input_data])
input_shapes = list([inp.shape for inp in input_data])
# print(input_shapes)

ov_model = ov.convert_model(trace, input=input_shapes)

