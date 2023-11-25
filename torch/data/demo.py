# test_id: 1
verify_model(torch.nn.ELU(inplace=True,).eval(), input_data=[torch.randn([13, 10, 19], dtype=torch.float64))
# test_id: 124
para_0 = torch.randn([1, 1280], dtype=torch.float16)
para_1 = 0.2
para_2 = False
para_3 = False
class dropout(Module):
    def forward(self, *args):
        return torch.nn.functional.dropout(args[0], para_1,para_2,para_3,)
verify_model(dropout().float().eval(), input_data=para_0)


# test_id: 125
verify_model(torch.nn.Dropout(p=0.2,).eval(), input_data=[torch.randn([1, 1280], dtype=torch.float16)])
# test_id: 126
para_0 = torch.randn([1, 1280], dtype=torch.float16)
para_1 = torch.randn([1000, 1280], dtype=torch.float16)
para_2 = torch.randn([1000], dtype=torch.float16)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)