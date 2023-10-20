# test_id: 1
para_0 = torch.randn([2, 3, 16, 13], dtype=torch.float64)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float64)
para_2 = torch.randn([2], dtype=torch.float64)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 2
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float64)
para_1 = torch.randn([3], dtype=torch.float64)
para_2 = torch.randn([3], dtype=torch.float64)
para_3 = torch.randn([3], dtype=torch.float64)
para_4 = torch.randn([3], dtype=torch.float64)
para_5 = True
para_6 = 0.0006353240152477764
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 3
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float64)
para_1 = torch.randn([3], dtype=torch.float64)
para_2 = torch.randn([3], dtype=torch.float64)
para_3 = torch.randn([3], dtype=torch.float64)
para_4 = torch.randn([3], dtype=torch.float64)
para_5 = True
para_6 = 0.0005506607929515419
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 4
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float64)
para_1 = torch.randn([3], dtype=torch.float64)
para_2 = torch.randn([3], dtype=torch.float64)
para_3 = torch.randn([3], dtype=torch.float64)
para_4 = torch.randn([3], dtype=torch.float64)
para_5 = True
para_6 = 0.0002835270768358378
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 5
para_0 = torch.randn([2, 2, 4], dtype=torch.float64)
para_1 = (1, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',)
verify_model(pad().float().eval(), input_data=para_0)