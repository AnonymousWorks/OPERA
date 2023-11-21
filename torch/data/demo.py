# test_id: 19260
para_0 = torch.randint(1, 100, [1, 2, 4, 5], dtype=torch.int64)
para_1 = torch.randint(1, 100, [2, 2, 2, 3], dtype=torch.int64)
para_2 = torch.randint(1, 100, [4], dtype=torch.int64)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (0, 0)
para_6 = 2
para_7 = (1, 1)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)