# test_id: 3138
para_0 = torch.randn([5, 5], dtype=torch.float32).cuda()
para_1 = 2.0
para_2 = True
class gumbel_softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.gumbel_softmax(args[0], para_1,para_2,)
verify_model(gumbel_softmax().float().eval(), input_data=para_0)
# test_id: 32450
verify_model(torch.nn.ReLU6().eval(), input_data=[torch.randint(1, 100, [4, 10], dtype=torch.uint8).cuda()])