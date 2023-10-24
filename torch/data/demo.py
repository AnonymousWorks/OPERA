# test_id: 1
para_0 = torch.randn([2, 3, 16, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 2
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0006353240152477764
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 3
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005506607929515419
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 4
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0002835270768358378
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 5
para_0 = torch.randn([2, 2, 4], dtype=torch.float32)
para_1 = (1, 1)
class pad(Module):
    def forward(self, *args):
        return torch.nn.functional.pad(args[0], para_1,mode='replicate',)
verify_model(pad().float().eval(), input_data=para_0)
# test_id: 6
para_0 = torch.randn([7, 176, 3, 10], dtype=torch.float32)
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], kernel_size=3,stride=None,padding=1,dilation=2,ceil_mode=True,)
verify_model(max_pool2d().float().eval(), input_data=para_0)
# test_id: 7
para_0 = torch.randn([2, 3, 21, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 8
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=3,padding=0,dilation=4,).eval(), input_data=[torch.randn([2, 3, 20, 14], dtype=torch.float32)])
# test_id: 9
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0004962779156327543
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 10
para_0 = torch.randn([2, 2, 10, 7], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = [1, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 11
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003679175864606328
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 12
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=2,stride=1,padding=2,dilation=4,).eval(), input_data=[torch.randn([2, 2, 13, 15], dtype=torch.float32)])
# test_id: 13
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00038182512409316535
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 14
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005353319057815846
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 15
para_0 = torch.randn([2, 3, 20, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (3, 3)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 16
para_0 = torch.randn([2, 3, 16, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 17
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=2,padding=2,dilation=4,).eval(), input_data=[torch.randn([2, 3, 14, 13], dtype=torch.float32)])
# test_id: 18
para_0 = torch.randn([3, 4, 5], dtype=torch.float32)
para_1 = 2
class softmax(Module):
    def forward(self, *args):
        return torch.nn.functional.softmax(args[0], para_1,_stacklevel=5,)
verify_model(softmax().float().eval(), input_data=para_0)
# test_id: 19
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.010638297872340425
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 20
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=1,padding=1,dilation=2,).eval(), input_data=[torch.randn([2, 3, 22, 13], dtype=torch.float32)])
# test_id: 21
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=2,dilation=3,).eval(), input_data=[torch.randn([2, 3, 15, 13], dtype=torch.float32)])
# test_id: 22
para_0 = torch.randn([2, 2, 12, 8], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = [0, 1]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 23
para_0 = torch.randn([2, 3, 20, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 24
para_0 = torch.randn([2, 3, 13, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (0, 0)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 25
para_0 = torch.randn([2, 3, 18, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 26
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 3, 18, 16], dtype=torch.float32)])
# test_id: 27
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=3,padding=1,dilation=1,).eval(), input_data=[torch.randn([2, 3, 23, 14], dtype=torch.float32)])
# test_id: 28
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00031387319522912746
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 29
para_0 = torch.randn([2, 3, 15, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 30
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00034818941504178273
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 31
para_0 = torch.randn([5, 16, 8, 10], dtype=torch.quint8)
para_1 = (2, 2)
para_2 = (1, 1)
para_3 = (1, 1)
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)
# test_id: 32
para_0 = torch.randn([2, 2, 20, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = [0, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 33
para_0 = torch.randn([2, 3, 13, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 34
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0009276437847866419
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 35
para_0 = torch.randn([2, 3, 18, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 36
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.0026041666666666665
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 37
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0024937655860349127
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 38
para_0 = torch.randn([2, 2, 10, 6], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = [0, 1]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 39
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003705075954057058
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 40
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0006501950585175553
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 41
para_0 = torch.randn([2, 3, 22, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 42
para_0 = torch.randn([2, 2, 7, 5], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = [1, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 43
para_0 = torch.randn([2, 3, 14, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 44
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.001455604075691412
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 45
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0001684352366515075
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 46
para_0 = torch.randn([2, 3, 15, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (0, 0)
para_5 = (3, 3)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 47
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00041631973355537054
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 48
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.000351000351000351
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 49
para_0 = torch.randn([2, 3, 13, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 50
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.008064516129032258
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 51
para_0 = torch.randn([2, 3, 14, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 52
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003652300949598247
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 53
para_0 = torch.randn([1, 1, 14, 14], dtype=torch.float32)
para_1 = torch.randn([32, 1, 3, 3], dtype=torch.float32)
para_2 = torch.randn([32], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 54
para_0 = torch.randn([1, 1, 5, 8], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=(2, 3),stride=(2, 3),)
verify_model(unfold().float().eval(), input_data=para_0)
# test_id: 55
para_0 = torch.randn([2, 3, 19, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 56
para_0 = torch.randn([2, 3, 20, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (1, 1)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 57
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00029231218941829873
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 58
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.004149377593360996
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 59
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=2,padding=1,dilation=2,).eval(), input_data=[torch.randn([2, 3, 19, 14], dtype=torch.float32)])
# test_id: 60
para_0 = torch.randn([2, 3, 20, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 61
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00021795989537925023
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 62
para_0 = torch.randn([10, 1, 1], dtype=torch.quint8)
class adaptive_avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.adaptive_avg_pool2d(args[0], output_size=1,)
verify_model(adaptive_avg_pool2d().float().eval(), input_data=para_0)
# test_id: 63
para_0 = torch.randn([2, 2, 8, 5], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = [1, 1]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 64
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.004098360655737705
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 65
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.003424657534246575
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 66
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0014903129657228018
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 67
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.000335795836131632
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 68
para_0 = torch.randint(1, 100, [15], dtype=torch.int64)
para_1 = torch.randn([31, 12], dtype=torch.float32)
para_2 = torch.randint(1, 100, [2], dtype=torch.int64)
para_3 = None
para_4 = 2.0
para_5 = False
para_6 = 'sum'
para_7 = False
para_8 = None
para_9 = True
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,para_8,para_9,)
verify_model(embedding_bag().float().eval(), input_data=para_0)
# test_id: 69
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0002936857562408223
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 70
para_0 = torch.randn([2, 3, 20, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 71
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=3,padding=1,dilation=2,).eval(), input_data=[torch.randn([2, 3, 23, 13], dtype=torch.float32)])
# test_id: 72
verify_model(torch.nn.Linear(5,5,).eval(), input_data=[torch.randn([tensor(5)], dtype=torch.float32)])
# test_id: 73
para_0 = torch.randn([1, 1, 5, 8], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=(3, 3),stride=(3, 3),)
verify_model(unfold().float().eval(), input_data=para_0)
# test_id: 74
para_0 = torch.randn([2, 2, 18, 11], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = [0, 0]
para_6 = 1
para_7 = (4, 4)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 75
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.001694915254237288
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 76
para_0 = torch.randn([2, 2, 7, 5], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = [1, 0]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 77
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=2,padding=1,dilation=3,).eval(), input_data=[torch.randn([2, 3, 21, 13], dtype=torch.float32)])
# test_id: 78
para_0 = torch.randn([2, 3, 14, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 79
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00036153289949385393
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 80
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003606202668589975
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 81
para_0 = torch.randn([2, 2, 21, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = [0, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 82
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=3,padding=2,dilation=4,).eval(), input_data=[torch.randn([2, 3, 16, 15], dtype=torch.float32)])
# test_id: 83
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = (2, 2)
para_2 = (2, 2)
para_3 = (1, 1)
para_4 = False
para_5 = True
para_6 = None
class avg_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(avg_pool2d().float().eval(), input_data=para_0)
# test_id: 84
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0001985308715505261
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 85
para_0 = torch.randn([2, 3, 19, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 86
para_0 = torch.randn([2, 3, 18, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (3, 3)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 87
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0008090614886731392
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 88
para_0 = torch.randn([1, 6, 5, 7], dtype=torch.float32)
para_1 = (3, 5, 2)
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,divisor_override=7,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)
# test_id: 89
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00035373187124159886
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 90
verify_model(torch.nn.ReLU().eval(), input_data=[torch.randn([tensor(24), tensor(64), tensor(300)], dtype=torch.float32)])
# test_id: 91
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0001853911753800519
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 92
para_0 = torch.randn([2, 2, 6, 5], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = [2, 1]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 93
para_0 = torch.randn([2, 3, 16, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 94
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0008403361344537816
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 95
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=3,padding=0,dilation=2,).eval(), input_data=[torch.randn([2, 3, 22, 14], dtype=torch.float32)])
# test_id: 96
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.014925373134328358
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 97
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00037979491074819596
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 98
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00019825535289452815
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 99
para_0 = torch.randn([2, 3, 21, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (1, 1)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 100
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0007830853563038371
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 101
para_0 = torch.randn([2, 3, 23, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = (3, 3)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 102
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003466204506065858
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 103
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=3,padding=2,dilation=2,).eval(), input_data=[torch.randn([2, 3, 14, 14], dtype=torch.float32)])
# test_id: 104
para_0 = torch.randn([2, 2, 16, 9], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (0, 0)
para_5 = [0, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 105
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=4,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 2, 22, 14], dtype=torch.float32)])
# test_id: 106
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0006138735420503376
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 107
para_0 = torch.randn([2, 2, 7, 8], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = [1, 1]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 108
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0006341154090044388
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 109
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.000297000297000297
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 110
para_0 = torch.randn([2, 2, 16, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (2, 2)
para_5 = [0, 0]
para_6 = 1
para_7 = (4, 4)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 111
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00034746351633078526
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 112
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=2,padding=1,dilation=4,).eval(), input_data=[torch.randn([2, 3, 15, 14], dtype=torch.float32)])
# test_id: 113
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00017409470752089137
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 114
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0002915451895043732
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 115
para_0 = torch.randn([2, 2, 7, 2], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = [0, 1]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 116
para_0 = torch.randn([2, 3, 15, 14], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = (2, 2)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 117
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=2,stride=1,padding=2,dilation=3,).eval(), input_data=[torch.randn([2, 2, 20, 16], dtype=torch.float32)])
# test_id: 118
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.003703703703703704
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 119
para_0 = torch.randn([3, 5], dtype=torch.float32)
para_1 = torch.randn([3, 5], dtype=torch.float32)
class soft_margin_loss(Module):
    def forward(self, *args):
        return torch.nn.functional.soft_margin_loss(args[0], para_1,)
verify_model(soft_margin_loss().float().eval(), input_data=para_0)
# test_id: 120
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005678591709256105
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 121
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003351206434316354
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 122
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00025713551041398817
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 123
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.007142857142857143
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 124
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=3,padding=0,dilation=3,).eval(), input_data=[torch.randn([2, 3, 21, 13], dtype=torch.float32)])
# test_id: 125
para_0 = torch.randn([2, 2, 3, 3], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = [2, 1]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 126
para_0 = torch.randn([1, 2, 4, 4], dtype=torch.float32)
class interpolate(Module):
    def forward(self, *args):
        return torch.nn.functional.interpolate(args[0], size=12,scale_factor=None,mode='nearest',)
verify_model(interpolate().float().eval(), input_data=para_0)
# test_id: 127
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=1,dilation=2,).eval(), input_data=[torch.randn([2, 3, 18, 14], dtype=torch.float32)])
# test_id: 128
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00037993920972644377
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 129
para_0 = torch.randn([4, 10], dtype=torch.float32)
para_1 = torch.randn([10], dtype=torch.float32)
para_2 = torch.randn([10], dtype=torch.float32)
para_3 = torch.randn([10], dtype=torch.float32)
para_4 = torch.randn([10], dtype=torch.float32)
para_5 = True
para_6 = 0.0027247956403269754
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 130
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005820721769499418
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 131
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00016121231662098983
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 132
para_0 = torch.randn([2, 2, 5, 4], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = [2, 0]
para_6 = 1
para_7 = (4, 4)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 133
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.008695652173913044
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 134
verify_model(torch.nn.Conv2d(80,1,4,1,0,bias=False,).eval(), input_data=[torch.randn([tensor(5), tensor(80), tensor(4), tensor(4)], dtype=torch.float32)])
# test_id: 135
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=4,stride=1,padding=2,dilation=2,).eval(), input_data=[torch.randn([2, 2, 11, 12], dtype=torch.float32)])
# test_id: 136
verify_model(torch.nn.Conv2d(3,3,(1, 1),stride=(1, 1),padding=(0, 0),dilation=(1, 1),groups=1,bias=False,padding_mode='zeros',).eval(), input_data=[torch.randn([10, 3, 15, 15], dtype=torch.float32)])
# test_id: 137
verify_model(torch.nn.ReLU6().eval(), input_data=[torch.randn([20], dtype=torch.float32)])
# test_id: 138
para_0 = torch.randn([2, 2, 4, 4], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (0, 0)
para_5 = [1, 2]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 139
para_0 = torch.randn([2, 3, 18, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 140
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00019331142470520006
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 141
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 3, 13, 16], dtype=torch.float32)])
# test_id: 142
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.013513513513513514
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 143
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005102040816326531
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 144
para_0 = torch.randn([2, 2, 8, 6], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (2, 2)
para_5 = [2, 0]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 145
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=1,padding=0,dilation=2,).eval(), input_data=[torch.randn([2, 3, 22, 15], dtype=torch.float32)])
# test_id: 146
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=3,padding=1,dilation=2,).eval(), input_data=[torch.randn([2, 3, 22, 16], dtype=torch.float32)])
# test_id: 147
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 3, 21, 15], dtype=torch.float32)])
# test_id: 148
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005959475566150178
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 149
para_0 = torch.randn([1, 10], dtype=torch.float32)
para_1 = torch.randn([20, 10], dtype=torch.float32)
para_2 = torch.randn([20], dtype=torch.float32)
class linear(Module):
    def forward(self, *args):
        return torch.nn.functional.linear(args[0], para_1,para_2,)
verify_model(linear().float().eval(), input_data=para_0)
# test_id: 150
para_0 = torch.randn([2, 3, 23, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 151
para_0 = torch.randn([2, 2, 6, 3], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = [1, 0]
para_6 = 1
para_7 = (3, 3)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 152
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,stride=1,padding=1,dilation=1,).eval(), input_data=[torch.randn([2, 3, 22, 16], dtype=torch.float32)])
# test_id: 153
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=3,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 2, 25, 16], dtype=torch.float32)])
# test_id: 154
para_0 = torch.randn([2, 2, 19, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (1, 1)
para_4 = (1, 1)
para_5 = [0, 0]
para_6 = 1
para_7 = (1, 1)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 155
para_0 = torch.randn([9, 7, 7, 8], dtype=torch.float32)
para_1 = 3
para_2 = 2
para_3 = 1
para_4 = 1
para_5 = False
para_6 = False
class max_pool2d(Module):
    def forward(self, *args):
        return torch.nn.functional.max_pool2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(max_pool2d().float().eval(), input_data=para_0)
# test_id: 156
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.012048192771084338
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 157
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=4,stride=2,padding=2,dilation=4,).eval(), input_data=[torch.randn([2, 3, 23, 16], dtype=torch.float32)])
# test_id: 158
para_0 = torch.randn([1, 1, 5, 7], dtype=torch.float32)
class unfold(Module):
    def forward(self, *args):
        return torch.nn.functional.unfold(args[0], kernel_size=(3, 2),stride=(3, 2),)
verify_model(unfold().float().eval(), input_data=para_0)
# test_id: 159
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0029069767441860465
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 160
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0003672420124862284
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 161
para_0 = torch.randn([2, 2, 9, 6], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (0, 0)
para_5 = [0, 1]
para_6 = 1
para_7 = (4, 4)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 162
para_0 = torch.randn([1, 6, 5, 7], dtype=torch.float32)
para_1 = (6, 3, 3)
class avg_pool3d(Module):
    def forward(self, *args):
        return torch.nn.functional.avg_pool3d(args[0], para_1,divisor_override=18,)
verify_model(avg_pool3d().float().eval(), input_data=para_0)
# test_id: 163
para_0 = torch.randn([2, 2, 9, 8], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = [0, 0]
para_6 = 1
para_7 = (1, 1)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 164
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0006184291898577613
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 165
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=4,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 2, 19, 15], dtype=torch.float32)])
# test_id: 166
para_0 = torch.randint(1, 100, [2, 3], dtype=torch.int64)
para_1 = torch.randn([4, 3], dtype=torch.float32)
para_2 = None
para_3 = None
para_4 = 2.0
para_5 = False
para_6 = 'max'
para_7 = False
para_8 = None
para_9 = False
class embedding_bag(Module):
    def forward(self, *args):
        return torch.nn.functional.embedding_bag(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,para_8,para_9,)
verify_model(embedding_bag().float().eval(), input_data=para_0)
# test_id: 167
para_0 = torch.randn([2, 2, 12, 8], dtype=torch.float32)
para_1 = torch.randn([2, 3, 2, 2], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = [0, 1]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 168
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0004940711462450593
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 169
para_0 = torch.randn([2, 2, 7, 6], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (1, 1)
para_5 = [1, 1]
para_6 = 1
para_7 = (2, 2)
class conv_transpose2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv_transpose2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(conv_transpose2d().float().eval(), input_data=para_0)
# test_id: 170
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=3,padding=2,dilation=4,).eval(), input_data=[torch.randn([2, 3, 22, 13], dtype=torch.float32)])
# test_id: 171
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=1,dilation=1,).eval(), input_data=[torch.randn([2, 3, 15, 14], dtype=torch.float32)])
# test_id: 172
para_0 = torch.randn([tensor(5), tensor(20), tensor(4), tensor(4)], dtype=torch.float32)
class relu(Module):
    def forward(self, *args):
        return torch.nn.functional.relu(args[0], )
verify_model(relu().float().eval(), input_data=para_0)
# test_id: 173
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.00016663889351774705
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 174
para_0 = torch.randn([1, 10], dtype=torch.float32)
para_1 = torch.randint(1, 100, [1], dtype=torch.int64)
class cross_entropy(Module):
    def forward(self, *args):
        return torch.nn.functional.cross_entropy(args[0], para_1,weight=None,ignore_index=-100,reduction='mean',)
verify_model(cross_entropy().float().eval(), input_data=para_0)
# test_id: 175
para_0 = torch.randn([2, 3, 6, 6], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.0005238344683080147
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 176
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=4,stride=1,padding=2,dilation=1,).eval(), input_data=[torch.randn([2, 2, 16, 16], dtype=torch.float32)])
# test_id: 177
para_0 = torch.randn([2, 3, 18, 13], dtype=torch.float32)
para_1 = torch.randn([2, 3, 3, 3], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (0, 0)
para_5 = (4, 4)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 178
verify_model(torch.nn.InstanceNorm2d(8,1e-05,False,False,False,).eval(), input_data=[torch.randn([2, 8, 1, 1], dtype=torch.quint8)])
# test_id: 179
verify_model(torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=2,stride=1,padding=1,dilation=4,).eval(), input_data=[torch.randn([2, 3, 18, 14], dtype=torch.float32)])
# test_id: 180
para_0 = torch.randn([tensor(1), tensor(2), tensor(4), tensor(4)], dtype=torch.float32)
para_1 = 0.0
para_2 = 6.0
para_3 = True
class hardtanh(Module):
    def forward(self, *args):
        return torch.nn.functional.hardtanh(args[0], para_1,para_2,para_3,)
verify_model(hardtanh().float().eval(), input_data=para_0)
# test_id: 181
para_0 = torch.randn([2, 3, 15, 15], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (2, 2)
para_4 = (2, 2)
para_5 = (1, 1)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
# test_id: 182
para_0 = torch.randn([2, 3, 4, 4, 4], dtype=torch.float32)
para_1 = torch.randn([3], dtype=torch.float32)
para_2 = torch.randn([3], dtype=torch.float32)
para_3 = torch.randn([3], dtype=torch.float32)
para_4 = torch.randn([3], dtype=torch.float32)
para_5 = True
para_6 = 0.023809523809523808
para_7 = 0.001
class batch_norm(Module):
    def forward(self, *args):
        return torch.nn.functional.batch_norm(args[0], para_1,para_2,para_3,para_4,para_5,para_6,para_7,)
verify_model(batch_norm().float().eval(), input_data=para_0)
# test_id: 183
verify_model(torch.nn.ConvTranspose2d(in_channels=2,out_channels=3,kernel_size=3,stride=1,padding=1,dilation=3,).eval(), input_data=[torch.randn([2, 2, 19, 11], dtype=torch.float32)])
# test_id: 184
para_0 = torch.randn([2, 3, 20, 16], dtype=torch.float32)
para_1 = torch.randn([2, 3, 4, 4], dtype=torch.float32)
para_2 = torch.randn([2], dtype=torch.float32)
para_3 = (3, 3)
para_4 = (1, 1)
para_5 = (3, 3)
para_6 = 1
class conv2d(Module):
    def forward(self, *args):
        return torch.nn.functional.conv2d(args[0], para_1,para_2,para_3,para_4,para_5,para_6,)
verify_model(conv2d().float().eval(), input_data=para_0)
