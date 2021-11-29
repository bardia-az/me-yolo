# supplementary DNN modules

from models.common import *



class Encoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        self.conv1 = nn.Conv2d(chs[0], chs[1], k, s, autopad(k, p), bias=False)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(chs[1], chs[2], k, s, autopad(k, p), bias=False)
        # self.act2 = nn.Sigmoid()
        self.act2 = nn.SiLU()
        

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))
        # return self.act2(self.conv2(self.conv1(x)))


class Decoder(nn.Module):
    def __init__(self, chs, k=1, s=1, p=None):
        super().__init__()
        self.conv1 = nn.Conv2d(chs[2], chs[1], k, s, autopad(k, p), bias=False)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(chs[1], chs[0], k, s, autopad(k, p), bias=False)
        self.act2 = nn.SiLU()

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))


class AutoEncoder(nn.Module):
    # def __init__(self, cin, cmid):
    def __init__(self, chs):
        super().__init__()
        # print(chs)
        self.enc = Encoder(chs, k=3)
        self.dec = Decoder(chs, k=3)

    def forward(self, x):
        return self.dec(self.enc(x))



# class Encoder(nn.Module):
#     def __init__(self, cin, cout, k=1, s=1, p=None):
#         super().__init__()
#         self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), bias=False)
#         self.act = nn.SiLU()
        

#     def forward(self, x):
#         return self.act(self.conv(x))


# class Decoder(nn.Module):
#     def __init__(self, cin, cout, k=1, s=1, p=None):
#         super().__init__()
#         self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), bias=False)
#         self.act = nn.SiLU()

#     def forward(self, x):
#         return self.act(self.conv(x))


# class AutoEncoder(nn.Module):
#     # def __init__(self, cin, cmid):
#     def __init__(self, chs):
#         super().__init__()
#         # print(chs)
#         # self.enc = Encoder(cin=cin, cout=cmid, k=1)
#         # self.dec = Decoder(cin=cmid, cout=cin, k=1)
#         self.enc = Encoder(cin=chs[0], cout=chs[1], k=1)
#         self.dec = Decoder(cin=chs[1], cout=chs[0], k=1)

#     def forward(self, x):
#         return self.dec(self.enc(x))

