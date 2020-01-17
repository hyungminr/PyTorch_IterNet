import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2L(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=3, stride=1, padding=1, bias=True):
        super(conv2L, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=dim_out, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        layers += [nn.ReLU(inplace=True)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class conv1L(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=1, stride=1, padding=0, bias=True, activation='sigmoid'):
        super(conv1L, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out)]
        if activation == 'sigmoid':
            layers += [nn.Sigmoid()]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class maxPool(nn.Module):
    def __init__(self, kernel=2, stride=2, padding=0):
        super(maxPool, self).__init__()
        layers = []
        layers += [nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class convUp(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=3, stride=1, padding=1, bias=True):
        super(convUp, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out*2, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out*2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=dim_out*2, out_channels=dim_out*2, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=dim_out*2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.ConvTranspose2d(in_channels=dim_out*2, out_channels=dim_out, kernel_size=2, stride=2)]
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class IterNetModule(nn.Module):
    def __init__(self, n):
        super(IterNetModule, self).__init__()
        self.pool = maxPool()
        self.cd = conv1L(dim_in= 32*n, dim_out= 32, kernel=1, stride=1, padding=0, bias=True, activation='relu')
        self.c0 = conv2L(dim_in= 32, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.c1 = conv2L(dim_in= 32, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.c2 = conv2L(dim_in= 64, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.c3 = convUp(dim_in=128, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.c4 = convUp(dim_in=256, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.c5 = convUp(dim_in=128, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.c6 = conv2L(dim_in= 64, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.co = conv1L(dim_in= 32, dim_out=  1, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, x0, x6, x9s=None):
        x9 = self.c0(x6)
        if x9s is not None:
            x9_ = torch.cat((x9s, x9), dim=1)
            x9_ = torch.cat((x9_, x0), dim=1)
            x9s = torch.cat((x9, x9s), dim=1)
        else:
            x9s = x9
            x9_ = torch.cat((x9, x0), dim=1)
        x0c = self.cd(x9_)
        x1p = self.pool(x0c)
        x1c = self.c1(x1p)
        x2p = self.pool(x1c)
        x2c = self.c2(x2p)
        x3p = self.pool(x2c)
        x3 = self.c3(x3p)
        x3 = torch.cat((x3, x2c), dim=1)
        x4 = self.c4(x3)
        x4 = torch.cat((x4, x1c), dim=1)
        x5 = self.c5(x4)
        x5 = torch.cat((x5, x0c), dim=1)
        x6 = self.c6(x5)
        xout = self.co(x6)
        return xout, x6, x9s
        
class IterNetInit(nn.Module):
    def __init__(self):
        super(IterNetInit, self).__init__()
        self.block1_pool = maxPool()
        self.block1_c1 = conv2L(dim_in=  3, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c2 = conv2L(dim_in= 32, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c3 = conv2L(dim_in= 64, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c4 = conv2L(dim_in=128, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c5 = convUp(dim_in=256, dim_out=256, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c6 = convUp(dim_in=512, dim_out=128, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c7 = convUp(dim_in=256, dim_out= 64, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c8 = convUp(dim_in=128, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_c9 = conv2L(dim_in= 64, dim_out= 32, kernel=3, stride=1, padding=1, bias=True)
        self.block1_co = conv1L(dim_in= 32, dim_out=  1, kernel=1, stride=1, padding=0, bias=True)
        
    def forward(self, img):
        x1c = self.block1_c1(img)
        x1p = self.block1_pool(x1c)
        x2c = self.block1_c2(x1p)
        x2p = self.block1_pool(x2c)
        x3c = self.block1_c3(x2p)
        x3p = self.block1_pool(x3c)
        x4c = self.block1_c4(x3p)
        x4p = self.block1_pool(x4c)
        
        x5 = self.block1_c5(x4p)
        x5 = torch.cat((x5, x4c), dim=1)
        x6 = self.block1_c6(x5)
        x6 = torch.cat((x6, x3c), dim=1)
        x7 = self.block1_c7(x6)
        x7 = torch.cat((x7, x2c), dim=1)
        x8 = self.block1_c8(x7)
        x8 = torch.cat((x8, x1c), dim=1)
        x9 = self.block1_c9(x8)
        xout = self.block1_co(x9)
        
        return xout, x1c, x9
    

class IterNet(nn.Module):
    def __init__(self):
        super(IterNet, self).__init__()
        self.net1 = IterNetInit()
        self.net2 = IterNetModule(2)
        self.net3 = IterNetModule(3)
        self.net4 = IterNetModule(4)
    def forward(self, img):
        out1, x0, x9 = self.net1(img)
        out2, x6, x9 = self.net2(x0, x9)
        out3, x6, x9 = self.net3(x0, x6, x9)
        out4, x6, x9 = self.net4(x0, x6, x9)
        return out1, out2, out3, out4