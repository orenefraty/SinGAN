import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class mySequential(nn.Sequential):
    def forward(self, x, style_idx):
        for module in self._modules.values():
            x = module(x, style_idx)
        return x

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, num_images=2):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        ## TODO:
        # we will start by replacing only the following batchnorm with style specific instance norm, as in the fried code
        # we need a style index as input to this block only
        # note: there are two additional convolutions in the code that are not followed by batch norm. we can experiment
        # with adding conditioned instance norm after them as well, but best to start without.
        #self.add_module('norm', nn.InstanceNorm2d(out_channel, affine=True)), # for single image instance norm
        self.add_module('norm',ConditionalInstanceNorm2d(out_channel, num_images, affine=True)), # for several image instance norm
        #self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))


    def forward(self, input: torch.Tensor, style_idx):
        x = self.conv(input)
        x = self.norm(x, style_idx)
        x = self.LeakyRelu(x)
        return x


class ConditionalInstanceNorm2d(torch.nn.Module):

    def __init__(self, num_channels, num_styles, affine=True):
        super(ConditionalInstanceNorm2d, self).__init__()
        # Create one norm 2d for each style
        self.norm2ds = torch.nn.ModuleList([torch.nn.InstanceNorm2d(num_channels, affine=affine)
                                            for _ in range(num_styles)])

    def forward(self, x, style_idx):
        return self.norm2ds[style_idx](x)

def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    #elif classname.find('Norm') != -1:
    elif classname == 'InstanceNorm2d':
        #for i in m.norm2ds:
         #   i.weight.data.normal_(1.0, 0.02)
         #   i.bias.data.fill_(0)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        #self.body = nn.Sequential()
        self.body = mySequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x,style_idx):
        x = self.head(x,style_idx)
        x = self.body(x,style_idx)
        x = self.tail(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        #self.body = nn.Sequential()
        self.body = mySequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y,style_idx):
        x = self.head(x,style_idx)
        x = self.body(x,style_idx)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y
