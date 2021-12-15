# Non-local block using embedded gaussian
# Code from
# https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.3.1/lib/non_local_embedded_gaussian.py
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels_q=None, inter_channels_kv=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels_q = inter_channels_q
        self.inter_channels_kv = inter_channels_kv

        if self.inter_channels_q is None:
            self.inter_channels_q = in_channels // 2
            if self.inter_channels_q == 0:
                self.inter_channels_q = 1
        
        if self.inter_channels_kv is None:
            self.inter_channels_kv = in_channels // 2
            if self.inter_channels_kv == 0:
                self.inter_channels_kv = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_kv,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels_q, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels_q, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_q,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels_kv,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels_kv, -1)
        g_x = g_x.permute(0, 2, 1)#B,C,HW => B,HW,C''

        theta_x = self.theta(x).view(batch_size, self.inter_channels_q, -1)#B,C,HW => B,C',HW
        #theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels_kv, -1)
        phi_x = phi_x.permute(0,2,1) #B,C,HW => B,HW,C''
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        f_dif_C = f_div_C.permute(0,2,1) #B,C'',C'

        #y = torch.matmul(f_div_C, g_x)
        y = torch.matmul(g_x, f_dif_C) #B,HW,C'' x B,C'',C' => B,HW,C'
        y = y.permute(0, 2, 1).contiguous() # => B,C',HW 
        y = y.view(batch_size, self.inter_channels_kv, *x.size()[2:])
        W_y = self.W(y) #=>B,C,H,W
        z = W_y + x

        return z


class CNONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels_q=None, inter_channels_kv=None, sub_sample=True, bn_layer=True):
        super(CNONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels_q = inter_channels_q,
                                              inter_channels_kv = inter_channels_kv,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class CNONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels_q=None, inter_channels_kv=None, sub_sample=True, bn_layer=True):
        super(CNONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels_q = inter_channels_q,
                                              inter_channels_kv = inter_channels_kv,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class CNONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels_q=None, inter_channels_kv=None, sub_sample=False, bn_layer=True):
        super(CNONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels_q = inter_channels_q,
                                              inter_channels_kv = inter_channels_kv,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class CNL3DWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(CNL3DWrapper, self).__init__()
        self.block = block
        self.cnl = CNONLocalBlock3D(block.bn3.num_features)
        self.n_segment = n_segment

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = self.cnl(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x


def make_c_non_local(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet):
        '''
        net.layer2 = nn.Sequential(
            CNL3DWrapper(net.layer2[0], n_segment),
            net.layer2[1],
            CNL3DWrapper(net.layer2[2], n_segment),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            CNL3DWrapper(net.layer3[0], n_segment),
            net.layer3[1],
            CNL3DWrapper(net.layer3[2], n_segment),
            net.layer3[3],
            CNL3DWrapper(net.layer3[4], n_segment),
            net.layer3[5],
        )
        '''
        net.layer4 = nn.Sequential(
            CNL3DWrapper(net.layer4[0], n_segment),
            net.layer4[1],
            CNL3DWrapper(net.layer4[2], n_segment),
        )

    else:
        raise NotImplementedError


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch

    sub_sample = True
    bn_layer = True

    img = Variable(torch.zeros(2, 3, 20))
    net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.zeros(2, 3, 20, 20))
    net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())

    img = Variable(torch.randn(2, 3, 10, 20, 20))
    net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
    out = net(img)
    print(out.size())
