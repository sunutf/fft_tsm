import torch
import torch.nn as nn
import torch.nn.functional as F

def make_pool_n_flatten(pool_type, net, channel_dim):
    import torchvision
    import archs

    if isinstance(net, torchvision.models.ResNet):
        if pool_type == "avg":
            net.avgpool = AvgPoolandFlatten(channel_dim)
        elif pool_type == "max":
            net.avgpool = MaxPoolandFlatten(channel_dim)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

### only to C,H,W @here
class AvgPoolandFlatten(nn.Module):
    def __init__(self, channel_dim):
        super(AvgPoolandFlatten, self).__init__()
        
        self.avgpoolHW = nn.AdaptiveAvgPool2d(4)
        #self.avgpoolC = nn.AdaptiveAvgPool1d(channel_dim//16)
        self.fcC = nn.Conv2d(channel_dim, channel_dim//16, 1)

    def forward(self, x):
        _bt, _c, _h, _w = x.shape
        x = self.avgpoolHW(x) #B*T,C,H,W => B*T,C,4,4
        x = self.fcC(x).view(-1,_c//16, 16).permute(0,2,1)  # => B*T,C//16,4,4 => B*T,16,C//16
        #x = self.avgpoolC(x.view(-1, _c, 16).permute(0,2,1)) #B*T,C,4,4 => B*T,16,C => B*T,16,C//16
        x = torch.flatten(x, start_dim=1) #B*T, C

        return x

class MaxPoolandFlatten(nn.Module):
    def __init__(self, channel_dim):
        super(MaxPoolandFlatten, self).__init__()
        
        self.maxpoolHW = nn.AdaptiveMaxPool2d(4)
        self.maxpoolC = nn.AdaptiveMaxPool1d(channel_dim//16)

    def forward(self, x):
        _bt, _c, _h, _w = x.shape
        x = self.maxpoolHW(x) #B*T,C,H,W => B*T,C,2,2
        x = self.maxpoolC(x.view(-1, _c, 16).permute(0,2,1)) #B*T,C,2,2 => B*T,4,C => B*T,4,C//4
        x = torch.flatten(x, start_dim=1) #B*T, C

        return x
