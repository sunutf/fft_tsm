import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

def make_channel_dctidct(net, n_segments):
    import torchvision
    import archs
    res50_spatial_feat_dim = {
        "layer2" : 28,
        "layer3" : 14,
        "layer4" : 7
    }
    
    if isinstance(net, torchvision.models.ResNet):
        '''
        net.layer2 = nn.Sequential(
            net.layer2[0],
            net.layer2[1],
            net.layer2[2],
            cDCTiDCTWrapper3D(net.layer2[3], n_segments, res50_spatial_feat_dim["layer2"]),
           ) 
        '''
        net.layer3 = nn.Sequential(
            net.layer3[0],
            net.layer3[1],
            net.layer3[2],
            net.layer3[3],
            net.layer3[4],
            cDCTiDCTWrapper3D(net.layer3[5], n_segments, res50_spatial_feat_dim["layer3"]),
           ) 

        net.layer4 = nn.Sequential(
            net.layer4[0],
            net.layer4[1],
            cDCTiDCTWrapper3D(net.layer4[2], n_segments, res50_spatial_feat_dim["layer4"]),
           )
    else:
        raise NotImplementedError 


class cDCTiDCTWrapper3D(nn.Module):
    def __init__(self, block, n_segments, spatial_dim):
        super(cDCTiDCTWrapper3D, self).__init__()
    
        self.block = block
        self.num_segments = n_segments
        self.dct_c = LinearDCT(block.bn3.num_features, "dct", norm='ortho')    
        #self.dct_h = LinearDCT(spatial_dim, "dct", norm='ortho')    
        #self.dct_w = LinearDCT(spatial_dim, "dct", norm='ortho')    
        #self.dct_t = LinearDCT(n_segments, "dct", norm='ortho')    
        self.idct_c = LinearDCT(block.bn3.num_features, "idct", norm='ortho')    
        #self.idct_h = LinearDCT(spatial_dim, "idct", norm='ortho')    
        #self.idct_w = LinearDCT(spatial_dim, "idct", norm='ortho')    
        #self.idct_t = LinearDCT(n_segments, "idct", norm='ortho')
        
        self.what_to_mask = n_segments*spatial_dim*spatial_dim
        self.mask_c = nn.Linear(self.what_to_mask, 1)
        #self.hard_mask_thw = nn.Conv3d(block.bn3.num_features, 2, 1)

        self.enhance_c = nn.Sequential(
                #nn.Conv3d(block.bn3.num_features, 1, 1),
                nn.Linear(self.what_to_mask, self.what_to_mask),
                #nn.ReLU()
                nn.Tanh()
                )
        
    def low_pass(self, x):
        _b, _t, _c, _h, _w = x.shape
        x[:,:_t//4,:_c//4,:_h//4,:_w//4] = 0 
        return x  

    def adaptive_pass(self, x):
        _b, _t, _c, _h, _w = x.shape
        mask_x = self.mask_thw(x.permute(0,2,1,3,4)) #B,T,C,H,W => B,1,T,H,W
        
        mask_x = mask_x.view(_b, 1, _t*_h*_w).permute(0,2,1)

        mask_x = F.softmax(mask_x, dim=1)
        mask_x = mask_x.view(_b, _t, _h, _w, 1)
        mask_x = mask_x.permute(0,1,4,2,3)
        mask_x = mask_x.expand(-1, -1, _c, -1, -1)

        return mask_x * x
        
        #p_t = torch.log(F.softmax(mask_x, dim=-1).clamp(min=1e-8))
        #r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])]) 
    
    def sigmoid_pass(self, x):
        _b, _t, _c, _h, _w = x.shape
        mask_x = self.mask_c(x.permute(0,2,1,3,4).reshape(_b,_c,-1)) #B,T,C,H,W => B,C,1
        
        #mask_x = mask_x.view(_b, 1, _t*_h*_w).permute(0,2,1)
        mask_x = nn.Sigmoid()(mask_x)
        #mask_x = mask_x.view(_b, _t, _h, _w, 1)
        #mask_x = mask_x.permute(0,1,4,2,3)
        mask_x = mask_x.expand(-1, -1, _t*_h*_w) #=>B,C,THW
        mask_x = mask_x.view(_b, _c, _t, _h, _w) #=>B,C,T,H,W
        
        mask_x = mask_x.permute(0,2,1,3,4) #=>B,T,C,H,W
        
        return mask_x * x
    '''
    def hard_adaptive_pass(self, x, tau):
        _b, _t, _c, _h, _w = x.shape
        mask_x = self.hard_mask_thw(x.permute(0,2,1,3,4)) #B,T,C,H,W => B,2,T,H,W
        
        mask_x = mask_x.view(_b, 2, _t*_h*_w).permute(0,2,1) #B,2,THW => B,THW,2
        
        p_t = torch.log(F.softmax(mask_x, dim=-1).clamp(min=1e-8))
        r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], tau, True) for b_i in range(p_t.shape[0])]) 
        
        r_t = r_t.view(_b, _t, _h, _w, 2)[...,1].unsqueeze(-1)

        mask_x = r_t.permute(0,1,4,2,3)
        mask_x = mask_x.expand(-1, -1, _c, -1, -1)

        return mask_x * x
    '''

    def enhancement(self, x):
        _b, _t, _c, _h, _w = x.shape
        enh_x = self.enhance_c(x.permute(0,2,1,3,4).reshape(_b,_c,-1)) #B,T,C,H,W => B,C,THW
        enh_x = enh_x.view(_b,_c,_t,_h,_w).permute(0,2,1,3,4)
        return enh_x
        #enh_x = enh_x.expand(-1, -1, _c, -1, -1)
        
        #return enh_x * x

    def forward(self, x):
        _bt, _c, _h, _w = x.shape
        x = self.block(x)
        x = x.view(_bt//self.num_segments, self.num_segments, _c, _h, _w) #B,T,C,H,W
        
        dct_x = apply_linear_4d_woTHW(x, self.dct_c) #B,T,C,H,W
        dct_x = self.sigmoid_pass(dct_x)
        dct_x = apply_linear_4d_woTHW(dct_x, self.idct_c)
        dct_x = self.enhancement(dct_x)

        return (x + dct_x).reshape(_bt, _c, _h, _w)
        

class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!
    

def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)

def apply_linear_4d_woTHW(x, c_FC):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X = c_FC(x.transpose(-1, -3))
    return X.transpose(-1, -3)

def apply_linear_4d_woC(x, t_FC, h_FC, w_FC):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = w_FC(x)
    X2 = h_FC(X1.transpose(-1, -2))
    #X3 = c_FC(X2.transpose(-1, -3))
    X4 = t_FC(X2.transpose(-1, -4))
    #return X4.transpose(-1, -4).transpose(-1, -3).transpose(-1,-2)
    return X4.transpose(-1, -4).transpose(-1,-2)

def apply_linear_4d(x, t_FC, c_FC, h_FC, w_FC):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = w_FC(x)
    X2 = h_FC(X1.transpose(-1, -2))
    X3 = c_FC(X2.transpose(-1, -3))
    X4 = t_FC(X3.transpose(-1, -4))
    return X4.transpose(-1, -4).transpose(-1, -3).transpose(-1,-2)
    #return X4.transpose(-1, -4).transpose(-1,-2)

def t_apply_linear_4d(x, c_FC, h_FC, w_FC):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = w_FC(x)
    X2 = h_FC(X1.transpose(-1, -2))
    X3 = c_FC(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1,-2)
    #return X4.transpose(-1, -4).transpose(-1,-2)

class test_DCTiDCTWrapper3D(nn.Module):
    def __init__(self, spatial_dim):
        super(test_DCTiDCTWrapper3D, self).__init__()

        self.dct_c = LinearDCT(3, "dct", norm='ortho')
        self.dct_h = LinearDCT(spatial_dim, "dct", norm='ortho')
        self.dct_w = LinearDCT(spatial_dim, "dct", norm='ortho')
        self.idct_c = LinearDCT(3, "idct", norm='ortho')
        self.idct_h = LinearDCT(spatial_dim, "idct", norm='ortho')
        self.idct_w = LinearDCT(spatial_dim, "idct", norm='ortho')


    def low_pass(self, x):
        _b, _c, _h, _w = x.shape
        x[:, :_c//7,:_h//7,:_w//7] = 0
        return x


    def forward(self, x):

        dct_x = t_apply_linear_4d(x, self.dct_c, self.dct_h, self.dct_w)
        dct_x = self.low_pass(dct_x)
        dct_x = t_apply_linear_4d(dct_x, self.idct_c, self.idct_h, self.idct_w)

        return x + dct_x

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import PIL
    import torchvision.transforms as transforms
    import torch
    from torchvision.transforms.functional import to_pil_image
    trans = transforms.Compose([transforms.Resize((320, 320)),
                            transforms.ToTensor()
                            ])
    img = PIL.Image.open('../bird.jpg').convert('RGB')
    img_t = trans(img)
    img_t = img_t.unsqueeze(0) * 255.0

    print(img_t[0].shape)
    #plt.imsave("ori.png", transforms.ToPILImage()(img_t[0]).convert("RGB"))
    plt.imsave("ori.png", img.convert("RGB"))

    dct_filter = test_DCTiDCTWrapper3D(320)
    dct_img_t = dct_filter(img_t)
    plt.imsave("dct.jpg", transforms.ToPILImage()(dct_img_t[0]).convert("RGB"))
    '''

    x = torch.Tensor(1000,4096)
    x.normal_(0,1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())
    '''

