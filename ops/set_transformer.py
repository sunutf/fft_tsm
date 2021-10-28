from ops.set_transformer_modules import *

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output)
                )

    def forward(self, X):
        return self.dec(self.enc(X))

class SetTransformerWrapper(nn.Module):
    def __init__(self, block, n_segment):
        super(SetTransformerWrapper, self).__init__()
        self.block = block
        self.set_transformer = SetTransformer(block.bn3.num_features, 1, block.bn3.num_features, num_inds=n_segment, dim_hidden=block.bn3.num_features//2)
        self.n_segment = n_segment
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.block(x)

        nt, c, h, w = x.size()
        x = self.avgpool(x).squeeze(-1).squeeze(-1) #nt, c 
        x = x.view(nt // self.n_segment, self.n_segment, c).transpose(1, 2)  # n, c, t

        x = self.set_transformer(x.transpose(1,2)) #n, 1, c
        return x
        
    
    
def make_set_transformer(net, n_segment):
    import torchvision
    import archs
    if isinstance(net, torchvision.models.ResNet):
        net.layer4 = nn.Sequential(
            net.layer4[0],
            net.layer4[1],
            SetTransformerWrapper(net.layer4[2], n_segment),
        )

    else:
        raise NotImplementedError

