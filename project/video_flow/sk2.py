import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
import pdb

class Aggregate(nn.Module):
    def __init__(self, dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class PCBlock4_Deep_nopool_res(nn.Module):
    def __init__(self, C_in, C_out, k_conv=[1, 15]):
        super().__init__()
        self.conv_list = nn.ModuleList([
            nn.Conv2d(C_in, C_in, kernel, stride=1, padding=kernel//2, groups=C_in) for kernel in k_conv])

        self.ffn1 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_in, 1, padding=0),
        )
        self.pw = nn.Conv2d(C_in, C_in, 1, padding=0)
        self.ffn2 = nn.Sequential(
            nn.Conv2d(C_in, int(1.5*C_in), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(int(1.5*C_in), C_out, 1, padding=0),
        )

    def forward(self, x):
        x = F.gelu(x + self.ffn1(x))
        for conv in self.conv_list:
            x = F.gelu(x + conv(x))
        x = F.gelu(x + self.pw(x))
        x = self.ffn2(x)
        return x


class SKMotionEncoder6_Deep_nopool_res(nn.Module):
    def __init__(self,  
            corr_radius=4, 
            corr_levels=4,
            k_conv=[1, 15],
        ):
        super().__init__()
        self.cost_heads_num = 1
        self.k_conv = k_conv
        self.cor_planes = cor_planes = (corr_radius*2+1)**2 * self.cost_heads_num * corr_levels
        self.convc1 = PCBlock4_Deep_nopool_res(cor_planes, 128, k_conv=self.k_conv)
        self.convc2 = PCBlock4_Deep_nopool_res(256, 192, k_conv=self.k_conv)

        self.convf1_ = nn.Conv2d(4, 128, 1, 1, 0)
        self.convf2 = PCBlock4_Deep_nopool_res(128, 64, k_conv=self.k_conv)

        self.conv = PCBlock4_Deep_nopool_res(64+192, 128-4, k_conv=self.k_conv)


    def forward(self, flow, corr):
        corr1, corr2 = torch.split(corr, [self.cor_planes, self.cor_planes], dim=1)
        cor = F.gelu(torch.cat([self.convc1(corr1), self.convc1(corr2)], dim=1))

        cor = self.convc2(cor)

        flo = self.convf1_(flow)
        flo = self.convf2(flo)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo)

        return torch.cat([out, flow], dim=1)


class SKUpdateBlock6_Deep_nopoolres_AllDecoder2(nn.Module):
    def __init__(self, hidden_dim=128, corr_radius=4, corr_levels=4, k_conv = [1, 15]):
        super().__init__()
        self.PCUpdater_conv = [1, 7]
        self.encoder = SKMotionEncoder6_Deep_nopool_res(corr_radius=corr_radius, corr_levels=corr_levels)
        self.gru = PCBlock4_Deep_nopool_res(128+hidden_dim+hidden_dim+128, 128, k_conv=self.PCUpdater_conv)
        self.flow_head = PCBlock4_Deep_nopool_res(128, 4, k_conv=k_conv)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9*2, 1, padding=0))

        self.aggregator = Aggregate(dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(torch.cat([net, inp_cat], dim=1))

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow 
