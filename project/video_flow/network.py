import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from .encoders import twins_svt_large
from .sk2 import SKUpdateBlock6_Deep_nopoolres_AllDecoder2
import pdb

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class RelPosEmb(nn.Module):
    def __init__(self, max_pos_size, dim_head):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(self, dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn

class BOFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_H = 2048
        self.MAX_W = 4096
        self.MAX_TIMES = 8
        # GPU -- ?

        self.hidden_dim = 128
        self.context_dim = 128
        self.corr_radius = 4
        self.corr_levels = 4
        self.decoder_depth = 32

        self.cnet = twins_svt_large(pretrained=False)
        self.fnet = twins_svt_large(pretrained=False)
        self.att = Attention(dim=128, heads=1, max_pos_size=160, dim_head=128)
        self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2(
            hidden_dim=self.hidden_dim, 
            corr_radius=self.corr_radius,
            corr_levels=self.corr_levels,
        )

        # self.load_weights(model_path="models/BOF_things.pth")
        # self.load_weights(model_path="models/BOF_things_288960noise.pth")
        self.load_weights()

    def load_weights(self, model_path="models/video_flow.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        new_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "")
            new_sd[k] = v
        self.load_state_dict(new_sd)


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        B, C, H, W = img.shape
        # coords0 = coords_grid(B, H // 8, W // 8).to(img.device)
        # coords1 = coords_grid(B, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        # return coords0, coords1
        return coords_grid(B, H // 8, W // 8).to(img.device)

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, images, flow_init=None):
        images = images.unsqueeze(0) # meet [T, N, _, H, W]
        T, B, C, H, W = images.size()
        images = 2 * images - 1.0

        fmaps = self.fnet(images.reshape(T*B, 3, H, W)).reshape(T, B, -1, H//8, W//8)

        fmap1 = fmaps[:, 0, ...]
        fmap2 = fmaps[:, 1, ...]
        fmap3 = fmaps[:, 2, ...]
        
        corr_fn_21 = CorrBlock(fmap2, fmap1, num_levels=self.corr_levels, radius=self.corr_radius)
        corr_fn_23 = CorrBlock(fmap2, fmap3, num_levels=self.corr_levels, radius=self.corr_radius)

        cnet = self.cnet(images[:, 1, ...]) # image2, image1/image3 is useless via RAFT paper

        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        attention = self.att(inp)

        # coords0_21, coords1_21 = self.initialize_flow(images[:, 0, ...])
        # coords0_23, coords1_23 = self.initialize_flow(images[:, 0, ...])

        coords0_21 = self.initialize_flow(images[:, 0, ...])
        coords1_21 = coords0_21.clone()
        coords0_23 = coords0_21.clone()
        coords1_23 = coords0_21.clone()

        flow_predict = torch.zeros(2, 2, H, W).to(images.device)
        for itr in range(self.decoder_depth):
            coords1_21 = coords1_21.detach()
            coords1_23 = coords1_23.detach()
            
            corr21 = corr_fn_21(coords1_21)
            corr23 = corr_fn_23(coords1_23)
            corr =  torch.cat([corr23, corr21], dim=1)
            
            flow21 = coords1_21 - coords0_21
            flow23 = coords1_23 - coords0_23
            flow = torch.cat([flow23, flow21], dim=1)
            
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            up_mask_21, up_mask_23 = torch.split(up_mask, [64*9, 64*9], dim=1)

            coords1_23 = coords1_23 + delta_flow[:, 0:2, ...]
            coords1_21 = coords1_21 + delta_flow[:, 2:4, ...]

            # upsample predictions
            flow_up_23 = self.upsample_flow(coords1_23 - coords0_23, up_mask_23)
            flow_up_21 = self.upsample_flow(coords1_21 - coords0_21, up_mask_21)

            flow_predict = torch.cat([flow_up_23, flow_up_21], dim=0)

        return flow_predict
