import torch
import torch.nn as nn
from Ablation.Mamba1 import VSSBlock
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from Ablation.Separable_convolution import S_conv
from thop import profile
import torch.nn.functional as F

class LKA1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim*3, dim, 1)


    def forward(self, x):
        u = x.clone()
        # x = x.chunk(3, dim=-3)
        attn3 = self.conv3(x)
        attn5 = self.conv5(x)
        attn7 = self.conv_spatial(x)
        attn = torch.cat([attn3, attn5, attn7], dim=-3)
        attn = self.conv1(attn)

        return u * attn

class FAM(nn.Module):
    def __init__(self, in_channels):
        super(FAM,self).__init__()
        self.dwconv = S_conv(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        # self.relu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        self.lka = LKA1(in_channels)

    def forward(self, x0):

        B, N, C = x0.shape
        H = W = int(N ** 0.5)
        x = x0.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x1 = self.dwconv(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        x2 = self.lka(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        y = x1 + x2
        y = self.lka(y)
        y = self.bn(y)

        y = y + x1 + x2
        y = y.reshape(B, C, -1).permute(0, 2, 1)

        return y + x0

class Embed(nn.Module):
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x

class Merge(nn.Module):
    def __init__(self, dim, h, w):
        super(Merge, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.h = h
        self.dim = dim
        self.w = w
        self.norm = nn.BatchNorm2d(dim*2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
        x = self.norm(self.conv(x))

        return x.reshape(B, self.dim*2, -1).permute(0, 2, 1)

class Expand(nn.Module):
    def __init__(self, dim, h):
        super(Expand, self).__init__()
        self.dim = dim
        self.h = h
        self.conv = nn.ConvTranspose2d(self.dim, self.dim//2, 2, stride=2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.h)
        x = self.conv(x)

        return x.reshape(B, self.dim//2, -1).permute(0, 2, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = Embed(512)

        self.l1 = nn.Sequential(VSSBlock(96, 0., 128),
                                VSSBlock(96, 0., 128),
                                VSSBlock(96, 0., 128),
                                VSSBlock(96, 0., 128),
                                # VSSBlock(96, 0., 128),
                                )

        self.l2 = nn.Sequential(VSSBlock(192, 0., 64),
                                VSSBlock(192, 0., 64),
                                VSSBlock(192, 0., 64),
                                VSSBlock(192, 0., 64),
                                )

        self.l3 = nn.Sequential(VSSBlock(384, 0., 32),
                                VSSBlock(384, 0., 32),
                                VSSBlock(384, 0., 32),
                                VSSBlock(384, 0., 32),
                                )

        self.l4 = nn.Sequential(VSSBlock(768, 0., 16),
                                VSSBlock(768, 0., 16),
                                )

        self.m1 = Merge(96, 128, 128)
        self.m2 = Merge(192, 64, 64)
        self.m3 = Merge(384, 32, 32)

        self.p3 = Expand(768, 16)
        self.p2 = Expand(384, 32)
        self.p1 = Expand(192, 64)


        self.d3 = nn.Sequential(VSSBlock(384, 0., 32),
                                VSSBlock(384, 0., 32),
                                )

        self.d2 = nn.Sequential(VSSBlock(192, 0., 64),
                                VSSBlock(192, 0., 64),
                                )

        self.d1 = nn.Sequential(VSSBlock(96, 0., 128),
                                VSSBlock(96, 0., 128),
                                )


        self.dbm3 = FAM(384)
        self.dbm2 = FAM(192)
        self.dbm1 = FAM(96)

        self.up = nn.PixelShuffle(4)
        self.seg = nn.Conv2d(6, 1, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.embed(x) #torch.Size([1, 16384, 96])

        x1 = self.l1(x) #torch.Size([1, 16384, 96])
        #
        x = self.m1(x1) #torch.Size([1, 4096, 192])
        x2 = self.l2(x) #torch.Size([1, 4096, 192])

        #
        x = self.m2(x2) #torch.Size([1, 1024, 384])
        x3 = self.l3(x) #torch.Size([1, 1024, 384])

        #
        x = self.m3(x3) #torch.Size([1, 256, 768])
        x4 = self.l4(x) #torch.Size([1, 256, 768])

        #
        x = self.p3(x4) #torch.Size([1, 1024, 384])
        x3_temp = self.dbm3(x3)

        y3 = x3_temp + x
        x = self.d3(y3) #torch.Size([1, 1024, 384])
        #
        x = self.p2(x) #torch.Size([1, 4096, 192])
        x2_temp = self.dbm2(x2)

        y2 = x2_temp + x
        x = self.d2(y2)
        #
        x = self.p1(x) #torch.Size([1, 16384, 96])
        x1_temp = self.dbm1(x1)

        y1 = x1_temp + x
        x = self.d1(y1) #128x128

        x = self.up(x.permute(0, 2, 1).reshape(B, 96, 128, 128)) #torch.Size([1, 6, 512, 512])
        x = self.seg(x)

        return x

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512).cuda()
    part = Net().cuda()

    out = part(x)
    print(out.shape)

