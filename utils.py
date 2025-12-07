import torch
import torch.nn as nn

# CBAM(Convolutional Block Attention Module) - Channel Attention
# input: (B, C, H, W)
# output: Attention Score
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全局最大池化: (B, C, H, W) -> (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 1x1卷积 <=> 全连接(降维): (B, C, 1, 1) -> (B, C/16, 1, 1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # 1x1卷积 <=> 全连接(复原): (B, C/16, 1, 1) -> (B, C, 1, 1)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# CBAM (Convolutional Block Attention Module) - Spatial Attention
# input: (B, C, H, W)
# output: attention score
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B, C, H, W) -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (B, C, H, W) -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # ~ -> (B, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # (B, 2, H, W) -> (B, 1, H, W)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

# CBAM:= Input -> Channel -> Spatial -> Output
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# SE Block (Squeeze-and-Excitation)
# 用于 Skip Connection 中的特征重标定
# input: (B, C, H, W)
# output: feature map (B, C, H, W)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1x1卷积 <=> 全连接(降维): (B, C, 1, 1) -> (B, C/16, 1, 1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        # 1x1卷积 <=> 全连接(复原): (B, C/16, 1, 1) -> (B, C, 1, 1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))))
        return x * out

# MHCA (Multi-Head Cross Attention) 
# Q <- Decoder
# K <- Eecoder
# V <- Eecoder
# Input: (B, C, H, W)
class MHCA(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super(MHCA, self).__init__()
        self.num_heads = num_heads
        # nn.MultiheadAttention(Seq_Len, Batch, Embed_Dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=qkv_bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat, key_value_feat):
        b, c, h, w = query_feat.shape        
        # 展平 spatial dimensions
        # flatten(2): (B, C, H, W) -> (B, C, H*W)
        # permutate(2,0,1): (B, C, H*W) -> (H*W, B, C)
        q = query_feat.flatten(2).permute(2, 0, 1)
        q = self.norm(q)
        k = key_value_feat.flatten(2).permute(2, 0, 1)
        k = self.norm(k)
        v = key_value_feat.flatten(2).permute(2, 0, 1)
        v = self.norm(v)
        attn_output, _ = self.attn(query=q, key=k, value=v)        
        # 恢复形状
        out = attn_output.permute(1, 2, 0).view(b, c, h, w)        
        # 残差连接
        return out + query_feat
    
# Conv:= Conv -> BN -> ReLU -> CBAM
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.cbam(out)
    
        # 如果输入输出通道数一样，可以加个残差，防止梯度消失
        if x.shape[1] == out.shape[1]:
            out += identity 
        return out