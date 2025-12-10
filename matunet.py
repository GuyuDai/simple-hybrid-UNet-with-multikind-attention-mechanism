import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CBAM, SEBlock, MHCA, ConvBlock

class MATUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base_channels=64):
        super(MATUNet, self).__init__()
        
        # --- Encoder (下采样路径) ---
        self.inc = ConvBlock(n_channels, base_channels)
        self.down1 = nn.MaxPool2d(2)
        
        self.enc1 = ConvBlock(base_channels, base_channels*2)
        self.down2 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_channels*2, base_channels*4)
        self.down3 = nn.MaxPool2d(2)
        
        # self.enc3 = ConvBlock(base_channels*4, base_channels*8)
        # self.down4 = nn.MaxPool2d(2)
        
        # Bottleneck (最底层)
        # self.bottleneck = ConvBlock(base_channels*8, base_channels*16)
        self.bottleneck = ConvBlock(base_channels*4, base_channels*8)

        # --- Skip Connection Processors (SE-ViT 部分) ---
        # 对编码器传来的特征进行 SE 处理
        # self.se3 = SEBlock(base_channels*8)
        self.se2 = SEBlock(base_channels*4)
        self.se1 = SEBlock(base_channels*2)
        self.se0 = SEBlock(base_channels)

        # --- Decoder (上采样路径 + MHCA) ---
        
        # Layer 4 Decoding
        # self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        # self.mhca4 = MHCA(dim=base_channels*8) # Cross Attention
        # self.dec4 = ConvBlock(base_channels*8, base_channels*8) # 融合后卷积
        
        # Layer 3 Decoding
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.mhca3 = MHCA(dim=base_channels*4)
        self.dec3 = ConvBlock(base_channels*4, base_channels*4)
        
        # Layer 2 Decoding
        # self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        # self.mhca2 = MHCA(dim=base_channels*2)
        # self.dec2 = ConvBlock(base_channels*2, base_channels*2)

        # Layer 2 Decoding without MHCA
        # directly cat
        # input channel numbers of dec2: up2 output (C*2) + se1 output (C*2) = C*4
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels*4, base_channels*2) 
        
        # Layer 1 Decoding
        # self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        # self.mhca1 = MHCA(dim=base_channels)
        # self.dec1 = ConvBlock(base_channels, base_channels)

        # Layer 2 Decoding without MHCA
        # directly cat
        # input channel numbers of dec1: up1 output (C) + se0 output (C) = C*2
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels*2, base_channels)

        # Final Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)            # 64, H, W
        x2 = self.enc1(self.down1(x1)) # 128, H/2, W/2
        x3 = self.enc2(self.down2(x2)) # 256, H/4, W/4
        # x4 = self.enc3(self.down3(x3)) # 512, H/8, W/8
        
        # x_base = self.bottleneck(self.down4(x4)) # 1024, H/16, W/16
        x_base = self.bottleneck(self.down3(x3))

        # --- Decoder with MHCA ---
        
        # Block 4
        # d4 = self.up4(x_base)        # Up: 512, H/8, W/8
        # s4 = self.se3(x4)            # Skip: 512, H/8, W/8 (Processed by SE)
        # d4 = self.mhca4(d4, s4)      # Fusion: Q=Decoder, K,V=Encoder
        # d4 = self.dec4(d4)           # Conv + CBAM
        
        # Block 3
        # d3 = self.up3(d4)            # Up: 256, H/4, W/4
        d3 = self.up3(x_base)
        s3 = self.se2(x3)            # Skip: 256
        d3 = self.mhca3(d3, s3)
        d3 = self.dec3(d3)
        
        # Block 2
        d2 = self.up2(d3)            # Up: 128, H/2, W/2
        s2 = self.se1(x2)
        # d2 = self.mhca2(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        
        # Block 1
        d1 = self.up1(d2)            # Up: 64, H, W
        s1 = self.se0(x1)
        # d1 = self.mhca1(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.outc(d1)
        return out
    
class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets):
        # inputs: [B, 1, H, W] (logits, 未经 sigmoid)
        # targets: [B, 1, H, W] (0 或 1)
        
        # BCE Loss (Binary Cross Entropy)
        # = - [y \log(p) + (1-y) \log(1-p)]
        # BCEWithLogitsLoss 内部自带 Sigmoid
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # Dice Loss
        # = 1 - \frac{2 \times |Pred \cap True|}{|Pred| + |True| + \epsilon}
        inputs_soft = torch.sigmoid(inputs) # 显式 sigmoid 用于 Dice 计算        
        # Flatten
        inputs_flat = inputs_soft.view(-1)
        targets_flat = targets.view(-1)
                
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice_score

        combined_loss = self.weight_bce * bce_loss + self.weight_dice * dice_loss
        
        return bce_loss, dice_loss, combined_loss