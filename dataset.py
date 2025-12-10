import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted(os.listdir(image_dir))
        
        # 基础预处理
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 灰度图
        
        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        # force change mask to 0 or 1
        mask = (mask > 0.5).float()
    
        return image, mask

class AugmentedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, mode='train'):
        """
        Args:
            image_dir (str): 图片路径
            mask_dir (str): Mask路径
            img_size (int): 缩放大小
            mode (str): 'train' 开启增强, 'val' 关闭增强
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.images = sorted(os.listdir(image_dir))
        
        # 预定义颜色增强器 (只变图片)
        # 亮度(brightness)、对比度(contrast)、饱和度(saturation) 浮动20%
        # 色相(hue) 浮动 5%
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.05
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # 1. 读取数据
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # 2. 训练集增强逻辑
        if self.mode == 'train':
            # --- A. 几何变换 (Geometric) ---
            # 必须保证 Image 和 Mask 同步变换
            
            # 1. 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 2. 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # --- B. 像素级变换 (Photometric) ---
            # 只变换 Image，绝对不要变换 Mask
            
            # 3. 颜色增强
            if random.random() > 0.2: # 80% 概率触发
                image = self.color_jitter(image)
                
            # 4. 高斯模糊 (模拟对焦不准)
            if random.random() > 0.8: # 20% 概率触发
                # kernel_size 必须是奇数，sigma 随机
                image = TF.gaussian_blur(image, kernel_size=3)

        # 3. 通用预处理 (Resize -> ToTensor -> Normalize)
        # Resize (图片用线性插值，Mask用最近邻插值)
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)
        
        # 转 Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # 标准化 (ImageNet 均值方差)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask 二值化 (消除 Resize 产生的中间值)
        mask = (mask > 0.5).float()
        
        return image, mask