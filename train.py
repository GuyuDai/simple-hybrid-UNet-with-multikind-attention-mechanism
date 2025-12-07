import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import wandb
from matunet import MATUNet, CombinedLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 # 根据显存调整，因为MHCA占显存较大，可能需要调小到 4 或 2
NUM_EPOCHS = 50
IMG_SIZE = 256
DATA_PATH_IMG = "./Kvasir-SEG/images"  # 修改为你的路径
DATA_PATH_MASK = "./Kvasir-SEG/masks"  # 修改为你的路径

def log_validation_images(model, val_loader, device):
    model.eval()
    images, masks = next(iter(val_loader)) # 取一个 Batch 的数据
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        # 生成预测 Mask (二值化)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    # 我们只取 Batch 中的第一张图来展示，避免上传太多
    # 1. 处理原图 (C, H, W) -> (H, W, C)
    img_display = images[0].cpu().permute(1, 2, 0).numpy()
    # 反归一化 (如果之前做了 Normalize，这里最好还原，不然图颜色很怪。这里简化直接显示)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    # 2. 处理 Mask (去掉 Channel 维度)
    mask_true = masks[0].cpu().squeeze().numpy()
    mask_pred = preds[0].cpu().squeeze().numpy()

    # 3. 创建 WandB 特有的 Mask 对象
    # 这会自动把原图、真实 Mask、预测 Mask 叠加在一起展示
    class_labels = {0: "Background", 1: "Polyp"}
    
    wandb_img = wandb.Image(
        img_display, 
        masks={
            "predictions": {
                "mask_data": mask_pred,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": mask_true,
                "class_labels": class_labels
            }
        },
        caption="Val Input vs Truth vs Pred"
    )
    return wandb_img

class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = os.listdir(image_dir)
        
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

def train():
    # record configurarion for wandb
    config = {
        "learning_rate": 1e-4,
        "batch_size": 8, # 假设是8
        "epochs": 100,
        "weight_decay": 1e-4,
        "architecture": "MATUNet",
        "dataset": "Kvasir-SEG"
    }

    # inint wandb project
    wandb.init(project="Medical-Seg-Project", config=config, name="MATUNet_Run_01")    
    # 获取 config (如果 wandb sweeping 可能会修改 config，所以这里重新读一下)
    cfg = wandb.config

    # prepare data
    dataset = KvasirSegDataset(DATA_PATH_IMG, DATA_PATH_MASK, img_size=IMG_SIZE)
    # split dataset: 80% train set, 20% validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # init model, optimizer, loss
    model = MATUNet(n_channels=3, n_classes=1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = CombinedLoss().to(DEVICE)

    # # 自动监控模型的梯度和参数分布
    # wandb.watch(model, log="all", log_freq=10)

    print(f"Start training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        bce_loss = 0
        dice_loss = 0
        train_loss = 0
        
        for images, masks in train_loader:
            # images: (B, 3, H, W)
            # masks:  (B, 1, H, W) -> 通常 DataLoader 出来是 Long 或 Float
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            # 梯度清零 (防止梯度累积)
            optimizer.zero_grad()            
            # Forward
            outputs = model(images)
            # 计算损失 (Calculate Loss)
            bce, dice, loss = criterion(outputs, masks)
            # Backward
            loss.backward()
            # 参数更新
            optimizer.step()
            # 记录 Loss
            bce_loss += bce.item()
            dice_loss += dice.item()
            train_loss += loss.item()

        avg_bce_loss = bce_loss / len(train_loader)
        avg_dice_loss = dice_loss / len(train_loader)    
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        model.eval()

        val_bce = 0
        val_dice = 0
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                v_bce, v_dice, v_loss = criterion(outputs, masks)
                val_bce += v_bce.item()
                val_dice += v_dice.item()
                val_loss += v_loss.item()
        
        avg_val_bce = val_bce / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)

        log_dict = {
            "epoch": epoch + 1,
            "train:total_loss": avg_train_loss,
            "train:bce_loss": avg_bce_loss,
            "train:dice_loss": avg_dice_loss,
            "val:total_loss": avg_val_loss,
            "val:bce_loss": avg_val_bce,
            "val:dice_loss": avg_val_dice
        }

        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epochs:
            visual_img = log_validation_images(model, val_loader, DEVICE)
            log_dict["val_prediction"] = visual_img
            
            # 保存模型文件到 wandb 云端 (可选，防止本地丢了)
            torch.save(model.state_dict(), "model.pth")
            wandb.save("model.pth")

        wandb.log(log_dict)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train -> Total: {avg_train_loss:.4f} | BCE: {avg_bce_loss:.4f} | Dice: {avg_dice_loss:.4f}")
        print(f"  Val   -> Total: {avg_val_loss:.4f} | BCE: {avg_val_bce:.4f} | Dice: {avg_val_dice:.4f}")
        
        # save model
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"hybrid_unet_epoch_{epoch+1}.pth")

        wandb.finish()

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH_IMG):
        print("请设置正确的数据集路径")
    else:
        train()