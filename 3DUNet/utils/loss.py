import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply softmax to logits to get probabilities
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        # Flatten
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        dice = 0
        for i in range(num_classes):
            probs_i = probs[:, i, :]
            targets_i = targets[:, i, :]
            intersection = (probs_i * targets_i).sum(dim=1)
            dice_i = (2. * intersection + self.smooth) / (probs_i.sum(dim=1) + targets_i.sum(dim=1) + self.smooth)
            dice += dice_i

        dice = dice / num_classes
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6, ce_weight=1, dice_weight=1):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        num_classes = logits.shape[1]
        one_hot_targets = torch.eye(num_classes)[targets].permute(0, 3, 1, 2).float().to(logits.device)

        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, one_hot_targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# 使用示例
if __name__ == "__main__":
    # 随机生成预测和标签张量
    logits = torch.randn(2, 3, 256, 256, requires_grad=True)  # 预测张量 (batch_size, num_classes, height, width)
    targets = torch.randint(0, 3, (2, 256, 256))  # 标签张量 (batch_size, height, width)

    # 定义损失函数
    criterion = CombinedLoss()

    # 计算损失
    loss = criterion(logits, targets)
    print("Loss:", loss.item())
