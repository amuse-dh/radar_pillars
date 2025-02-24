import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (B, N, num_classes) logits
            targets: (B, N) class labels
            mask: (B, N) boolean mask for valid predictions
        Returns:
            loss: scalar tensor
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta
    
    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (B, N, 7) box predictions
            targets: (B, N, 7) box targets
            mask: (B, N) boolean mask for valid boxes
        Returns:
            loss: scalar tensor
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < self.beta,
                          0.5 * diff ** 2 / self.beta,
                          diff - 0.5 * self.beta)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, N, 1)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, masks):
        """
        Args:
            predictions: (B, H*W, num_classes)
            targets: (B, H*W)
            masks: (B, H*W)
        """
        # Flatten predictions and targets
        B = predictions.shape[0]
        predictions = predictions.reshape(B, -1, predictions.shape[-1])  # (B, H*W, num_classes)
        targets = targets.reshape(B, -1)  # (B, H*W)
        masks = masks.reshape(B, -1)  # (B, H*W)
        
        # Apply mask
        valid_mask = masks > 0
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate loss
        ce_loss = F.cross_entropy(valid_predictions, valid_targets, reduction='mean')
        return ce_loss

class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: (B, N, 7) box predictions
            targets: (B, N, 7) box targets
            mask: (B, N) boolean mask for valid boxes
        Returns:
            loss: scalar tensor
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < 0.11,
                          0.5 * diff ** 2 / 0.11,
                          diff - 0.5 * 0.11)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, N, 1)
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        return loss.mean()

class RadarPillarsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_loss = ClassificationLoss()
        self.reg_loss = RegressionLoss()
        
        #self.cls_weight = config.cls_weight
        self.reg_weight = config.reg_weight
    
    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets, reg_masks):
        """
        Args:
            cls_preds: (B, H, W, num_classes)
            reg_preds: (B, H, W, 7)
            cls_targets: (B, H, W)
            reg_targets: (B, H, W, 7)
            reg_masks: (B, H, W)
        """
        B = cls_preds.shape[0]
        H = cls_preds.shape[1]
        W = cls_preds.shape[2]
        
        # Reshape predictions and targets
        cls_preds = cls_preds.reshape(B, H*W, -1)  # (B, H*W, num_classes)
        reg_preds = reg_preds.reshape(B, H*W, -1)  # (B, H*W, 7)
        cls_targets = cls_targets.reshape(B, H*W)  # (B, H*W)
        reg_targets = reg_targets.reshape(B, H*W, -1)  # (B, H*W, 7)
        reg_masks = reg_masks.reshape(B, H*W)  # (B, H*W)
        
        print("reg_targets : ", reg_targets)
        
        # Calculate losses
        cls_loss = self.cls_loss(cls_preds, cls_targets, reg_masks)
        reg_loss = self.reg_loss(reg_preds, reg_targets, reg_masks)
        
        # Combine losses
        #total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        total_loss = cls_loss + reg_loss
        
        # loss_dict
        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
