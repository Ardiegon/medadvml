import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for imbalanced classification.
        Args:
            alpha (Tensor, optional): Class weighting factor, must be a tensor of shape (num_classes,).
            gamma (float): Focusing parameter.
            reduction (str): 'mean' or 'sum' (default: 'mean').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss.
        Args:
            inputs: Logits from the model (shape: [batch_size, num_classes]).
            targets: True class labels (shape: [batch_size]).
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # Standard cross-entropy loss
        p_t = torch.exp(-ce_loss)  # Convert CE loss to probability

        focal_loss = (1 - p_t) ** self.gamma * ce_loss  # Apply focal factor
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Select alpha for each sample
            focal_loss = alpha_t * focal_loss  # Apply class weighting

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction