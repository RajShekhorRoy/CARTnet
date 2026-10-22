import torch


def dice_loss_targetwise(logits, targets, mask=None, eps=1e-6):
    """
    Target-wise Dice loss.

    logits:  [B, L]
    targets: [B, L]
    mask:    [B, L] (optional)

    Returns:
        Mean Dice loss across B targets/proteins.
    """

    probs = torch.sigmoid(logits)
    targets = targets.float()

    if mask is not None:
        mask = mask.float()
        probs = probs * mask
        targets = targets * mask

    # Calculate separately for each target
    intersection = (probs * targets).sum(dim=1)       # [B]
    denom = probs.sum(dim=1) + targets.sum(dim=1)     # [B]

    dice = (2 * intersection + eps) / (denom + eps)   # [B]

    dice_loss = 1 - dice                               # [B]

    # Average across targets
    return dice_loss.mean()

def dice_loss_from_logits(logits, targets, mask=None, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()

    if mask is not None:
        probs = probs * mask
        targets = targets * mask

    intersection = (probs * targets).sum()
    denom = probs.sum() + targets.sum()

    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice