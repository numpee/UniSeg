import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

__all__ = ['SegmentationBCELoss', 'SegmentationBCELossNoReduce']


class SegmentationBCELoss(BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, x, target):
        ignore_mask = target.sum(1) != 0
        num_gts = ignore_mask.sum()

        loss = super().forward(x, target)
        loss = loss * ignore_mask.unsqueeze(1).type(loss.type())
        return loss.sum() / num_gts


class SegmentationBCELossNoReduce(BCEWithLogitsLoss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, x, target):
        ignore_mask = target.sum(1) != 0
        num_gts = ignore_mask.sum()

        loss = super().forward(x, target)
        loss = loss * ignore_mask.unsqueeze(1).type(loss.type())
        return loss, num_gts


class SegmentationBCENullSecondStage(BCEWithLogitsLoss):
    """
    Assumes -1 values for targets in 'null' categories
    """

    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, x, target):
        loss = super().forward(x, target)
        # num_classes = target.size(1)
        loss_type = loss.type()
        non_ignore_cls_mask = (target == 1).sum(1).bool()  # Size([B, H, W])
        valid_cls_mask = (target != -1)  # Size([B, C, H, W])
        full_ignore_mask = valid_cls_mask * non_ignore_cls_mask.unsqueeze(1)
        loss = (loss * full_ignore_mask.type(loss_type)).sum()
        # loss = loss * num_classes / full_ignore_mask.sum()
        loss = loss / non_ignore_cls_mask.sum()
        return loss


class CustomCELoss(CrossEntropyLoss):
    def __init__(self, cls_masks, num_classes, ignore_index=-1, reduction='mean', device_idx=0):
        self.num_classes = num_classes
        self.cls_masks = torch.zeros(num_classes, num_classes).cuda(device_idx)
        self.ignore_index = ignore_index
        for i, t in enumerate(cls_masks):
            self.cls_masks[i, t] = float('-inf')
        super().__init__(ignore_index=ignore_index, reduction=reduction)

    def forward(self, pred, gt):
        ignore_gt_mask = gt != self.ignore_index
        ignore_gt_mask = ignore_gt_mask.type(gt.type())
        gt_masked = gt * ignore_gt_mask
        batch_size, C, H, W = pred.size()
        mask = torch.index_select(self.cls_masks, 0, gt_masked.flatten())
        mask = mask.view(-1, H, W, C).permute(0, 3, 1, 2)
        pred = pred + mask
        return super().forward(pred, gt)
