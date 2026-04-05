# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
from mmdet.models.task_modules.coders import BaseBBoxCoder
from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class GaussianPredictionBBoxCoder(BaseBBoxCoder):
    """Gaussian Prediction BBox coder.

    This coder encodes gt bboxes (sig_x, sig_y, delta_x, delta_x)
    into (xmin, ymin, xmax, ymax) and decode it back to the original.
    """

    def __init__(self, clip_border: Optional[bool] = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_border = clip_border

    def gaussian2bbox(self, points, pred_bboxes, max_shape):
        """
        Args:
            points: Tensor of shape (B, N, 2)，通常为锚点或特征图中心点 (cx, cy)
            pred_bboxes: Tensor of shape (B, N, 4)，包含 (σₓ, σᵧ, dx, dy)

        Returns:
            bboxes: Tensor of shape (B, N, 4)，格式为 (xmin, ymin, xmax, ymax)
        """
        sigma_x = pred_bboxes[..., 0]
        sigma_y = pred_bboxes[..., 1]
        dx = pred_bboxes[..., 2]
        dy = pred_bboxes[..., 3]

        center_x = points[..., 0] + dx
        center_y = points[..., 1] + dy

        xmin = center_x - (sigma_x / 2)
        ymin = center_y - (sigma_y / 2)
        xmax = center_x + (sigma_x / 2)
        ymax = center_y + (sigma_y / 2)

        bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        return bboxes

    def bbox2gaussian(self, points, gt_bboxes, max_dis, eps):
        """
        Args:
            points: Tensor of shape (B, N, 2)，参考点 (cx, cy)
            gt_bboxes: Tensor of shape (B, N, 4)，格式为 (xmin, ymin, xmax, ymax)

        Returns:
            gaussians: Tensor of shape (B, N, 4)，格式为 (σₓ, σᵧ, dx, dy)
        """
        x_center = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
        y_center = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        sigma_x = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) / 2
        sigma_y = (gt_bboxes[..., 3] - gt_bboxes[..., 1]) / 2

        dx = x_center - points[..., 0]
        dy = y_center - points[..., 1]

        pred_bboxes = torch.stack([sigma_x, sigma_y, dx, dy], dim=-1)
        return pred_bboxes

    def decode(
        self,
        points: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride: torch.Tensor,
        max_shape: Optional[Union[Sequence[int], torch.Tensor,
                                  Sequence[Sequence[int]]]] = None
    ) -> torch.Tensor:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            stride (Tensor): Featmap stride.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None

        # pred_bboxes = pred_bboxes * stride[None, :, None]

        return self.gaussian2bbox(points, pred_bboxes, max_shape)

    def encode(self,
               points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               max_dis: float = 16.,
               eps: float = 0.01) -> torch.Tensor:
        """Encode bounding box to distances. The rewrite is to support batch
        operations.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2), The format is [x, y].
            gt_bboxes (Tensor or :obj:`BaseBoxes`): Shape (N, 4), The format
                is "xyxy"
            max_dis (float): Upper bound of the distance. Default to 16..
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.01.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4) or
             (B, N, 4).
        """

        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return self.bbox2gaussian(points, gt_bboxes, max_dis, eps)
