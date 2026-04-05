# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
from torch import Tensor
from mmdet.models.task_modules.coders import \
    DistancePointBBoxCoder as MMDET_DistancePointBBoxCoder
# from mmdet.structures.bbox import bbox2distance, distance2bbox

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DistancePointBBoxCoder(MMDET_DistancePointBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.
    """

    def distance2bbox(
            self,
            points: Tensor,
            distance: Tensor,
            max_shape: Optional[Union[Sequence[int], Tensor,
            Sequence[Sequence[int]]]] = None
    ) -> Tensor:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
            max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
                optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]]
                and the length of max_shape should also be B.

        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """

        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]

        bboxes = torch.stack([x1, y1, x2, y2], -1)

        if max_shape is not None:
            if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
                # speed up
                bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
                bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
                return bboxes

            # clip bboxes with dynamic `min` and `max` for onnx
            if torch.onnx.is_in_onnx_export():
                # TODO: delete
                from mmdet.core.export import dynamic_clip_for_onnx
                x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
                bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
                return bboxes
            if not isinstance(max_shape, torch.Tensor):
                max_shape = x1.new_tensor(max_shape)
            max_shape = max_shape[..., :2].type_as(x1)
            if max_shape.ndim == 2:
                assert bboxes.ndim == 3
                assert max_shape.size(0) == bboxes.size(0)

            min_xy = x1.new_tensor(0)
            max_xy = torch.cat([max_shape, max_shape],
                               dim=-1).flip(-1).unsqueeze(-2)
            bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
            bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        return bboxes

    def bbox2distance(
            self,
            points: Tensor,
            bbox: Tensor,
            min_dis: Optional[float] = 0,
            max_dis: Optional[float] = None,
            eps: float = 0.1) -> Tensor:
        """Decode bounding box based on distances.

        Args:
            points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
            bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
            max_dis (float, optional): Upper bound of the distance.
            eps (float): a small value to ensure target < max_dis, instead <=

        Returns:
            Tensor: Decoded distances.
        """
        left = points[..., 0] - bbox[..., 0]
        top = points[..., 1] - bbox[..., 1]
        right = bbox[..., 2] - points[..., 0]
        bottom = bbox[..., 3] - points[..., 1]
        if max_dis is not None:
            left = left.clamp(min=min_dis, max=max_dis - eps)
            top = top.clamp(min=min_dis, max=max_dis - eps)
            right = right.clamp(min=min_dis, max=max_dis - eps)
            bottom = bottom.clamp(min=min_dis, max=max_dis - eps)
        return torch.stack([left, top, right, bottom], -1)

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
        # todo modify-3plus -- 已经使用离散分布预测了, 为什么这里还要乘上尺度??? 中心点岂不是会超出当前尺度的边界
        #  不会, 因为这里乘上尺度是将特征图上的位置坐标转换成原始图像上的坐标位置. 而离散分布预测是预测目标边界的偏移单位
        pred_bboxes = pred_bboxes * stride[None, :, None]

        return self.distance2bbox(points, pred_bboxes, max_shape)

    def encode(self,
               points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               min_dis: Optional[float] = 0,
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
        return self.bbox2distance(points, gt_bboxes, min_dis, max_dis, eps)
