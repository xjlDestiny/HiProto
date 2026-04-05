# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple, Union, Optional
import os
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, InstanceList)
from mmdet.structures import SampleList
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess, make_divisible
from .yolov5_head import YOLOv5Head

import torch.nn as nn
from mmcv.cnn import ConvModule

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, cls_out_channels,
                 num_classes, norm_cfg, act_cfg):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=cls_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            in_channels=cls_out_channels,
            out_channels=cls_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.out_conv = nn.Conv2d(
            in_channels=cls_out_channels,
            out_channels=num_classes + 1,  # 添加背景原型向量
            kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        instance_vectors = self.conv2(x)
        cls_logit = self.out_conv(instance_vectors)
        cls_logit_wo_bg = cls_logit[:, :self.num_classes, ...]
        # # TODO-6 modify-6 Orthogonal Subspace Projection Algorithm
        # # 1. 获取类别原型 (num_class, dim)
        # cls_prototypes = self.out_conv.weight.squeeze()
        # num_class_w_bg, dim = cls_prototypes.shape
        # b, _, h, w = instance_vectors.shape
        # # 2. 重塑输入 (b, h*w, dim)
        # x_flat = instance_vectors.permute(0, 2, 3, 1).reshape(b, h * w, dim)
        # # 3. 初始化输出 (b, num_class, h, w)
        # cls_logit = torch.zeros((b, num_class_w_bg, h, w), device=x.device)
        # # 4. 对每个类别进行向量化OSP计算
        # for cls_idx in range(num_class_w_bg):
        #     # 当前类别目标信号 (dim, 1)
        #     d = cls_prototypes[cls_idx].unsqueeze(-1)  # (dim, 1)
        #     # 背景信号 (dim, num_class-1)
        #     U = torch.cat([
        #         cls_prototypes[:cls_idx],
        #         cls_prototypes[cls_idx + 1:]
        #     ], dim=0).T
        #     # 计算正交投影矩阵 (dim, dim)
        #     U_pinv = torch.linalg.pinv(U)
        #     P_ortho_U = torch.eye(dim, device=x.device) - U @ U_pinv
        #     # 计算OSP检测算子 (dim, 1)
        #     numerator = P_ortho_U @ d
        #     denominator = d.T @ P_ortho_U @ d
        #     P_OSP = numerator / (denominator + 1e-8)
        #     # 批量计算所有样本的logit (b, h*w, 1)
        #     logits = torch.matmul(x_flat, P_OSP)  # 关键优化点
        #     # 填充结果 (b, 1, h, w)
        #     cls_logit[:, cls_idx] = logits.view(b, h, w)
        # cls_logit_wo_bg = cls_logit[:, :self.num_classes, ...]
        return instance_vectors, cls_logit_wo_bg

class RegressionHead(nn.Module):
    def __init__(self, in_channels, reg_out_channels,
                 reg_len, norm_cfg, act_cfg):
        super().__init__()

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=reg_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            in_channels=reg_out_channels,
            out_channels=reg_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.out_conv = nn.Conv2d(
            in_channels=reg_out_channels,
            out_channels=4 * reg_len,
            kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        bbox_vectors = self.conv2(x)
        reg_out = self.out_conv(bbox_vectors)
        return bbox_vectors, reg_out


@MODELS.register_module()
class YOLOv8OSPHeadModule(BaseModule):
    """YOLOv8OSPHeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 img_scale: Sequence[int] = (1024, 1024),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_levels = len(featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.img_scale = img_scale
        self.featmap_strides = featmap_strides
        reg_max = img_scale[0] // featmap_strides[-1]
        # # TODO 小中大目标设定
        # self.reg_min = [0, 1, 2]
        # self.reg_max = [2, 4, reg_max]
        # # TODO 限定预测区间 (1)
        # self.reg_min = [0, 0, 0]  # specific_levels-BGProto-(5)-2
        # self.reg_max = [2, 4, reg_max]
        # TODO 限定预测区间 (2)
        self.reg_min = [0, 0, 0]  # specific_levels-BGProto-(5)-2
        self.reg_max = [4, 8, reg_max]
        # # TODO 限定预测区间 (3)
        # self.reg_min = [0, 0, 0]  # specific_levels-BGProto-(5)-2
        # self.reg_max = [8, 16, reg_max]
        ##################################################################
        ##################################################################
        # self.reg_min = [0, 0, 0]  # specific_levels-BGProto-(5)
        # self.reg_max = [4, 8, (reg_max) // 2]
        # self.reg_min = [0, 0, 0]  # specific_levels-BGProto-(5)-2-(di-dj-d1)
        # self.reg_max = [reg_max // 4, reg_max // 2, reg_max]
        # self.reg_min = [0, 1, 2]
        # self.reg_max = [4, 8, (reg_max) // 2]
        # # 限定更小范围的预测区间
        # self.reg_min = [0, 0, 0]
        # self.reg_max = [2, 4, (reg_max) // 2]
        # # TODO 原始YOLOv8设定
        # self.reg_min = [0, 0, 0]
        # self.reg_max = [reg_max, reg_max, reg_max]
        self.reg_len = []
        self.proj_ml = {}

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """ Initialize the weight and bias of PPYOLOE head. """
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            # todo xjl-modify
            reg_pred.out_conv.bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred.out_conv.bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (self.img_scale[0] / stride)**2)

    def _init_layers(self):
        """ initialize conv layers in YOLOv8 head. """
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(16, self.in_channels[0] // 4)
        cls_out_channels = max(self.in_channels[0], self.num_classes)
        for i in range(self.num_levels):
            proj = torch.arange(self.reg_min[i], self.reg_max[i] + 1, dtype=torch.float)
            self.proj_ml[self.img_scale[0] // self.featmap_strides[i]] = proj
            self.cls_preds.append(
                ClassificationHead(
                    self.in_channels[i], cls_out_channels,
                    self.num_classes, self.norm_cfg, self.act_cfg
                )
            )
            self.reg_preds.append(
                RegressionHead(
                    self.in_channels[i], reg_out_channels,
                    len(proj), self.norm_cfg, self.act_cfg
                )
            )
            self.reg_len.append(len(proj))

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tuple:
        """ Forward feature of a single scale level. """
        b, _, h, w = x.shape
        b, h, w = int(b), int(h), int(w)
        instance_vectors, cls_logit = cls_pred(x)
        bbox_vectors, bbox_dist_preds = reg_pred(x)
        if len(self.proj_ml[h]) > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, len(self.proj_ml[h]), h * w]).permute(0, 3, 1, 2)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj_ml[h].to(bbox_dist_preds.device).view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds

        # # todo test
        # if h == (self.img_scale[0] // 8) or h == (self.img_scale[0] // 16):
        #     bbox_preds = torch.zeros_like(bbox_preds)
        #     cls_logit = torch.zeros_like(cls_logit)

        if self.training:
            return instance_vectors, cls_logit, bbox_vectors, bbox_preds, bbox_dist_preds
        else:
            return instance_vectors, cls_logit, bbox_vectors, bbox_preds


@MODELS.register_module()
class YOLOv8OSPHead(YOLOv5Head):
    """YOLOv8OSPHead head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8OSPHead
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_osp: bool = True):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        # YOLOv8 doesn't need loss_obj
        self.loss_obj = None
        self.l1_loss = nn.L1Loss()
        self.use_osp = use_osp

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        instance_vectors, cls_logit, bbox_vectors, bbox_preds = self(x)

        predictions = self.predict_by_feat(
            instance_vectors, cls_logit, bbox_vectors, bbox_preds,
            batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions, instance_vectors, bbox_vectors

    def predict_by_feat(self,
                        instance_vectors: Sequence[Tensor],
                        cls_scores: Sequence[Tensor],
                        bbox_vectors: Sequence[Tensor],
                        bbox_preds: Sequence[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def loss(self,
             x: Tuple[Tensor],
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            instance_vectors: Sequence[Tensor],
            cls_scores: Sequence[Tensor],
            bbox_vectors: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)
        # todo 生成多层级（多尺度）先验框————根据特征图的下采样比例, 为每一个像素点生成先验框信息
        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # 如果特征图尺寸发生变化（可能是由于多尺度训练或者输入图像大小变化），就需要重新生成先验框
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # todo gt info --> 按照 batch 中目标数量的最大值进行填充, 使得所有图像的 gt 数据都具有相同的尺寸
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        # todo pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_dist_preds = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_len[i] * 4)
            for i, bbox_pred_org in enumerate(bbox_dist_preds)
        ]
        # flatten_pred_dists = [
        #     bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_len[i] * 4)
        #     for i, bbox_pred_org in enumerate(bbox_dist_preds)
        # ]
        # flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        # todo Decode regression results (left, top, right, bottom) to bboxes (tl_x, tl_y, br_x, br_y)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes, self.stride_tensor[..., 0])
        # todo 动态正样本分配 -- 将预测框（Prior Boxes）与真实框（Ground Truth）进行匹配，生成训练所需的监督信号
        #  此时, 预测框的格式已经被解码成 xyxy, 和真实框的格式一致
        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        # assigned_bboxes --> 每个 prior 相对应的 gt
        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        # print(flatten_cls_preds.max(), flatten_cls_preds.min()) # 预测的类别分数可能会负值
        # print(assigned_scores.max(), assigned_scores.min()) # 通过 normalize 后, assigned_scores的最大值不一定为1
        # TODO-1 分类损失  (这里的损失是否会和 CSMLoss 相对抗?? 只取出 !=0 的值计算损失 --> ××××)
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum
        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor
        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # TODO-2 iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # TODO-3 modify-3 dfl loss 是否可以优化一下???????
            state_index = 0
            end_index = 0
            loss_dfl = 0.
            for i in range(self.head_module.num_levels):
                bbox_dist_pred = flatten_dist_preds[i]
                end_index += bbox_dist_pred.size(1)  # index 前移
                fg_mask_cur_scale = fg_mask_pre_prior[:, state_index:end_index]
                # print(fg_mask_cur_scale.sum())
                if fg_mask_cur_scale.sum() == 0:
                    state_index = end_index  # 别忘了移动 index
                    continue  # 跳过当前 scale
                # 正常计算
                pred_dist_pos = bbox_dist_pred[fg_mask_cur_scale]
                assigned_ltrb = self.bbox_coder.encode(
                    self.flatten_priors_train[state_index:end_index, :2] / self.stride_tensor[state_index:end_index],
                    assigned_bboxes[:, state_index:end_index, :],
                    min_dis=self.head_module.reg_min[i],
                    max_dis=self.head_module.reg_max[i],
                    eps=0.01)
                assigned_ltrb_pos = torch.masked_select(
                    assigned_ltrb, prior_bbox_mask[:, state_index:end_index, :]).reshape([-1, 4])
                bbox_weight_cur_scale = torch.masked_select(
                    assigned_scores[:, state_index:end_index, :].sum(-1), fg_mask_cur_scale).unsqueeze(-1)
                pred_dist_pos_padded = F.pad(
                    pred_dist_pos.reshape(-1, self.head_module.reg_len[i]),
                    pad=(self.head_module.reg_min[i], 0),
                    mode='constant',
                    value=-float('inf'))
                loss_dfl_cur = self.loss_dfl(
                    pred_dist_pos_padded,
                    assigned_ltrb_pos.reshape(-1),
                    weight=bbox_weight_cur_scale.expand(-1, 4).reshape(-1),
                    avg_factor=assigned_scores_sum)
                loss_dfl += loss_dfl_cur
                state_index = end_index  # index 前移
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        # TODO-4 modify-4 Prototype Orthogonal Loss
        cls_prototypes = [cls_pred.out_conv.weight for cls_pred in self.head_module.cls_preds]
        loss_prototype = 0.
        for cls_prototypes_i in cls_prototypes:
            # loss_prototype_sl = self.prototype_regularization(cls_prototypes_i, type='svd')
            # loss_prototype_sl = self.prototype_regularization(cls_prototypes_i, type='cosine')
            loss_prototype_sl = self.prototype_regularization(cls_prototypes_i, type='soft_inner')
            loss_prototype += loss_prototype_sl

        # TODO-5 modify-5 Class Saliency Map Loss
        loss_saliency = self.compute_saliency_loss(
            instance_vectors=instance_vectors,
            cls_prototypes=cls_prototypes,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            strides=self.head_module.featmap_strides,
            use_osp=self.use_osp)

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
            loss_prototype=loss_prototype * num_imgs * world_size * 1.0,
            loss_saliency=loss_saliency * num_imgs * world_size * 1.0)
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size,
        #     loss_saliency=loss_saliency * num_imgs * world_size * 1.0)
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size,
        #     loss_prototype=loss_prototype * num_imgs * world_size * 1.0)
        # return dict(
        #     loss_cls=loss_cls * num_imgs * world_size,
        #     loss_bbox=loss_bbox * num_imgs * world_size,
        #     loss_dfl=loss_dfl * num_imgs * world_size)

    # myself
    def compare_loss(self, x_ori, x_aug, flip_mode, loss_type='distribution'):
        """
        Args:
            x_ori: 原始图像特征 (C, H, W)
            x_aug: 翻转图像特征 (C, H, W)
        """

        if flip_mode == 'h':
            # todo 水平翻转
            flip_x_aug = torch.flip(x_aug, dims=[2])
        elif flip_mode == 'v':
            # todo 垂直翻转
            flip_x_aug = torch.flip(x_aug, dims=[1])
        elif flip_mode == 'h-v':
            # todo 水平＋垂直翻转
            flip_x_aug = torch.flip(x_aug, dims=[1, 2])
        else:
            flip_x_aug = x_ori

        # # todo 可视化
        # x_ori_gray = torch.mean(x_ori, dim=0, keepdim=False)
        # flip_x_aug_gray = torch.mean(flip_x_aug, dim=0, keepdim=False)
        # plt_imshow(x_ori_gray.detach().cpu().numpy(), title='Original', gray=True)
        # plt_imshow(flip_x_aug_gray.detach().cpu().numpy(), title='Augmented', gray=True)

        # 像素级对齐损失
        pixel_loss = self.l1_loss(x_ori, flip_x_aug)
        # 分布对齐损失
        def distribution_loss(x, y):
            # 计算协方差差异
            x = x.view(x.size(0), -1)  # (C, H*W)
            y = y.view(y.size(0), -1)

            x_cov = torch.mm(x, x.t()) / (x.size(1) - 1)
            y_cov = torch.mm(y, y.t()) / (y.size(1) - 1)
            return F.mse_loss(x_cov, y_cov)
        dist_loss = distribution_loss(x_ori, flip_x_aug)
        # 动态权重调整
        alpha = torch.sigmoid(pixel_loss.detach() - dist_loss.detach())
        # 组合损失
        if loss_type == 'pixel':
            return pixel_loss
        elif loss_type == 'distribution':
            return dist_loss
        else:  # hybrid
            return alpha * pixel_loss + (1 - alpha) * dist_loss

    def prototype_regularization(self, W, type='svd'):
        def svd_regularization(W):
            """ SVD正则化 """
            W = W.view(W.size(0), -1)
            _l, s, _r = torch.svd(W)
            return torch.sum(torch.abs(s - 1))  # 强制奇异值为1
        def stiefel_regularization(W):
            """ 基于Stiefel流形的正交正则化 """
            if W.dim() != 2:
                W = W.squeeze()
            # 投影到Stiefel流形
            U, _, V = torch.svd(W)
            W_orth = torch.mm(U, V.t())
            # 计算到最近正交矩阵的距离
            return torch.norm(W - W_orth, p="fro") ** 2
        def orthogonal_regularization(W):
            """ 计算正交正则化损失 """
            if W.dim() != 2:
                W = W.squeeze()
            C = W.size(0)
            I = torch.eye(C, device=W.device)
            WWT = torch.mm(W, W.T)
            return torch.norm(WWT - I, p="fro") ** 2
        def pairwise_cosine_loss(W):
            """
            计算W中所有成对向量之间的余弦相似度，并最小化其平方（鼓励它们彼此正交）
            """
            if W.dim() != 2:
                W = W.squeeze()
            W = F.normalize(W, p=2, dim=1)  # L2归一化每一行
            cosine_sim = torch.matmul(W, W.T)  # (C, C)
            C = W.size(0)
            mask = ~torch.eye(C, dtype=bool, device=W.device)  # 排除对角线
            loss = (cosine_sim[mask] ** 2).mean()  # 余弦相似度平方越小越正交
            return loss
        def soft_inner_product_loss(W):
            """ 原型正交损失：最小化不同原型之间的内积绝对值 """
            if W.dim() != 2:
                W = W.squeeze()
            W = F.normalize(W, p=2, dim=1)  # 单位化每个原型向量
            inner_product = torch.matmul(W, W.T)
            C = W.size(0)
            mask = ~torch.eye(C, dtype=bool, device=W.device)
            loss = torch.abs(inner_product[mask]).mean()
            return loss

        if type == 'svd':
            return svd_regularization(W)
        elif type == 'stiefel':
            return stiefel_regularization(W)
        elif type == 'cosine':
            return pairwise_cosine_loss(W)
        elif type == 'soft_inner':
            return soft_inner_product_loss(W)
        else:
            return orthogonal_regularization(W)

    def compute_saliency_loss(
            self,
            instance_vectors: Sequence[Tensor],
            cls_prototypes: Sequence[Tensor],
            gt_bboxes: Tensor,
            gt_labels: Tensor,
            strides: Sequence[int],
            use_osp: bool = True) -> Tensor:
        """ Batch-optimized OSP saliency loss calculation. """
        num_levels = len(instance_vectors)
        batch_size = gt_bboxes.shape[0]
        device = gt_bboxes.device
        total_loss = torch.tensor(0.0, device=device)

        # Pre-process GT boxes for batch processing
        valid_masks = [(bboxes[..., 2:] - bboxes[..., :2]).min(dim=-1)[0] > 0
                       for bboxes in gt_bboxes]
        processed_bboxes = [bboxes[mask] for bboxes, mask in zip(gt_bboxes, valid_masks)]
        processed_labels = [labels[mask] for labels, mask in zip(gt_labels, valid_masks)]
        valid_images = sum(len(bboxes) > 0 for bboxes in processed_bboxes)

        if valid_images == 0:
            return torch.tensor(0.0, device=device)

        for level in range(num_levels):
            stride = strides[level]
            h, w = instance_vectors[level].shape[2:]

            # 获取当前层的预测范围
            min_size = self.head_module.reg_min[level] * stride  # 最小预测尺寸
            max_size = self.head_module.reg_max[level] * stride  # 最大预测尺寸

            # Prepare instance vectors (b, dim, h, w) -> (b, h*w, dim)
            x = instance_vectors[level].permute(0, 2, 3, 1).reshape(batch_size, h * w, -1)

            # Get prototypes (num_class+1, dim, 1, 1) -> (num_class+1, dim)
            prototypes = cls_prototypes[level].squeeze(-1).squeeze(-1)
            num_classes = prototypes.shape[0] - 1  # 最后一个类别是背景
            bg_class_idx = num_classes  # 背景类别索引

            # Initialize target and prediction saliency maps
            target_saliency = torch.zeros((batch_size, num_classes + 1, h, w), device=device)
            pred_saliency = torch.zeros_like(target_saliency)

            # Create grid coordinates for all images
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device) * stride + stride // 2,
                torch.arange(w, device=device) * stride + stride // 2,
                indexing='ij')
            grid_points = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)

            # Batch process GT boxes
            for img_idx in range(batch_size):
                # 记录所有前景目标区域
                fg_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
                for bbox, label in zip(processed_bboxes[img_idx], processed_labels[img_idx]):
                    cls_idx = int(label)
                    # 计算目标框的宽高
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    # 检查目标框尺寸是否在当前层的预测范围内
                    # if True:
                    if (bbox_w >= min_size and bbox_w <= max_size) and (bbox_h >= min_size and bbox_h <= max_size):
                    # min_w_h = torch.min(bbox_w, bbox_h)
                    # if (min_w_h >= min_size and min_w_h <= max_size):
                        # Calculate positive points for this bbox
                        positive_points = ((grid_points[..., 0] >= bbox[0] + stride // 2) &
                                           (grid_points[..., 0] <= bbox[2] - stride // 2) &
                                           (grid_points[..., 1] >= bbox[1] + stride // 2) &
                                           (grid_points[..., 1] <= bbox[3] - stride // 2))
                        # 使用torch.logical_or来叠加同一类别的多个目标框
                        target_saliency[img_idx, cls_idx] = torch.logical_or(
                            target_saliency[img_idx, cls_idx], positive_points).float()
                        # 更新全局前景掩码
                        fg_mask = torch.logical_or(fg_mask, positive_points)
                # 处理背景区域 (非前景区域即为背景)
                bg_mask = ~fg_mask
                target_saliency[img_idx, bg_class_idx] = bg_mask.float()

            # Skip level if no valid targets
            if target_saliency.sum() == 0:
                continue

            # Batch OSP calculation for all classes
            for cls_idx in range(num_classes + 1):
                # Target signal (dim, 1)
                d = prototypes[cls_idx].unsqueeze(-1)
                if use_osp:
                    # Background signals (dim, num_classes-1)
                    U = torch.cat([prototypes[:cls_idx], prototypes[cls_idx + 1:]], dim=0).T
                    # Compute OSP projection matrix
                    U_pinv = torch.linalg.pinv(U)
                    P_ortho_U = torch.eye(U.shape[0], device=device) - U @ U_pinv
                    # Compute OSP detector
                    numerator = P_ortho_U @ d
                    denominator = d.T @ P_ortho_U @ d
                    P_OSP = numerator / (denominator + 1e-8)
                    # Batch compute saliency for all images
                    saliency = torch.matmul(x, P_OSP)  # (b, h*w, 1)
                else:
                    saliency = torch.matmul(x, d)  # (b, h*w, 1)
                pred_saliency[:, cls_idx] = torch.sigmoid(saliency.view(batch_size, h, w))

            # Compute batch BCE loss
            pos_mask = (target_saliency == 1.0)
            neg_mask = (target_saliency == 0.0)

            pos_loss = -torch.log(pred_saliency[pos_mask] + 1e-8).sum()
            neg_loss = -torch.log(1.0 - pred_saliency[neg_mask] + 1e-8).sum()

            level_loss = (pos_loss + neg_loss) / (pos_mask.sum() + neg_mask.sum() + 1e-8)
            total_loss += level_loss

        return total_loss / num_levels if num_levels > 0 else total_loss

