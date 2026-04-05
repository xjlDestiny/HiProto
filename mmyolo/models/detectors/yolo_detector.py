# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
import os
from torch import Tensor
import torch
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmyolo.registry import MODELS, VISUALIZERS

from datetime import datetime
import random
import cv2
import numpy as np
import shutil


INDEX_TO_CLS = {
    "DOTA-v1.0": {0: "plane", 1: "baseball-diamond", 2: "bridge", 3: "ground-track-field", 4: "small-vehicle",
                  5: "large-vehicle", 6: "ship", 7: "tennis-court", 8: "basketball-court", 9: "storage-tank",
                  10: "soccer-ball-field", 11: "roundabout", 12: "harbor", 13: "swimming-pool", 14: "helicopter"},
    "VOC2007": {0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
                5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
                10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
                15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"},
    "VOC2012": {0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
                5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
                10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
                15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"},
    "Exdark": {0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus", 4: "Car", 5: "Cat",
                6: "Chair", 7: "Cup", 8: "Dog", 9: "Motorbike", 10: "People", 11: "Table"},
    "RTTS": {0: "bicycle", 1: "bus", 2: "car", 3: "motorbike", 4: "person"},
    "UTDAC2020": {0: "echinus", 1: "starfish", 2: "holothurian", 3: "scallop"},
    "DUO": {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
}
CLS_TO_INDEX = {
    "DOTA-v1.0": {"plane": 0, "baseball-diamond": 1, "bridge": 2, "ground-track-field": 3, "small-vehicle": 4,
                  "large-vehicle": 5, "ship": 6, "tennis-court": 7, "basketball-court": 8, "storage-tank": 9,
                  "soccer-ball-field": 10, "roundabout": 11, "harbor": 12, "swimming-pool": 13, "helicopter": 14},
    "VOC2007": {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19},
    "VOC2012": {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19},
    "Exdark": {"Bicycle": 0, "Boat": 1, "Bottle": 2, "Bus": 3, "Car": 4, "Cat": 5,
                "Chair": 6, "Cup": 7, "Dog": 8, "Motorbike": 9, "People": 10, "Table": 11},
    "RTTS": {"bicycle": 0, "bus": 1, "car": 2, "motorbike": 3, "person": 4},
    "UTDAC2020": {"echinus": 0, "starfish": 1, "holothurian": 2, "scallop": 3},
    "DUO": {"holothurian": 0, "echinus": 1, "scallop": 2, "starfish": 3}
}

def adjust_boxes_hflip(boxes, image_width):
    """水平翻转后的坐标调整"""
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [2, 4]] = image_width - boxes[:, [4, 2]]  # 镜像x坐标
    return flipped_boxes
def adjust_boxes_vflip(boxes, image_height):
    """垂直翻转后的坐标调整"""
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [3, 5]] = image_height - boxes[:, [5, 3]]  # 镜像y坐标
    return flipped_boxes
def rotate_image_cv2(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # 获取旋转矩阵(2x3)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 执行仿射变换
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated_img


@MODELS.register_module()
class YOLODetector(SingleStageDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 dataset_name: str = "VOC2012",
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

        # init visualizer
        visualizer_cfg = dict(type='MyDetLocalVisualizer',
                              name='visualizer',
                              vis_backends=[dict(type='LocalVisBackend')])
        # visualizer_cfg = dict(type='mmdet.DetLocalVisualizer',
        #                       name='visualizer',
        #                       vis_backends=[dict(type='LocalVisBackend')])
        self.visualizer = VISUALIZERS.build(visualizer_cfg)
        # ## exdark dataset
        # self.visualizer.dataset_meta = {'classes': (
        #     'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'),
        #     'palette': [(101, 205, 228), (240, 128, 128), (154, 205, 50), (34, 139, 34), (0, 255, 255), (255, 165, 0),
        #                 (255, 0, 255), (255, 255, 0), (29, 123, 243), (139, 0, 0), (101, 205, 128), (240, 128, 28)]}
        # ## RTTS dataset
        # self.visualizer.dataset_meta = {
        #     'classes': ('bicycle', 'bus', 'car', 'motorbike', 'person'),
        #     'palette': [
        #         (101, 205, 228), (240, 128, 128), (160, 189, 97), (0, 139, 139), (255, 165, 0)
        #     ]
        # }
        ## VOC2012-FOG dataset
        self.visualizer.dataset_meta = {
            'classes': ("aeroplane", 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
            'palette': [
                (101, 205, 228), (240, 128, 128), (154, 205, 50), (34, 139, 34), (139, 0, 0),
                (139, 0, 255), (255, 0, 255), (255, 255, 0), (29, 123, 243), (0, 255, 255),
                (101, 205, 128), (240, 128, 28), (154, 205, 0), (34, 139, 200), (255, 165, 0),
                (255, 165, 150), (255, 150, 255), (255, 255, 150), (29, 123, 143), (150, 255, 255)
            ]
        }
        self.dataset_name = dataset_name
        # 获取当前工作目录
        current_directory = os.getcwd()
        now = datetime.now()
        # 格式化为字符串 (如：2025-04-20_15-30-45)
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-RetinexFormer/{dataset_name}/{current_time}/'
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-DiffUIR/{dataset_name}/{current_time}/'
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-Diff-Retinex++/{dataset_name}/{current_time}/'
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-DNMGDT/{dataset_name}/{current_time}/'
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-DiffDehaze/{dataset_name}/{current_time}/'
        # self.dir_result_display = f'{current_directory}/displayed_results/origin-SGND/{dataset_name}/{current_time}/'
        self.dir_result_display = f'{current_directory}/displayed_results/origin/{dataset_name}/{current_time}/'

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # # TODO modify-1 FE (flip enhancement)
        # bboxes_labels = batch_data_samples['bboxes_labels']
        # img_metas = batch_data_samples['img_metas']
        # batch_size = len(batch_inputs)
        # aug_inputs = []
        # aug_boxes = []
        # aug_img_metas = []
        # batch_flip_mode = []
        # # use cv2
        # for i, batch_input in enumerate(batch_inputs):
        #     device = batch_input.device
        #     ori_img = batch_input.clone().permute(1, 2, 0).detach().cpu().numpy()
        #     H, W = ori_img.shape[:2]
        #     original_boxes = bboxes_labels[bboxes_labels[:, 0] == float(i)]
        #     # 生成随机翻转模式 (各 33% 概率)
        #     flip_mode = random.choice(['h', 'v', 'h-v'])
        #     batch_flip_mode.append(flip_mode)
        #     if flip_mode == 'h':
        #         # todo 水平翻转
        #         flip_img = cv2.flip(ori_img, 1)
        #         flip_boxes = adjust_boxes_hflip(original_boxes, W)
        #         flip_boxes[:, 0] = flip_boxes[:, 0] + float(batch_size)
        #     elif flip_mode == 'v':
        #         # todo 垂直翻转
        #         flip_img = cv2.flip(ori_img, 0)
        #         # 坐标调整
        #         flip_boxes = adjust_boxes_vflip(original_boxes, H)
        #         flip_boxes[:, 0] = flip_boxes[:, 0] + float(batch_size)
        #     elif flip_mode == 'h-v':
        #         # todo 水平＋垂直翻转
        #         hflipped_img = cv2.flip(ori_img, 1)
        #         flip_img = cv2.flip(hflipped_img, 0)
        #         # 坐标调整
        #         hflipped_boxes = adjust_boxes_hflip(original_boxes, W)
        #         flip_boxes = adjust_boxes_vflip(hflipped_boxes, H)
        #         flip_boxes[:, 0] = flip_boxes[:, 0] + float(batch_size)
        #     else:
        #         flip_img = ori_img
        #         flip_boxes = original_boxes.clone()
        #         flip_boxes[:, 0] = flip_boxes[:, 0] + float(batch_size)
        #     aug_inputs.append(torch.from_numpy(flip_img.transpose(2, 0, 1)).to(device))
        #     aug_boxes.append(flip_boxes)
        #     aug_img_metas.append(img_metas[i])
        # aug_inputs = torch.stack(aug_inputs, 0)
        # aug_boxes = torch.cat(aug_boxes, 0)
        # batch_inputs_aug = torch.cat([batch_inputs, aug_inputs], 0)
        # batch_data_samples['bboxes_labels'] = torch.cat([bboxes_labels, aug_boxes], 0)
        # batch_data_samples['img_metas'] = img_metas + aug_img_metas
        # # # # todo 边缘提取
        # # # self.ted_model.to(batch_inputs_aug.device)
        # # # edges = self.ted_model(batch_inputs_aug * 255.)
        # # # edges = edges[3].clamp(0).squeeze().cpu().numpy()
        # # # todo 可视化
        # # for i, batch_input in enumerate(batch_inputs):
        # #     plt_imshow(batch_input.permute(1, 2, 0).cpu().numpy(), title="Original Image {}".format(i))
        # #     # plt_imshow(edges[i], title="Original Image {} -- edge".format(i), gray=True)
        # # for i, batch_input in enumerate(aug_inputs):
        # #     plt_imshow(batch_input.permute(1, 2, 0).cpu().numpy(), title="Augmented Image {}".format(i))
        # #     # plt_imshow(edges[i + batch_size], title="Augmented Image {} -- edge".format(i), gray=True)
        # x = self.extract_feat(batch_inputs_aug)

        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        x = self.extract_feat(batch_inputs)
        results_list, instance_vectors, bbox_vectors = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        # TODO modify 测试阶段可视化
        ind_class = [
            torch.unique(data_samples.gt_instances.labels) for data_samples in batch_data_samples
        ]
        for batch, img_sample in enumerate(batch_data_samples):
            path_ori_img = img_sample.img_path
            img_name = os.path.splitext(os.path.basename(path_ori_img))[0]
            dir_file = os.path.join(self.dir_result_display, f"val/{img_name}")
            os.makedirs(dir_file, exist_ok=True)
            # 复制图片到目标目录
            shutil.copy2(path_ori_img, dir_file)
            # 使用 batch_inputs[batch] 作为原始图像
            ori_img = batch_inputs[batch].permute(1, 2, 0).cpu().numpy()  # 从 CxHxW 转换为 HxWxC
            ori_img = cv2.normalize(ori_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)  # 假设原始是RGB格式, 转换为BGR
            img_h, img_w = ori_img.shape[:2]
            for level in range(len(x)):
                x_ori = x[level][batch]
                x_ori_gray = torch.mean(x_ori, dim=0, keepdim=False)
                # 归一化到 0-255 再保存
                x_norm = cv2.normalize(x_ori_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
                x_uint8 = x_norm.astype(np.uint8)
                # 创建热力图并叠加
                heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (img_w, img_h))
                superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
                # 保存结果
                cv2.imwrite(os.path.join(dir_file, f'gray-0_fpn_feat_layer{level + 1}.png'), x_uint8)
                cv2.imwrite(os.path.join(dir_file, f'heat-0_fpn_feat_layer{level + 1}.png'), superimposed_img)
            for level in range(len(instance_vectors)):
                cls_prototypes = self.bbox_head.head_module.cls_preds[level].out_conv.weight[ind_class[batch]]
                x_cls = instance_vectors[level][batch]
                x_cls_gray = torch.mean(x_cls, dim=0, keepdim=False)
                # 归一化到 0-255 再保存
                x_norm = cv2.normalize(x_cls_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
                x_uint8 = x_norm.astype(np.uint8)
                # 创建热力图并叠加
                heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (img_w, img_h))
                superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
                # 保存结果
                cv2.imwrite(os.path.join(dir_file, f'gray-1_cls_feat_layer{level + 1}.png'), x_uint8)
                cv2.imwrite(os.path.join(dir_file, f'heat-1_cls_feat_layer{level + 1}.png'), superimposed_img)
                # class activate maps
                x_cls_weighted = torch.sum(cls_prototypes * x_cls.unsqueeze(0), dim=1, keepdim=False)
                for k in range(len(x_cls_weighted)):
                    x_k = cv2.normalize(x_cls_weighted[k].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
                    x_k = x_k.astype(np.uint8)
                    # 创建热力图并叠加
                    heatmap = cv2.applyColorMap(x_k, cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (img_w, img_h))
                    superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
                    cls_name = INDEX_TO_CLS[self.dataset_name][ind_class[batch][k].item()]
                    # 保存结果
                    cv2.imwrite(os.path.join(dir_file, f'gray-layer{level + 1}--class_{cls_name}.png'), x_k)
                    cv2.imwrite(os.path.join(dir_file, f'heat-layer{level + 1}--class_{cls_name}.png'), superimposed_img)
            for level in range(len(bbox_vectors)):
                x_reg = bbox_vectors[level][batch]
                x_reg_gray = torch.mean(x_reg, dim=0, keepdim=False)
                # 归一化到 0-255 再保存
                x_norm = cv2.normalize(x_reg_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
                x_uint8 = x_norm.astype(np.uint8)
                # 创建热力图并叠加
                heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (img_w, img_h))
                superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
                # 保存结果
                cv2.imwrite(os.path.join(dir_file, f'gray-2_reg_feat_layer{level + 1}.png'), x_uint8)
                cv2.imwrite(os.path.join(dir_file, f'heat-2_reg_feat_layer{level + 1}.png'), superimposed_img)
            # 可视化目标框标注图
            image = cv2.imread(path_ori_img)
            rgb_img = cv2.cvtColor((image), cv2.COLOR_BGR2RGB)
            self.visualizer.add_datasample(
                img_name, rgb_img, img_sample,
                show=False, wait_time=0, draw_gt=True,
                draw_pred=True, pred_score_thr=0.4,
                out_file=os.path.join(dir_file, 'rgb-image-drew.jpg')
            )

        # # TODO modify 测试阶段可视化
        # # for i, batch_input in enumerate(batch_inputs):
        # #     plt_imshow(batch_input.permute(1, 2, 0).cpu().numpy(), title="Original Image {}".format(i))
        # if (self.epoch + 1) % 500 == 0:
        #     batch_size = len(batch_inputs)
        #     ind_class = [
        #         torch.unique(data_samples.gt_instances.labels) for data_samples in batch_data_samples
        #     ]
        #     for batch in range(batch_size):
        #         path_ori_img = batch_data_samples[batch].img_path
        #         img_name = os.path.splitext(os.path.basename(path_ori_img))[0]
        #         dir_file = os.path.join(self.dir_result_display, f"{self.epoch + 1}/{img_name}")
        #         os.makedirs(dir_file, exist_ok=True)
        #         # 使用 batch_inputs[batch] 作为原始图像
        #         ori_img = batch_inputs[batch].permute(1, 2, 0).cpu().numpy()  # 从 CxHxW 转换为 HxWxC
        #         ori_img = cv2.normalize(ori_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #         ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)  # 假设原始是RGB格式, 转换为BGR
        #         img_h, img_w = ori_img.shape[:2]
        #         # 复制图片到目标目录
        #         shutil.copy2(path_ori_img, dir_file)
        #         for level in range(len(x)):
        #             x_ori = x[level][batch]
        #             x_ori_gray = torch.mean(x_ori, dim=0, keepdim=False)
        #             # 归一化到 0-255 再保存
        #             x_norm = cv2.normalize(x_ori_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
        #             x_uint8 = x_norm.astype(np.uint8)
        #             # 创建热力图并叠加
        #             heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
        #             heatmap = cv2.resize(heatmap, (img_w, img_h))
        #             superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
        #             # 保存结果
        #             cv2.imwrite(os.path.join(dir_file, f'gray-0_fpn_feat_layer{level + 1}.png'), x_uint8)
        #             cv2.imwrite(os.path.join(dir_file, f'heat-0_fpn_feat_layer{level + 1}.png'), superimposed_img)
        #         for level in range(len(instance_vectors)):
        #             cls_prototypes = self.bbox_head.head_module.cls_preds[level].out_conv.weight[ind_class[batch]]
        #             x_cls = instance_vectors[level][batch]
        #             x_cls_gray = torch.mean(x_cls, dim=0, keepdim=False)
        #             # 归一化到 0-255 再保存
        #             x_norm = cv2.normalize(x_cls_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
        #             x_uint8 = x_norm.astype(np.uint8)
        #             # 创建热力图并叠加
        #             heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
        #             heatmap = cv2.resize(heatmap, (img_w, img_h))
        #             superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
        #             # 保存结果
        #             cv2.imwrite(os.path.join(dir_file, f'gray-1_cls_feat_layer{level + 1}.png'), x_uint8)
        #             cv2.imwrite(os.path.join(dir_file, f'heat-1_cls_feat_layer{level + 1}.png'), superimposed_img)
        #             # class activate maps
        #             x_cls_weighted = torch.sum(cls_prototypes * x_cls.unsqueeze(0), dim=1, keepdim=False)
        #             for k in range(len(x_cls_weighted)):
        #                 x_k = cv2.normalize(x_cls_weighted[k].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
        #                 x_k = x_k.astype(np.uint8)
        #                 # 创建热力图并叠加
        #                 heatmap = cv2.applyColorMap(x_k, cv2.COLORMAP_JET)
        #                 heatmap = cv2.resize(heatmap, (img_w, img_h))
        #                 superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
        #                 cls_name = INDEX_TO_CLS[self.dataset_name][ind_class[batch][k].item()]
        #                 # 保存结果
        #                 cv2.imwrite(os.path.join(dir_file, f'gray-layer{level + 1}--class_{cls_name}.png'), x_k)
        #                 cv2.imwrite(os.path.join(dir_file, f'heat-layer{level + 1}--class_{cls_name}.png'),
        #                             superimposed_img)
        #         for level in range(len(bbox_vectors)):
        #             x_reg = bbox_vectors[level][batch]
        #             x_reg_gray = torch.mean(x_reg, dim=0, keepdim=False)
        #             # 归一化到 0-255 再保存
        #             x_norm = cv2.normalize(x_reg_gray.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
        #             x_uint8 = x_norm.astype(np.uint8)
        #             # 创建热力图并叠加
        #             heatmap = cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)
        #             heatmap = cv2.resize(heatmap, (img_w, img_h))
        #             superimposed_img = cv2.addWeighted(heatmap, 0.4, ori_img, 0.6, 0)
        #             # 保存结果
        #             cv2.imwrite(os.path.join(dir_file, f'gray-2_reg_feat_layer{level + 1}.png'), x_uint8)
        #             cv2.imwrite(os.path.join(dir_file, f'heat-2_reg_feat_layer{level + 1}.png'), superimposed_img)

        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
