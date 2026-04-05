
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F

import time

# start = time.time()
# x_1 = torch.rand(32, 64, 49152)
# x_2 = x_1.permute(0, 2, 1)
# out = torch.bmm(x_1, x_2)
# end = time.time()
# print(end - start)

# TODO 加载图像
ori_img = cv2.imread("/data/home/xjl/workspace/mmyolo-main/data/DOTA-v1.0-gap200/train/P0012__1__0___0.png")
# ori_img = cv2.imread("/data/home/xjl/workspace/mmyolo-main/data/DOTA-v1.0-gap200/train/P1580__1__1648___2976.png")
# ori_img = cv2.imread("/data/home/xjl/workspace/mmyolo-main/data/DOTA-v1.0-gap200/train/P1589__1__2472___2976.png")
# ori_img = cv2.imread("/data/home/xjl/workspace/mmyolo-main/data/DOTA-v1.0-gap200/train/P2377__1__1345___0.png")
h, w, c = ori_img.shape
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 12))
plt.title("Original Image")
plt.imshow(ori_img)
plt.axis('off')
plt.show()
stride = 32
tensor_img = torch.from_numpy(ori_img).to(torch.float32).permute(2, 0, 1)
print("tensor_img.shape : ", tensor_img.shape)
# 添加 batch 维度, 变成 (1, 3, 1024, 1024)
img = tensor_img.unsqueeze(0)
# 使用 unfold 提取 patch（注意：每个 patch 是一个滑窗区域）
patches = F.unfold(img, kernel_size=stride, stride=stride)
patches_img = patches.permute(0, 2, 1).reshape(-1, 3, stride, stride)
print("patches_img.shape : ", patches_img.shape)
# for i in range(h // stride):
#     patch_i_img = patches_img[i].permute(1, 2, 0).to(torch.int16).numpy()
#     plt.figure(figsize=(12, 12))
#     plt.title("Patch_i Image")
#     plt.imshow(patch_i_img)
#     plt.axis('off')
#     plt.show()
# patches_img_ = patches_img.reshape(h // stride, h // stride, 3, stride, stride)
# print("patches_img_.shape : ", patches_img_.shape)
# for i in range(h // stride):
#     patch_i_img_ = patches_img_[0, i].permute(1, 2, 0).to(torch.int16).numpy()
#     plt.figure(figsize=(12, 12))
#     plt.title("Patch_i Image")
#     plt.imshow(patch_i_img_)
#     plt.axis('off')
#     plt.show()

flattened_patches_img = patches_img.view(patches_img.size(0), -1)
