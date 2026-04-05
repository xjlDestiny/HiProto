import os
import shutil

# # 原始 PNG 文件所在文件夹
# src_folder = "/Home/destiny/mmdetection_9/tests/data/Exdark_Voc/JPEGImages/IMGS_SCI_png"
# # 修改后文件保存的目标文件夹
# dst_folder = "/Home/destiny/mmdetection_9/tests/data/Exdark_Voc/JPEGImages/IMGS_SCI"
# # 原始 PNG 文件所在文件夹
# src_folder = "/Home/destiny/mmyolo-main/data/RTTS/VOC2007/JPEGImages_LH2D_png"
# # 修改后文件保存的目标文件夹
# dst_folder = "/Home/destiny/mmyolo-main/data/RTTS/VOC2007/JPEGImages_LH2D"
# # 原始 PNG 文件所在文件夹
# src_folder = "/Home/destiny/mmyolo-main/data/VOC2012/JPEGImages-FOG-0.1_LH2D_png"
# # 修改后文件保存的目标文件夹
# dst_folder = "/Home/destiny/mmyolo-main/data/VOC2012/JPEGImages-FOG-0.1_LH2D"
# # 原始 PNG 文件所在文件夹
# src_folder = "/Home/destiny/mmyolo-main/data/RTTS/VOC2007/JPEGImages_SGDN_png"
# # 修改后文件保存的目标文件夹
# dst_folder = "/Home/destiny/mmyolo-main/data/RTTS/VOC2007/JPEGImages_SGDN"
# 原始 PNG 文件所在文件夹
src_folder = "/Home/destiny/mmdetection_9/tests/data/Exdark_Voc/JPEGImages/IMGS_RetinexFormer_SDSD_indoor_png"
# 修改后文件保存的目标文件夹
dst_folder = "/Home/destiny/mmdetection_9/tests/data/Exdark_Voc/JPEGImages/IMGS_RetinexFormer_SDSD_indoor"

# 创建目标文件夹（如果不存在）
os.makedirs(dst_folder, exist_ok=True)

# 遍历所有 PNG 文件
for filename in os.listdir(src_folder):
    if filename.lower().endswith(".png"):
        # 原始文件完整路径
        src_path = os.path.join(src_folder, filename)
        # 新的文件名（.jpg后缀）
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        dst_path = os.path.join(dst_folder, new_filename)
        # 复制文件到目标文件夹并重命名
        shutil.copy(src_path, dst_path)
        print(f"已复制并重命名: {filename} -> {new_filename}")
