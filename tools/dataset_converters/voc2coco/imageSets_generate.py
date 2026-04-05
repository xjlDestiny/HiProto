import os

# 设置路径
train_folder = "/Home/destiny/mmyolo-main/data/VOC-FOG/train/JPEGImages"
val_folder = "/Home/destiny/mmyolo-main/data/VOC-FOG/val/JPEGImages"
output_folder = "/Home/destiny/mmyolo-main/data/VOC-FOG/ImageSets/Main"

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

def get_filenames_without_extension(folder):
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

# 提取文件名
train_filenames = get_filenames_without_extension(train_folder)
val_filenames = get_filenames_without_extension(val_folder)

# 写入 txt 文件
with open(os.path.join(output_folder, "train.txt"), 'w') as f:
    f.write('\n'.join(train_filenames))

with open(os.path.join(output_folder, "val.txt"), 'w') as f:
    f.write('\n'.join(val_filenames))

print("文件写入完成！")
