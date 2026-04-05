
import os
import shutil

images_file_path = '../../../tests/data/Exdark_Voc/JPEGImages/IMGS_ZeroDCE/'
split_data_file_path = '../../../tests/data/Exdark_Voc/ImageSets/main/'
new_images_file_path = '../../../tests/data/Exdark_ZeroDCE/'

if not os.path.exists(new_images_file_path + 'train'):
    os.makedirs(new_images_file_path + 'train')
if not os.path.exists(new_images_file_path + 'val'):
    os.makedirs(new_images_file_path + 'val')
if not os.path.exists(new_images_file_path + 'test'):
    os.makedirs(new_images_file_path + 'test')

dst_train_Image = new_images_file_path + 'train/'
dst_val_Image = new_images_file_path + 'val/'
dst_test_Image = new_images_file_path + 'test/'

total_txt = os.listdir(split_data_file_path)
for i in total_txt:
    name = i[:-4]
    if name == 'train':
        txt_file = open(split_data_file_path + i, 'r')
        for line in txt_file:
            line = line.strip('\n')
            line = line.strip('\r')
            # srcImage = images_file_path + line + '.jpg'
            # dstImage = dst_train_Image + line + '.jpg'
            srcImage = images_file_path + line
            dstImage = dst_train_Image + line
            shutil.copyfile(srcImage, dstImage)
        txt_file.close()
    elif name == 'val':
        txt_file = open(split_data_file_path + i, 'r')
        for line in txt_file:
            line = line.strip('\n')
            line = line.strip('\r')
            # srcImage = images_file_path + line + '.jpg'
            # dstImage = dst_val_Image + line + '.jpg'
            srcImage = images_file_path + line
            dstImage = dst_val_Image + line
            shutil.copyfile(srcImage, dstImage)
        txt_file.close()
    elif name == 'test':
        txt_file = open(split_data_file_path + i, 'r')
        for line in txt_file:
            line = line.strip('\n')
            line = line.strip('\r')
            # srcImage = images_file_path + line + '.jpg'
            # dstImage = dst_test_Image + line + '.jpg'
            srcImage = images_file_path + line
            dstImage = dst_test_Image + line
            shutil.copyfile(srcImage, dstImage)
        txt_file.close()
    else:
        print(f"Error, Please check the file name of folder--------{name}")