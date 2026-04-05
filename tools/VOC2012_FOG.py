import os
import cv2
import math
import numpy as np
from numba import jit
from tqdm import tqdm

@jit()
def add_haze(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

def process_images(input_dir, output_dir, A=0.5, beta=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_list)} images.")

    for img_name in tqdm(image_list, desc="Generating foggy images"):
        image_path = os.path.join(input_dir, img_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            continue

        img_f = image / 255.0
        row, col, chs = image.shape
        center = (row // 2, col // 2)
        size = math.sqrt(max(row, col))

        foggy_image = add_haze(img_f, center, size, beta, A)
        img_foggy = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img_foggy)

if __name__ == '__main__':
    beta = 0.01 * 3 + 0.05
    input_folder = '/Home/destiny/mmyolo-main/data/VOC2012/JPEGImages'
    output_folder = f'/Home/destiny/mmyolo-main/data/VOC2012/JPEGImages-FOG-{beta}'
    process_images(input_folder, output_folder, A=0.5, beta=beta)
