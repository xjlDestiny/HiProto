import os
from typing import List, Tuple, Set
from numpy.core.multiarray import ndarray
import matplotlib.pyplot as plt
import copy
import math
import time
import cv2
import numpy as np
from sklearn.neighbors import KDTree

import torch
from third_parties.ted.ted import TED


def ndarraycoords_2_rect(coords: ndarray):
    coords = np.transpose(coords)
    row_coords = coords[0]
    col_coords = coords[1]
    if len(row_coords) == 0:
        return 0, 0, 0, 0
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords)

def rect_2_coords(left: int, top: int, right: int, bottom: int) -> Set[Tuple[int, int]]:
    return {(tb, lr)
        for tb in range(top, bottom + 1)
        for lr in range(left, right + 1)
    }

def coords_2_filtercoords(coords: Set[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    if len(coords) == 0:
        return [0], [0]
    coords_T = np.transpose(np.array(list(coords)))
    return coords_T[0], coords_T[1]


def coords_2_rect(coords: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    coords_T = np.transpose(list(coords))
    row_coords = coords_T[0]
    col_coords = coords_T[1]
    if len(row_coords) == 0:
        return 0, 0, 0, 0
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords)

def parse_txt_label(label_path):
    """解析TXT格式的旋转框标注文件"""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:  # 跳过无效行
                continue
            coords = list(map(float, parts[:8]))
            category = parts[8] if len(parts) > 8 else 'unknown'
            difficult = parts[9] if len(parts) > 9 else '0'
            points = [(int(coords[i]), int(coords[i + 1])) for i in range(0, 8, 2)]
            annotations.append({
                'points': points,
                'category': category,
                'difficult': difficult
            })
    return annotations

def parse_xml_label(label_path):
    """解析XML格式的水平框标注文件"""
    annotations = []
    tree = ET.parse(label_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        # 提取类别名称
        name = obj.find('name').text if obj.find('name') is not None else 'unknown'
        # 提取difficult标志
        difficult_elem = obj.find('difficult')
        difficult = difficult_elem.text if difficult_elem is not None else '0'
        # 提取坐标
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        # 将水平框转换为四个顶点
        points = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax)
        ]
        annotations.append({
            'points': points,
            'category': name,
            'difficult': difficult
        })
    return annotations

def get_n8(matrix: ndarray, r_idx: int, p_idx: int) -> List[Tuple[int, int]]:
    all_possibilities = [(r_idx - 1, p_idx - 1),
                         (r_idx - 1, p_idx),
                         (r_idx - 1, p_idx + 1),
                         (r_idx, p_idx - 1),
                         (r_idx, p_idx),
                         (r_idx, p_idx + 1),
                         (r_idx + 1, p_idx - 1),
                         (r_idx + 1, p_idx),
                         (r_idx + 1, p_idx + 1)]
    px_idx_max = len(matrix[0])
    row_idx_max = len(matrix)
    result = list(filter(lambda x: 0 <= x[0] < row_idx_max and 0 <= x[1] < px_idx_max, all_possibilities))
    return result


# TODO-1 生成图像的边缘图和边缘方向, 并对边缘响应进行分组
def detect_edges(img: ndarray) -> Tuple[ndarray, ndarray]:
    # TODO 初始化边缘检测模型
    ted_model = TED()
    for param in ted_model.parameters():
        param.requires_grad = False
    ted_model.load_state_dict(torch.load('../third_parties/ted/ted.pth'))
    ted_model.eval()

    modelFilename = "../model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    # # todo 结构化边缘提取
    # img_processed = (img / np.max(img)).astype(np.float32)
    # edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    # todo TED
    temp_img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    edges = ted_model(temp_img)
    edges = edges[3].clamp(0).squeeze().numpy()
    edges = edges / edges.max()
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    return edges_nms, orientation_map

# 对边缘检测结果进行分组，将连续且方向相似的边缘像素归为同一组
def group_edges(
        edges_nms_orig: ndarray,
        orientation_map: ndarray,
        threshold : float = 0.1) -> Tuple[ndarray, ndarray]:

    # 找到下一个待处理且未分组边缘像素坐标
    def get_new_todo(matrix: ndarray) -> Tuple[int, int]:
        todo = [coord for coord in coords_of_edges if matrix[coord[0], coord[1], 1] == -1]
        if len(todo) == 0:
            return -1, -1
        return todo[0]

    def get_next_todo(matrix: ndarray, curr_r_idx: int, curr_p_idx: int) -> Tuple[int, int]:
        root_coord = groups_members[edges_with_grouping[curr_r_idx][curr_p_idx][1]][0]
        for (ro, pi) in sorted(get_n8(matrix, curr_r_idx, curr_p_idx),
                               key=lambda coord: ((coord[0] - root_coord[0])**2 + (coord[1] - root_coord[1])**2)):
            if edges_with_grouping[ro][pi][0] != 1 or edges_with_grouping[ro][pi][1] != -1:
                continue
            return ro, pi
        return get_new_todo(matrix)

    edges_nms = edges_nms_orig
    edges_nms[edges_nms < threshold] = 0      # thresholding
    edges_nms[edges_nms >= threshold] = 1.0   # thresholding
    edges_nms = np.uint8(edges_nms)
    new_group_id: int = 0
    groups_diff_cum: List[float] = []
    groups_members: List[List[List[int]]] = []
    start = time.time()
    # 优化1：初始化分组矩阵
    edges_with_grouping = np.empty((*edges_nms.shape, 2), dtype=np.int64)
    edges_with_grouping[..., 0] = edges_nms
    edges_with_grouping[..., 1] = -1
    # # 优化2：快速获取边缘坐标
    # y_coords, x_coords = np.where(edges_nms == 1)
    # coords_of_edges = list(zip(y_coords, x_coords))  # 这个 y 轴是外层循环
    # edges_with_grouping = np.array([[[edges_nms[row_idx, px_idx], -1]
    #                                  for px_idx in range(len(edges_nms[0]))]
    #                                 for row_idx in range(len(edges_nms))])
    coords_of_edges = [(row_idx, px_idx)
                       for px_idx in range(len(edges_with_grouping[0]))
                       for row_idx in range(len(edges_with_grouping))
                       if edges_with_grouping[row_idx, px_idx, 0] == 1]
    end = time.time()
    print(f'Elapsed time: {end - start}')
    half_pi = math.pi / 2.0

    (row_idx, px_idx) = get_new_todo(edges_with_grouping)
    while True:
        if row_idx == -1 or px_idx == -1:
            break

        new_group_id_candidate: int = new_group_id
        # check N8 neighborhood
        px_orientation = orientation_map[row_idx, px_idx]
        for (r, p) in get_n8(edges_nms, row_idx, px_idx):
            if edges_nms[r, p] != 1 \
                    or edges_with_grouping[r][p][1] == -1 \
                    or groups_diff_cum[edges_with_grouping[r][p][1]] > (half_pi * 2.0):     # TODO Hier sollte man nicht verdoppeln
                continue
            current_diff: float = abs(px_orientation - orientation_map[r, p])
            current_diff = min(math.pi - current_diff, current_diff)  # difference in a circle
            new_group_id_candidate = edges_with_grouping[r][p][1]
            # update group information...
            groups_members[new_group_id_candidate].append([row_idx, px_idx])
            groups_diff_cum[new_group_id_candidate] += current_diff
            break
        else:
            # new group created:
            groups_diff_cum.append(0.0)
            groups_members.append([[row_idx, px_idx]])
            new_group_id += 1

        edges_with_grouping[row_idx][px_idx] = [edges_nms[row_idx, px_idx], new_group_id_candidate]
        edges_with_grouping[row_idx][px_idx][0] = edges_nms[row_idx, px_idx]
        edges_with_grouping[row_idx][px_idx][1] = new_group_id_candidate
        (row_idx, px_idx) = get_next_todo(edges_with_grouping, row_idx, px_idx)
    return edges_with_grouping, groups_members

def calculate_affinities(groups_members: ndarray, orientation_map: ndarray):
    def mean_of_coords(idx: int) -> ndarray:
        rows = [coord[0] for coord in groups_members[idx]]
        columns = [coord[1] for coord in groups_members[idx]]
        # Falls Zeit übrig: Investigieren, weshalb hier np.array steht, statt einfach ein Paar zurück zu geben
        return np.array([sum(rows) / len(rows), sum(columns) / len(columns)])

    def mean_of_orientations(idx: int) -> float:
        orientations = [orientation_map[(coord[0], coord[1])] for coord in groups_members[idx]]
        return sum(orientations) / len(orientations)

    groups_mean_position = [mean_of_coords(idx) for idx in range(len(groups_members))]
    groups_mean_orientation = [mean_of_orientations(idx) for idx in range(len(groups_members))]
    groups_min_row_idx = [np.min([g[0] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_max_row_idx = [np.max([g[0] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_min_col_idx = [np.min([g[1] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_max_col_idx = [np.max([g[1] for g in groups_members[idx]]) for idx in range(len(groups_members))]

    def calc_angle_between_points(coord_1: (int, int), coord_2: (int, int)) -> float:
        coord_diff = list(map(lambda a, b: a - b, coord_1, coord_2))
        if coord_diff[1] == 0.0:
            coord_diff[1] = 0.0001
        return (np.arctan(coord_diff[0]/coord_diff[1]) + (math.pi / 2.0)) / math.pi

    def calc_distance(group_id_1: int, group_id_2: int) -> float:
        distance = 10.0
        if(groups_min_row_idx[group_id_1] - groups_max_row_idx[group_id_2] > distance
                or groups_min_row_idx[group_id_2] - groups_max_row_idx[group_id_1] > distance
                or groups_min_col_idx[group_id_1] - groups_max_col_idx[group_id_2] > distance
                or groups_min_col_idx[group_id_2] - groups_max_col_idx[group_id_1] > distance):
            return math.inf
        mean_1 = groups_mean_position[group_id_1]
        mean_2 = groups_mean_position[group_id_2]
        c_with_d_1 = [(r, p, (r - mean_2[0])**2 + (p - mean_2[1])**2) for (r, p) in groups_members[group_id_1]]
        c_with_d_2 = [(r, p, (r - mean_1[0])**2 + (p - mean_1[1])**2) for (r, p) in groups_members[group_id_2]]
        nearest_1: Tuple[int, int, float] = sorted(c_with_d_1, key=lambda triple: triple[2])[0]
        nearest_2: Tuple[int, int, float] = sorted(c_with_d_2, key=lambda triple: triple[2])[0]
        return (nearest_1[0] - nearest_2[0])**2 + (nearest_1[1] - nearest_2[1])**2

    def calculate_affinity(group_id_1: int, group_id_2: int) -> float:
        if group_id_column == group_id_row:
            return 1.0
        max_distance = 8
        if calc_distance(group_id_1, group_id_2) > max_distance:
            return 0.0
        pos_1 = groups_mean_position[group_id_1]
        pos_2 = groups_mean_position[group_id_2]
        theta_12: float = calc_angle_between_points((pos_1[0], pos_1[1]), (pos_2[0], pos_2[1]))
        theta_1: float = groups_mean_orientation[group_id_1]
        theta_2: float = groups_mean_orientation[group_id_2]
        aff = abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 2.0
        # aff = abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 0.25  #** 2.0 TODO Eigentlich sollte hier quadriert werden
        if aff <= 0.05:
            return 0.0
        return aff

    # Code um die Ausrichtung des Winkels zu testen
    # def calculate_color_from_group(group_id: int):
    #     if group_id == -1:
    #         return [0.0, 0.0, 0.0, 0.0]
    #     group_coord = groups_mean_position[group_id]
    #     angle = calc_angle_between_points(group_coord, (len(edges_with_grouping) // 2, len(edges_with_grouping) // 2))
    #     rgb = colorsys.hsv_to_rgb(angle, 1.0, 1.0)
    #     return [rgb[0], rgb[1], rgb[2], 1.0]
    #
    #
    # return np.array([[calculate_color_from_group(edges_with_grouping[row_idx, px_idx, 1])
    #                   for px_idx in range(len(edges_with_grouping[0]))]
    #                  for row_idx in range(len(edges_with_grouping))])

    number_of_groups: int = len(groups_members)
    affinities: ndarray = np.zeros(shape=(number_of_groups, number_of_groups))
    for group_id_row in range(number_of_groups):
        for group_id_column in range(number_of_groups):  # range(group_id_row, number_of_groups):
            affinities[group_id_row, group_id_column] = calculate_affinity(group_id_row, group_id_column)
    return affinities


# TODO-2 可视化
def plt_show(img, title='Unnamed', save_path=None, figsize=(12, 12), show=False):
    """
    智能显示灰度或彩色图像

    参数:
        img: 输入图像 (numpy数组)
             - 单通道灰度图 (H,W)
             - 彩色图像 (H,W,3)
        title: 图像标题 (可选)
        figsize: 图像显示尺寸 (默认12x12英寸)
    """
    plt.figure(figsize=figsize)

    # 自动判断图像类型
    if len(img.shape) == 2:
        # 单通道灰度图
        plt.imshow(img, cmap='gray')
    elif img.shape[2] == 3:
        # 3通道BGR彩色图 (OpenCV默认格式)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        # 其他情况按灰度显示
        plt.imshow(img[:, :, 0], cmap='gray')
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    # 保存图像 (如果给定路径)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def draw_annotations(image_path, label_path):
    """ 可视化标注并返回所有中心点坐标和旋转角度 """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    drawn_image = copy.deepcopy(image)

    # 解析标签文件
    if label_path.endswith('.txt'):
        annotations = parse_txt_label(label_path)
    elif label_path.endswith('.xml'):
        annotations = parse_xml_label(label_path)
    else:
        raise ValueError("Unsupported label format. Only .txt and .xml are supported.")

    # 定义样式和存储中心点
    color_map = {
        'small-vehicle': (0, 255, 0),
        'large-vehicle': (255, 0, 0),
        'unknown': (0, 0, 255)
    }
    rbox_categories = []  # 储存所有旋转框的类别
    rbox_centers = []  # 存储所有中心点坐标
    rotation_angles = []  # 存储所有旋转角度

    # 绘制旋转框并计算角度
    for ann in annotations:
        points = np.array(ann['points'], dtype=np.float32)
        category = ann['category']
        rbox_categories.append(category)

        # 计算最小外接矩形
        rect = cv2.minAreaRect(points)
        (cx, cy), (w, h), angle = rect

        # 规范化角度到0-180度（基于长边）
        if w < h:  # 确保角度始终相对于长边
            angle += 90
            w, h = h, w  # 交换宽高
        angle = angle % 180  # 限制在0-180度

        # 存储计算结果
        rbox_centers.append((int(cx), int(cy)))
        rotation_angles.append(angle)

        # 绘制旋转框
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        box = np.int0(box)
        color = color_map.get(category, (0, 0, 255))
        cv2.drawContours(drawn_image, [box], 0, color, 2)

        # 绘制中心点
        cv2.circle(drawn_image, (int(cx), int(cy)), 6, (255, 0, 0), -1)

        # 绘制角度指示线（基于长边方向）
        angle_rad = np.deg2rad(angle)
        line_length = max(w, h) * 0.6
        end_point = (
            int(cx + line_length * np.cos(angle_rad)),
            int(cy + line_length * np.sin(angle_rad))
        )
        cv2.line(drawn_image, (int(cx), int(cy)), end_point, (255, 0, 0), 2)

        # # todo 显示角度文本
        # text = f"{angle:.1f}"
        # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        # text_org = (int(cx) + 10, int(cy) + 10)
        # # 文本背景
        # cv2.rectangle(drawn_image,
        #               (text_org[0] - 2, text_org[1] - text_size[1] - 2),
        #               (text_org[0] + text_size[0] + 2, text_org[1] + 2),
        #               (255, 255, 255), -1)
        # # 文本
        # cv2.putText(drawn_image, text, text_org,
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 保存和显示结果
    drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

    return drawn_image, rbox_centers, rbox_categories, annotations  # 返回中心点、标注和旋转角度

def visualize_groups_with_centers(filtered_groups, rbox_centers, img_shape):
    """
    可视化边缘组并标记中心点

    :param filtered_groups: 过滤后的边缘组列表
    :param rbox_centers: 中心点坐标列表 [(x1,y1), (x2,y2), ...]
    :param img_shape: 原图像尺寸 (height, width)
    :return: RGB可视化图像 (带红色中心点标记)
    """
    # 1. 创建灰度可视化矩阵
    vis_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for group_id, group in enumerate(filtered_groups, start=1):
        for (row, col) in group:
            vis_mask[row, col] = 255

    # 2. 转换为RGB图像
    vis_rgb = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)

    # 3. 绘制红色中心点 (半径为3像素)
    for center in rbox_centers:
        x, y = map(int, center)  # 确保坐标为整数
        cv2.circle(vis_rgb, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # 红色实心圆

    return vis_rgb


# TODO 通过边缘图和边缘方向生成伪目标框
def compute_pca_orientation(points):
    """使用PCA计算边缘组主方向（带异常处理）"""
    if len(points) < 2:
        return 0.0
    try:
        mean = np.mean(points, axis=0)
        cov = np.cov((points - mean).T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        return np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    except:
        return 0.0

def robust_min_area_rect(points, min_side=10):
    """带最小边长约束的最小外接矩形（带验证）"""
    if len(points) < 3:
        return ((0, 0), (min_side, min_side), 0)

    points = np.asarray(points, dtype=np.float32)
    if not np.all(np.isfinite(points)):
        return ((0, 0), (min_side, min_side), 0)

    try:
        rect = cv2.minAreaRect(points)
        w, h = rect[1]
        if min(w, h) < min_side:
            if w < h:
                rect = (rect[0], (min_side, max(h, min_side)), rect[2])
            else:
                rect = (rect[0], (max(w, min_side), min_side), rect[2])
        return rect
    except:
        return ((0, 0), (min_side, min_side), 0)

def build_enhanced_box(center, groups, min_points=10):
    """增强的旋转框构建方法（安全版）"""
    try:
        all_points = np.vstack([g['points'] for g in groups if 'points' in g])
        if len(all_points) < min_points:
            return None

        rect = robust_min_area_rect(all_points)
        box = cv2.boxPoints(rect)
        if box is None or len(box) < 4:
            return None

        box = np.asarray(box, dtype=np.float32)
        box += (center - np.mean(box, axis=0))
        return box
    except:
        return None

def create_smart_default(center, main_group):
    """基于物理尺寸的默认框（安全版）"""
    try:
        points = main_group.get('points', np.array([[0, 0], [1, 1]]))
        points = np.asarray(points, dtype=np.float32)

        x_coords = points[:, 0]
        y_coords = points[:, 1]
        length = max(x_coords.max() - x_coords.min(), 20)
        width = max(y_coords.max() - y_coords.min(), 10)

        group_center = main_group.get('center', center)
        vec = center - group_center
        width = max(width, np.linalg.norm(vec) * 1.5)

        angle = main_group.get('angle', 0)
        return cv2.boxPoints((
            center,
            (length, width),
            np.degrees(angle)
        ))
    except:
        return cv2.boxPoints(((center[0], center[1]), (40, 20), 0))

def rotated_iou(box1, box2):
    """计算两个旋转框的IOU（安全版）"""
    try:
        box1 = np.asarray(box1, dtype=np.float32)
        box2 = np.asarray(box2, dtype=np.float32)

        if len(box1) < 3 or len(box2) < 3:
            return 0.0
        if not np.all(np.isfinite(box1)) or not np.all(np.isfinite(box2)):
            return 0.0

        rect1 = cv2.minAreaRect(box1)
        rect2 = cv2.minAreaRect(box2)

        intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)

        if intersection_type == cv2.INTERSECT_NONE:
            return 0.0
        elif intersection_type == cv2.INTERSECT_PARTIAL:
            if len(intersection_points) < 3:
                return 0.0
            intersection_area = cv2.contourArea(intersection_points.astype(np.float32))
        else:
            intersection_area = min(cv2.contourArea(box1), cv2.contourArea(box2))

        area1 = cv2.contourArea(box1)
        area2 = cv2.contourArea(box2)
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0
    except:
        return 0.0

def scale_box(box, scale):
    """缩放旋转框（安全版）"""
    try:
        box = np.asarray(box, dtype=np.float32)
        if len(box) < 4 or not np.all(np.isfinite(box)):
            return box

        center = np.mean(box, axis=0)
        if isinstance(scale, (tuple, list)):
            scale = np.array([scale[0], scale[1]])
        return (box - center) * scale + center
    except:
        return box

def resolve_overlaps_enhanced(boxes, centers):
    """改进的重叠处理方法（安全版）"""
    valid_boxes = []
    for box in boxes:
        box = np.asarray(box, dtype=np.float32)
        if len(box) >= 4 and np.all(np.isfinite(box)):
            valid_boxes.append(box)

    for i in range(len(valid_boxes)):
        for j in range(i + 1, len(valid_boxes)):
            if rotated_iou(valid_boxes[i], valid_boxes[j]) > 0.1:
                try:
                    rect_i = cv2.minAreaRect(valid_boxes[i])
                    rect_j = cv2.minAreaRect(valid_boxes[j])

                    angle_diff = abs(rect_i[2] - rect_j[2]) % 180

                    if angle_diff < 15 or angle_diff > 165:
                        valid_boxes[i] = scale_box(valid_boxes[i], (0.9, 0.7))
                        valid_boxes[j] = scale_box(valid_boxes[j], (0.9, 0.7))
                    else:
                        valid_boxes[i] = scale_box(valid_boxes[i], 0.8)
                        valid_boxes[j] = scale_box(valid_boxes[j], 0.8)
                except:
                    continue

    return valid_boxes

def generate_improved_boxes(rbox_centers, groups_members, img_shape):
    """生成改进的旋转框（完整安全版）"""
    # 边缘组特征提取（带验证）
    group_features = []
    for group in groups_members:
        try:
            if len(group) < 5:
                continue

            points = np.array([(c, r) for (r, c) in group], dtype=np.float32)
            if len(points) < 2:
                continue

            angle = compute_pca_orientation(points)
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            group_features.append({
                'points': points,
                'center': np.mean(points, axis=0),
                'angle': angle,
                'length': max(x_coords.max() - x_coords.min(), 20),
                'width': max(y_coords.max() - y_coords.min(), 10)
            })
        except:
            continue

    # 构建KDTree（带空值检查）
    if not group_features:
        return [], np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

    kdtree = KDTree([gf['center'] for gf in group_features])

    # 生成旋转框（带异常处理）
    rotated_boxes = []
    for center in rbox_centers:
        try:
            center = np.asarray(center, dtype=np.float32)
            indices = kdtree.query_radius([center], r=50)[0]
            candidate_groups = [group_features[i] for i in indices if i < len(group_features)]

            if not candidate_groups:
                box = cv2.boxPoints(((center[0], center[1]), (40, 20), 0))
                rotated_boxes.append(np.asarray(box, dtype=np.float32))
                continue

            main_group = max(candidate_groups, key=lambda x: x['length'])
            box = build_enhanced_box(center, [main_group])
            if box is None:
                box = create_smart_default(center, main_group)
            rotated_boxes.append(np.asarray(box, dtype=np.float32))
        except:
            continue

    # 重叠处理（带验证）
    rotated_boxes = [b for b in rotated_boxes if len(b) >= 4 and np.all(np.isfinite(b))]
    if len(rotated_boxes) > 1:
        rotated_boxes = resolve_overlaps_enhanced(rotated_boxes, rbox_centers)

    # 可视化（安全渲染）
    vis_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for box in rotated_boxes:
        try:
            if len(box) >= 4:
                cv2.drawContours(vis_img, [np.int0(box)], 0, (0, 255, 0), 2)
        except:
            continue

    for center in rbox_centers:
        try:
            cv2.circle(vis_img, tuple(map(int, center)), 3, (0, 0, 255), -1)
        except:
            continue

    return rotated_boxes, vis_img