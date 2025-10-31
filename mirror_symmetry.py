import sys
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob


class Mirror_Symmetry_detection:
    def __init__(self, image_path: str):
        # 读取原图并转为灰度图
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # 生成水平翻转的镜像图像（用于对称特征匹配）
        self.reflected_image = np.fliplr(self.image)
        self.reflected_gray = cv2.cvtColor(self.reflected_image, cv2.COLOR_BGR2GRAY)

        # 初始化特征检测器：优先使用 SIFT，若不可用则回退到 ORB
        try:
            self.detector = cv2.SIFT_create()
            self.is_sift = True
        except Exception:
            try:
                # 一些旧版 OpenCV 可能在 xfeatures2d 中
                self.detector = cv2.xfeatures2d.SIFT_create()
                self.is_sift = True
            except Exception:
                # 回退到 ORB（无专利问题）
                self.detector = cv2.ORB_create(nfeatures=2000)
                self.is_sift = False

        # 提取特征（OpenCV 接口）
        self.kp1, self.des1 = self.detector.detectAndCompute(self.gray_image, None)
        self.kp2, self.des2 = self.detector.detectAndCompute(self.reflected_gray, None)

        # OpenCV KeyPoint 列表可以直接用于绘制
        self.cv_kp1 = self.kp1 if self.kp1 is not None else []
        self.cv_kp2 = self.kp2 if self.kp2 is not None else []

    def find_matchpoints(self):
        """使用暴力匹配器筛选优质特征匹配对"""
        if self.des1 is None or self.des2 is None:
            return []

        # 根据描述子类型选择距离度量
        norm_type = cv2.NORM_L2 if self.is_sift else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm_type)

        # K-近邻匹配（k=2）
        matches = bf.knnMatch(self.des1, self.des2, k=2)

        # 应用Lowe's比率测试筛选可靠匹配
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            # ratio 可根据算法选择
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 按匹配距离排序（距离越小越优）
        return sorted(good_matches, key=lambda x: x.distance)

    @staticmethod
    def angle_with_x_axis(x1, y1, x2, y2):
        """计算两点连线与x轴的夹角（弧度）"""
        dx = x2 - x1
        dy = y2 - y1
        return np.arctan2(dy, dx)

    @staticmethod
    def midpoint(x1, y1, x2, y2):
        """计算两点的中点坐标"""
        return (x1 + x2) / 2, (y1 + y2) / 2

    def find_points_r_theta(self, matches):
        """计算匹配点对的极坐标(r, theta)，用于对称轴投票"""
        r_list = []
        theta_list = []
        for match in matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx

            # 使用 OpenCV KeyPoint 的 pt 属性获取坐标
            x1, y1 = self.kp1[idx1].pt
            x2, y2 = self.kp2[idx2].pt

            theta = self.angle_with_x_axis(x1, y1, x2, y2)
            xc, yc = self.midpoint(x1, y1, x2, y2)
            r = xc * np.cos(theta) + yc * np.sin(theta)

            r_list.append(r)
            theta_list.append(theta)
        return r_list, theta_list

    def draw_matches(self, matches, num_matches=10):
        """绘制前N个最佳匹配的特征点对"""
        if not matches:
            print("无匹配点可绘制")
            return

        # cv2.drawMatches 支持直接传入 OpenCV KeyPoint 列表
        matched_img = cv2.drawMatches(
            self.image, self.cv_kp1,
            self.reflected_image, self.cv_kp2,
            matches[:num_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Top {num_matches} 特征匹配对")
        plt.axis('off')
        plt.show()

    def draw_hex(self, r_list, theta_list):
        """绘制极坐标投票的六边形直方图，可视化对称轴分布"""
        if not r_list or not theta_list:
            print("无投票数据可绘制")
            return None

        theta_deg = [np.rad2deg(theta) for theta in theta_list]

        plt.figure(figsize=(10, 8))
        hexbin = plt.hexbin(theta_deg, r_list, gridsize=30, cmap='viridis')
        plt.colorbar(hexbin, label='投票数')
        plt.xlabel('角度 (度)')
        plt.ylabel('r值')
        plt.title('对称特征投票分布')
        plt.show()
        return hexbin

    @staticmethod
    def sort_hexbin_by_votes(hexbin):
        """按投票数降序排列直方图单元"""
        if hexbin is None:
            return np.array([], dtype=int), np.array([], dtype=int)
        counts = hexbin.get_array()
        if counts is None or counts.size == 0:
            return np.array([], dtype=int), counts
        non_zero_idx = np.where(counts > 0)[0]
        sorted_idx = non_zero_idx[np.argsort(-counts[non_zero_idx])]
        return sorted_idx, counts

    @staticmethod
    def find_coordinate_maxhexbin(hexbin, sorted_idx):
        """找到投票最高的单元对应的(r, theta)"""
        if getattr(sorted_idx, "size", None) is not None:
            if sorted_idx.size == 0:
                return None, None
        elif not sorted_idx:
            return None, None

        verts = hexbin.get_offsets()
        if verts is None or len(verts) == 0:
            return None, None

        max_theta_deg, max_r = verts[sorted_idx[0]]
        max_theta = np.deg2rad(max_theta_deg)
        return max_r, max_theta

    def draw_mirrorLine(self, r, theta):
        """根据极坐标(r, theta)绘制镜像对称轴"""
        if r is None or theta is None:
            print("无法确定对称轴")
            return

        img_copy = self.image.copy()
        h, w = img_copy.shape[:2]

        # 极坐标方程：x*cos(theta) + y*sin(theta) = r
        # 计算直线与图像边界的交点
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if np.abs(sin_t) > 1e-6 and np.abs(cos_t) > 1e-6:
            x0 = r / cos_t
            x1 = (r - h * sin_t) / cos_t
            pt1 = (int(np.clip(x0, 0, w)), 0)
            pt2 = (int(np.clip(x1, 0, w)), h)
        elif np.abs(cos_t) <= 1e-6:  # 近似垂直
            x = int(np.clip(r / cos_t if cos_t != 0 else 0, 0, w))
            pt1 = (x, 0)
            pt2 = (x, h)
        else:  # 近似水平
            y = int(np.clip(r, 0, h))
            pt1 = (0, y)
            pt2 = (w, y)

        cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title(f"镜像对称轴 (r={r:.2f}, 角度={np.rad2deg(theta):.2f}°)")
        plt.axis('off')
        plt.show()


def detecting_mirrorLine(image_path):
    """主函数：检测图像的镜像对称轴"""
    try:
        detector = Mirror_Symmetry_detection(image_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    matches = detector.find_matchpoints()
    if len(matches) < 5:
        print(f"匹配点数量不足（仅{len(matches)}个），无法检测对称轴")
        return

    detector.draw_matches(matches)

    r_list, theta_list = detector.find_points_r_theta(matches)
    hexbin = detector.draw_hex(r_list, theta_list)

    sorted_idx, counts = detector.sort_hexbin_by_votes(hexbin)
    max_r, max_theta = detector.find_coordinate_maxhexbin(hexbin, sorted_idx)

    detector.draw_mirrorLine(max_r, max_theta)


def test_case(image_pattern):
    """批量处理图像（支持通配符路径）"""
    image_paths = glob.glob(image_pattern)
    if not image_paths:
        print(f"未找到匹配的图像: {image_pattern}")
        return

    for img_path in image_paths:
        print(f"\n处理图像: {img_path}")
        detecting_mirrorLine(img_path)


if __name__ == "__main__":
    test_case("*.jpg")
    test_case("*.png")
