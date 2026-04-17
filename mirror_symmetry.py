import sys
import glob

# 尝试导入必要的依赖
try:
    import cv2
except ImportError:
    print("错误: 无法导入 cv2 模块")
    print("解决方案: 请运行以下命令安装 OpenCV:")
    print("python -m pip install opencv-python")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("错误: 无法导入 matplotlib 模块")
    print("解决方案: 请运行以下命令安装 matplotlib:")
    print("python -m pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("错误: 无法导入 numpy 模块")
    print("解决方案: 请运行以下命令安装 numpy:")
    print("python -m pip install numpy")
    sys.exit(1)

# 创建sift特征检测器
sift = cv2.SIFT_create()
# 创建 BFMatcher 对象
bf = cv2.BFMatcher()


def detecting_mirrorLine(title, picture_name, show_detail=False):
    """
    主要功能：检测图像的镜像对称轴并进行可视化展示
    """
    # 创建镜像对称检测对象
    mirror = Mirror_symmetry_detection(picture_name)

    # 寻找匹配点
    match_points = mirror.find_matchpoints()

    # 计算对称轴参数
    r_list, theta_list = mirror.find_point_r_theta(match_points)

    # 可视化匹配点
    if show_detail:
        mirror.draw_matches(match_points, top=10)
        mirror.draw_hex(r_list, theta_list)

    # 根据六边形图的计数进行排序
    image_hexbin = plt.hexbin(r_list, theta_list, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=True)

    # 绘制镜像对称轴
    mirror.draw_mirrorLine(r, theta, title)


def test_case(files_path):
    """
    测试用例:对指定路径下的图像文件进行镜像对称检测
    """
    files = sorted([f for f in glob.glob(files_path)])
    # 修复参数顺序错误
    for file in files:
        detecting_mirrorLine("With Mirror Line", file)


class Mirror_symmetry_detection:
    def __init__(self, image_path):
        self.image = self._read_color_image(image_path)
        self.reflected_image = np.fliplr(self.image)

        # 用SIFT检测关键点和描述符
        self.keypoints1, self.descriptors1 = sift.detectAndCompute(self.image, None)
        self.keypoints2, self.descriptors2 = sift.detectAndCompute(self.reflected_image, None)

    def _read_color_image(self, image_path):
        """
        读取彩色图像:将图像路径作为输入，返回读取的彩色图像
        """
        image = cv2.imread(image_path)
        # 增加图像读取失败检查
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}，请检查路径是否正确或文件是否存在")
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])  # 转换为RGB格式
        return image

    def find_matchpoints(self):
        """
        寻找匹配点:使用BFMatcher找到原图像和镜像图像之间的匹配点
        """
        matches = bf.knnMatch(self.descriptors1, self.descriptors2, k=2)

        # 仅保留优质匹配点（使用Lowe's ratio test）
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # 按距离排序
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        return good_matches

    def find_point_r_theta(self, match_points: list):
        """
        计算对称轴参数:根据匹配点计算镜像对称轴的参数（r和theta）
        """
        r_list = []
        theta_list = []
        h, w = self.image.shape[:2]  # 获取图像尺寸

        for match in match_points:
            pt1 = self.keypoints1[match.queryIdx].pt
            pt2 = self.keypoints2[match.trainIdx].pt

            # 修复镜像点坐标转换错误
            pt2 = (w - pt2[0], pt2[1])  # 镜像图像的点映射回原图像坐标系

            # 计算两点连线与x轴的夹角
            theta = angle_with_x_axis(pt1, pt2)
            # 计算中点
            xc, yc = midpoint(pt1, pt2)
            # 计算r参数
            r = xc * np.cos(theta) + yc * np.sin(theta)
            r_list.append(r)
            theta_list.append(theta)
        return r_list, theta_list

    def draw_matches(self, match_points, top=10):
        """
        可视化匹配点:在图像上绘制匹配点连线以进行可视化
        """
        # 修复变量名错误（kp1 -> keypoints1, kp2 -> keypoints2）
        img = cv2.drawMatches(
            self.image, self.keypoints1,
            self.reflected_image, self.keypoints2,
            match_points[:top], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.title(f"Top {top} pairs of symmetry points")
        plt.axis('off')
        plt.show()

    def draw_hex(self, r_list, theta_list):
        """
        绘制六边形图:根据r和theta列表绘制六边形图以展示对称轴参数分布
        """
        plt.figure(figsize=(8, 6))
        image_hexbin = plt.hexbin(r_list, theta_list, bins=200, cmap=plt.cm.Spectral_r)
        plt.colorbar(label='Counts')
        plt.xlabel('r')
        plt.ylabel('theta')
        plt.title('Hexbin plot of symmetry axes parameters')
        plt.show()

    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        """
        找到最大六边形图的坐标:返回最大六边形图的r和theta值
        """
        for k, v in sorted_vote.items():
            if vertical:
                return k[0], k[1]
            else:
                # 排除水平对称轴（theta接近0或pi）
                if not np.isclose(k[1], 0) and not np.isclose(k[1], np.pi):
                    return k[0], k[1]
        # 处理没有找到符合条件的情况
        return 0, np.pi / 2  # 默认返回垂直中线

    def sort_hexbin_by_votes(self, image_hexbin):
        """
        根据hexbin的计数对其进行排序
        """
        counts = image_hexbin.get_array()
        verts = image_hexbin.get_offsets()
        output = {}

        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc] > 0:  # 只保留有计数的点
                output[(binx, biny)] = counts[offc]

        # 按计数降序排序
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    def draw_mirrorLine(self, r, theta, title: str):
        """
        绘制镜像对称轴:在图像上绘制计算得到的镜像对称轴
        """
        # 创建图像副本以避免修改原图
        img_with_line = self.image.copy()
        h, w = img_with_line.shape[:2]

        # 绘制对称轴
        if np.isclose(np.cos(theta), 0):  # 垂直线
            x = int(r / np.sin(theta)) if not np.isclose(np.sin(theta), 0) else 0
            if 0 <= x < w:
                img_with_line[:, x:x + 2] = [255, 0, 0]  # 红色线
        else:
            for y in range(h):
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < w:
                    img_with_line[y, x:x + 2] = [255, 0, 0]  # 红色线

        # 显示结果
        plt.figure(figsize=(8, 6))
        plt.imshow(img_with_line)
        plt.axis('off')
        plt.title(title)
        plt.show()


def angle_with_x_axis(pt1, pt2):
    """
    计算两点连线与x轴的夹角（弧度制）
    """
    delta_y = pt2[1] - pt1[1]
    delta_x = pt2[0] - pt1[0]

    if delta_x == 0 and delta_y == 0:
        return 0  # 避免两点重合的情况

    angle = np.arctan2(delta_y, delta_x)  # 修复函数调用错误（arctan -> arctan2）
    # 确保角度在0到pi之间
    if angle < 0:
        angle += np.pi
    return angle


def midpoint(pt1, pt2):
    """
    计算两点的中点坐标
    """
    mx = (pt1[0] + pt2[0]) / 2
    my = (pt1[1] + pt2[1]) / 2
    return mx, my