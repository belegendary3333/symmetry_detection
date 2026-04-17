import sys
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from mirror_symmetry import *  # 调用优化后的镜像对称检测类


def main():
    argc = len(sys.argv)
    if not (argc == 2):
        print("Usage: python detect.py choice")
        return

    elif sys.argv[1] == 'example':  # 运行蝴蝶示例（自动生成优化后的特征点图）
        # 优化：show_detail=True时，会显示加权六边形图（原仅显示基础六边形图）
        detecting_mirrorLine("butterflywith Mirror Line", 'images/5_butterfly.png', show_detail=True)
        return

    elif sys.argv[1] == 'test':  # 运行测试用例（含阴影/倒影图片，自动应用预处理）
        """
        test animal and people image
        """
        test_case("images/1_*.png")

        """
        test symmetry architecture
        """
        test_case("images/2_*.png")

        """
        test other natural phenomenon
        """
        test_case("images/3_*.png")

        """
        test symmetry object with reflection（倒影测试，优化重点场景）
        """
        test_case("images/4_*.png")

        """
        test symmetry object with shadow（阴影测试，优化重点场景）
        """
        test_case("images/5_*.png")

        """
        test rotational symmetry
        """
        test_case("images/6_*.png")

        """
        test images includes multiple symmetry objects
        """
        test_case("images/7_*.png")

        """
        test non-symmetry object
        """
        test_case("images/8_*.png")
        return
    else:
        print("Error")
        return


if __name__ == '__main__':
    main()