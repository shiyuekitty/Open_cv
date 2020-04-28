import cv2
import numpy as np
from matplotlib import pyplot as plt
import function

if __name__ == '__main__':
    # 读入图片
    img = cv2.imread('../image/lena-gray.png', -1)

    # 分块均衡化直方图
    level_img = function.devide(img)

    # prewitt算子
    t_prewitt_1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    t_prewitt_2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    t_prewitt_3 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    t_prewitt_4 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    # 原始图prewitt算子进行边缘锐化
    img_prewitt_1 = function.prewittEdge(img, t_prewitt_1, t_prewitt_2)
    # 直方图均衡化+prewitt算子进行边缘锐化
    img_prewitt_2 = function.prewittEdge(level_img, t_prewitt_3, t_prewitt_4)

    # 图像二值化
    binaryzation_1 = function.custom_threshold(img_prewitt_1)
    binaryzation_2 = function.custom_threshold(img_prewitt_2)

    # 轮廓拟合
    Image_1= function.contours_demo(binaryzation_1)
    Image_2= function.contours_demo(binaryzation_2)


    # 结果显示
    # 原图
    plt.subplot(241), plt.imshow(img, cmap='gray')
    plt.title('primary'), plt.xticks([]), plt.yticks([])
    # 局部增强
    plt.subplot(242), plt.imshow(level_img, cmap='gray')
    plt.title('level_img'), plt.xticks([]), plt.yticks([])
    # 边缘锐化
    plt.subplot(243), plt.imshow(img_prewitt_1, cmap='gray')
    plt.title('prewitt_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(244), plt.imshow(img_prewitt_2, cmap='gray')
    plt.title('prewitt_2'), plt.xticks([]), plt.yticks([])
    # 二值化
    plt.subplot(245), plt.imshow(binaryzation_1, cmap='gray')
    plt.title('binaryzation_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(246), plt.imshow(binaryzation_2, cmap='gray')
    plt.title('binaryzation_2'), plt.xticks([]), plt.yticks([])
    # 提取边缘轮廓
    plt.subplot(247), plt.imshow(Image_1, cmap='gray')
    plt.title('Image_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(248), plt.imshow(Image_2, cmap='gray')
    plt.title('Image_2'), plt.xticks([]), plt.yticks([])
    plt.show()
