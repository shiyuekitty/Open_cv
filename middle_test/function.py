import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


# 绘制直方图
def histogram(img):
    if img.dtype == 'uint8':
        hist = np.zeros(256)
        l = 256
    elif img.dtype == 'uint16':
        hist = np.zeros(65536)
        l = 65536
    else:
        return 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1
    return l, hist


# 直方图均衡化
def level(img):
    # 绘制直方图
    L, hist = histogram(img)
    # 建一个空数组
    Cumulative_histogram = {}
    tmp = 0
    for i in range(hist.size):
        if hist[i] != 0:
            Cumulative_histogram[i] = hist[i] / img.size + tmp
            tmp = Cumulative_histogram[i]

    for i in Cumulative_histogram.keys():
        tmp = math.floor((L - 1) * Cumulative_histogram[i] + 0.5)
        Cumulative_histogram[i] = tmp

    new_img = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] in Cumulative_histogram.keys():
                new_img[i][j] = Cumulative_histogram[img[i][j]]
    l, new_hist = histogram(new_img)
    return new_img


# 分块直方图均衡化，获取局部增强结果
def devide(img):
    # 图像分块
    m, n = img.shape
    img1 = img[:int(m / 2), :int(n / 2)]
    img2 = img[:int(m / 2), int(n / 2):]
    img3 = img[int(m / 2):, :int(n / 2)]
    img4 = img[int(m / 2):, int(n / 2):]

    # 对分块后的直方图均衡化
    # 方法一
    # ret1 = cv2.equalizeHist(img1)
    # ret2 = cv2.equalizeHist(img2)
    # ret3 = cv2.equalizeHist(img3)
    # ret4 = cv2.equalizeHist(img4)

    # 方法二
    ret1 = level(img1)
    ret2 = level(img2)
    ret3 = level(img3)
    ret4 = level(img4)

    # 合并图像
    level1_img = np.zeros(img.shape)
    level1_img[:math.ceil(m / 2), :math.ceil(n / 2)] = ret1
    level1_img[:math.ceil(m / 2), math.ceil(n / 2):] = ret2
    level1_img[math.ceil(m / 2):, :math.ceil(n / 2)] = ret3
    level1_img[math.ceil(m / 2):, math.ceil(n / 2):] = ret4

    return level1_img


# 卷积
def imgConvolve(image, kernel):
    '''
    :param image: 图片矩阵
    :param kernel: 滤波窗口
    :return:卷积后的矩阵
    '''

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    # padding
    padding_h = int((kernel_h - 1) / 2)
    padding_w = int((kernel_w - 1) / 2)

    convolve_h = int(img_h + 2 * padding_h)
    convolve_W = int(img_w + 2 * padding_w)

    # 分配空间
    img_padding = np.zeros((convolve_h, convolve_W))
    # 中心填充图片
    img_padding[padding_h:padding_h + img_h, padding_w:padding_w + img_w] = image[:, :]
    # 卷积结果
    image_convolve = np.zeros(image.shape)
    # 卷积
    for i in range(padding_h, padding_h + img_h):
        for j in range(padding_w, padding_w + img_w):
            image_convolve[i - padding_h][j - padding_w] = int(
                np.sum(img_padding[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * kernel))

    return image_convolve


# Prewitt算子边缘锐化
def prewittEdge(image, prewitt_x, prewitt_y):
    '''
    :param image: 图片矩阵
    :param prewitt_x: 竖直方向
    :param prewitt_y:  水平方向
    :return:处理后的矩阵
    '''
    img_X = imgConvolve(image, prewitt_x)
    img_Y = imgConvolve(image, prewitt_y)

    img_prediction = np.zeros(img_X.shape)
    for i in range(img_prediction.shape[0]):
        for j in range(img_prediction.shape[1]):
            img_prediction[i][j] = max(img_X[i][j], img_Y[i][j])
    return img_prediction


# 图像二值化
def custom_threshold(image):
    # 计算图像均值
    h, w = image.shape[:2]
    m = np.reshape(image, [1, w * h])
    mean = m.sum() / (w * h)
    # print("mean:",mean)
    ret, binary = cv2.threshold(image, mean, 255, cv2.THRESH_BINARY)

    return binary


# 轮廓拟合
def contours_demo(image):
    # 原始二值化得到的图像是float类型，进行类型转换
    image = np.array(image, dtype='uint8')
    """
        cloneimage 显示图像，查找轮廓
        需要注意的是此处是cv2版本
        cv在2和4版本里面cv2.findContours需要的是两个参数
        在3版本里面只需要两个参数改成
        contours, heriachy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        即可
    """
    cloneimage, contours, heriachy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制图像
    for i, contour in enumerate(contours):
        # 函数cv2.drawContours()被用来绘制轮廓。
        # 第一个参数是一张图片，可以是原图或者其他。
        # 第二个参数是轮廓，也可以说是cv2.findContours()找出来的点集，一个列表。
        # 第三个参数是对轮廓（第二个参数）的索引，当需要绘制独立轮廓时很有用，若要全部绘制可设为-1。
        # 接下来的参数是轮廓的颜色和厚度。
        # print(i)
        cv2.drawContours(image, contours, i, (0, 0, 255), 2)

    return image


# 展示
def show(img, img1, img2, img3, img4, img5, img6, img7):
    # 原图
    plt.subplot(241), plt.imshow(img, cmap='gray')
    plt.title('primary'), plt.xticks([]), plt.yticks([])
    # 局部增强
    plt.subplot(242), plt.imshow(img1, cmap='gray')
    plt.title('level_img'), plt.xticks([]), plt.yticks([])
    # 边缘锐化
    plt.subplot(243), plt.imshow(img2, cmap='gray')
    plt.title('prewitt_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(244), plt.imshow(img3, cmap='gray')
    plt.title('prewitt_2'), plt.xticks([]), plt.yticks([])
    # 二值化
    plt.subplot(245), plt.imshow(img4, cmap='gray')
    plt.title('binaryzation_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(246), plt.imshow(img5, cmap='gray')
    plt.title('binaryzation_2'), plt.xticks([]), plt.yticks([])
    # 提取边缘轮廓
    plt.subplot(247), plt.imshow(img6, cmap='gray')
    plt.title('Image_1'), plt.xticks([]), plt.yticks([])
    plt.subplot(248), plt.imshow(img7, cmap='gray')
    plt.title('Image_2'), plt.xticks([]), plt.yticks([])
    # 结果显示
    plt.show()
