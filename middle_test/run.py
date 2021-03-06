import numpy as np
import function


def run(img):
    # 分块均衡化直方图
    level_img = function.devide(img)

    # prewitt算子
    t_prewitt_1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    t_prewitt_2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    t_prewitt_3 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    t_prewitt_4 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])

    '''
        纵横方向
        原始图prewitt算子进行边缘锐化
    '''
    img_prewitt_1 = function.prewittEdge(img, t_prewitt_1, t_prewitt_2)
    '''
        对角线方向
        直方图均衡化后的prewitt算子进行边缘锐化 
    '''
    img_prewitt_2 = function.prewittEdge(level_img, t_prewitt_3, t_prewitt_4)

    # 图像二值化
    binaryzation_1 = function.custom_threshold(img_prewitt_1)
    binaryzation_2 = function.custom_threshold(img_prewitt_2)

    # 轮廓拟合
    Image_1 = function.contours_demo(binaryzation_1)
    Image_2 = function.contours_demo(binaryzation_2)

    # 展示
    function.show(img,
                  level_img,
                  img_prewitt_1, img_prewitt_2,
                  binaryzation_1, binaryzation_2,
                  Image_1, Image_2)