import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage import util,color
import random
import datetime
import pywt

## for color image show with plt
'''
    cv读入图像与plt显示图像之间bgr->rgb的转换
'''
def img_plt(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img

def img_translation(img,tx,ty):
    dst_img = np.zeros((img.shape),dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i+tx<dst_img.shape[0] and j+ty<dst_img.shape[1]:
               dst_img[i+tx][j+ty] = img[i][j]
    return dst_img

def img_resize(img,sx,sy):
    if len(img.shape)<=2:
        dst_img = np.zeros((round(img.shape[0]*sx),round(img.shape[1]*sy)),dtype='uint8')
    else:
        dst_img = np.zeros((round(img.shape[0] * sx), round(img.shape[1] * sy),img.shape[2]), dtype='uint8')
    for i in range(dst_img.shape[0]):
        for j in range(dst_img.shape[1]):
            if round(i/sx) < img.shape[0] and round(j/sy) < img.shape[1]:
                dst_img[i][j] = img[round(i/sx)][round(j/sy)]
    return dst_img

def img_rotation(img,th):
    dst_img = np.zeros((img.shape), dtype='uint8')
    row = img.shape[0]
    col = img.shape[1]
    # x = x'cos(theta)-y'sin(theta) + m/2*(1-cos(theta))+n/2*sin(theta)
    # y = x'sin(theta)+y'cos(theta) + n/2*(1-cos(theta))-m/2*sin(theta)
    for i in range(row):
        for j in range(col):
            m = i*math.cos(th)-j*math.sin(th)+row/2*(1-math.cos(th))+col/2*math.sin(th)
            n = i*math.sin(th)+j*math.cos(th)+col/2*(1-math.cos(th))-row/2*math.sin(th)
            if m >=0 and m < row and n >=0 and n<col:
                dst_img[i][j] = img[math.floor(m)][math.floor(n)]
    return dst_img

# 最近邻插值算法
# dst_h为新图的高;dst_w为新图的宽
def NN_interpolation(img,dst_h,dst_w):
    scr_h = img.shape[0]
    scr_w = img.shape[1]
    if len(img.shape)>2:
        dst_img=np.zeros((dst_h,dst_w,img.shape[2]),dtype=np.uint8)
    else:
        dst_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            scr_x=round(i*(scr_h/dst_h))
            scr_y=round(j*(scr_w/dst_w))
            if scr_x < scr_h and scr_y < scr_w:
                dst_img[i,j]=img[scr_x,scr_y]
    return dst_img

## 双线性插值
def bilinear_interpolation(img, dst_h,dst_w):
    src_h = img.shape[0]
    src_w = img.shape[1]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    if len(img.shape) > 2:
        dst_img = np.zeros((dst_h, dst_w, img.shape[2]), dtype=np.uint8)
    else:
        dst_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
    scale_x, scale_y = float(src_h) / dst_h, float(src_w) / dst_w
    for dstx in range(dst_h):
        for dsty in range(dst_w):

            # find the origin x and y coordinates of dst image x and y
            # use geometric center symmetry
            # if use direct way, src_x = dst_x * scale_x
            srcy = (dsty + 0.5) * scale_y - 0.5  # yp  y'
            srcx = (dstx + 0.5) * scale_x - 0.5  # xp  x'

            # find the coordinates of the points which will be used to compute the interpolation
            src_y0 = int(math.floor(srcy))       # j
            src_y1 = min(src_y0 + 1, src_w - 1)   # j+1
            src_x0 = int(math.floor(srcx))       # i
            src_x1 = min(src_x0 + 1, src_h - 1)   # i+1
            ##  A(i,j) B(i+1,j)  C(i,j+1)  D(i+1,j+1)
            if src_x0 != src_x1 and src_y1 != src_y0:
                ### calculate the interpolation
                ge = ((src_x1 - srcx) * img[src_x0, src_y0] + (srcx - src_x0) * img[src_x1, src_y0]) / (src_x1 - src_x0)
                gf = ((src_x1 - srcx) * img[src_x0, src_y1] + (srcx - src_x0) * img[src_x1, src_y1] )/ (src_x1 - src_x0)
                dst_img[dstx, dsty] = ((src_y1 - srcy) * ge + (srcy - src_y0) * gf) / (src_y1 - src_y0)
    return dst_img

if __name__ == '__main__':
    #####  Image correction
    ### warpAffine
    # cv2.flip()                                   # 图像翻转
    # cv2.warpAffine()                             # 图像仿射
    # cv2.getRotationMatrix2D()                    #取得旋转角度的Matrix
    # cv2.GetAffineTransform(src, dst, mapMatrix)  #取得图像仿射的matrix
    # cv2.getPerspectiveTransform(src, dst)        #取得图像透视的４个点起止值
    # cv2.warpPerspective()                        #图像透视

    ###  Gray interpolation
    # img = cv2.imread('lena-color.jpg')
    # dst_NN = NN_interpolation(img, 1024,1024)
    # dst_bi = bilinear_interpolation(img, 1024,1024)
    # img = img_plt(img)
    # dst_NN = img_plt(dst_NN)
    # dst_bi = img_plt(dst_bi)
    # dif = dst_NN - dst_bi
    # plt.figure(1)
    # plt.subplot(221),plt.xlabel("original image"),plt.imshow(img)
    # plt.subplot(222),plt.xlabel("NN interpolation"), plt.imshow(dst_NN)
    # plt.subplot(223),plt.xlabel("Bilinear interpolation"), plt.imshow(dst_bi)
    # plt.subplot(224), plt.xlabel("Dif with NN and Bi"), plt.imshow(dif)
    # plt.show()
    ##### Image Inpaint with opencv
    ## get the mask image
    # img1 = cv2.imread('lena-color-need-process.png')
    # img2 = cv2.imread('lena-color.jpg')
    # mask_img = img1-img2
    # for i in range(mask_img.shape[0]):
    #     for j in range(mask_img.shape[1]):
    #         if mask_img[i,j,0] != 0 and mask_img[i,j,1] != 0 and mask_img[i,j,2] != 0:
    #             mask_img[i,j,:] = 255
    # cv2.imwrite('lena-color-mask.png',mask_img)
    #### image inpaint
    img1 = cv2.imread('lena-color-need-process.png')
    mask_img = cv2.imread('lena-color-mask.png')
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    dst_TELEA = cv2.inpaint(img1, mask_img, 3, cv2.INPAINT_TELEA)
    dst_NS = cv2.inpaint(img1, mask_img, 3, cv2.INPAINT_NS)
    img1 = img_plt(img1)
    dst_TELEA = img_plt(dst_TELEA)
    dst_NS = img_plt(dst_NS)
    plt.figure(1)
    plt.xlabel("Image patching")
    plt.subplot(221),plt.xlabel("degraded image"),plt.imshow(img1)
    plt.subplot(222), plt.xlabel("mask image"), plt.imshow(mask_img,cmap = 'gray')
    plt.subplot(223),plt.xlabel("TELEA"),plt.imshow(dst_TELEA)
    plt.subplot(224), plt.xlabel("NS"), plt.imshow(dst_NS)


    img = cv2.imread('lena-color.jpg')
    img = img_plt(img)
    plt.figure(2)
    plt.imshow(img)  # 显示原图像
    plt.show()