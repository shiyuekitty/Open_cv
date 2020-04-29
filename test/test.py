import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


#########################  template process  ########################

##   average-neighborhood
##   template convolution
def template_convolution(img,temp):
    r,c = img.shape
    h1,v1 = temp.shape
    h = math.floor(h1 / 2)  ## threshold of row
    v = math.floor(v1 / 2)  ## threshold ofColumn
    N = sum(sum(temp))
    new_img = np.zeros((r,c))
    for i in range(h,r-h):
        for j in range(v,c-v):
            if N !=0:
                new_img[i][j] = abs(np.sum(img[i-h:i+h+1,j-v:j+v+1] * temp))/N
            else:
                new_img[i][j] = abs(np.sum(img[i-h:i+h+1,j-v:j+v+1] * temp))
    return new_img

if __name__ == '__main__':
    img = cv2.imread('../image/lena-gray.png', -1)
    ##   define the template
    t_3 = np.ones((3,3))
    t_11 = np.ones((11,11))
    t_Gaussian = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,5,4,1]])
    t_Lap = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    # new_img3 = template_convolution(img,t_3)
    # new_img11 = template_convolution(img,t_11)
    new_imgG = template_convolution(img,t_Gaussian)
    new_imgL = template_convolution(img,t_Lap)
    plt.subplot(231),plt.imshow(img,cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(new_imgG,cmap = 'gray')
    plt.title('Gaussian Average'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(new_imgL,cmap = 'gray')
    plt.title('Laplace sharp'), plt.xticks([]), plt.yticks([])
    plt.show()
##   template_sort for max\min\median\mean
##   flag = None is mean, =1 is max, =-1 is min =0 is median
##   temp is the size of the template(row and column are euqal)
# def template_sort(img,temp,flag=None):
#     r,c = img.shape
#     h = math.floor(temp/2)  ## threshold of row and column
#     new_img = np.zeros((r,c))
#     for i in range(h,r-h):
#         for j in range(h,c-h):
#             if flag == None:
#                 new_img[i][j] = np.mean(img[i-h:i+h+1,j-h:j+h+1])
#             elif flag == 1:
#                 new_img[i][j] = np.max(img[i-h:i+h+1,j-h:j+h+1])
#             elif flag == -1:
#                 new_img[i][j] = np.min(img[i-h:i+h+1,j-h:j+h+1])
#             elif flag == 0:
#                 new_img[i][j] = np.median(img[i-h:i+h+1,j-h:j+h+1])
#     return new_img
